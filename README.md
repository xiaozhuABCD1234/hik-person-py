# hik-person-py

HIKROBOT（海康）工业相机 → YOLO 人体检测 → [Rerun](https://rerun.io) 可视化。
取流直接用 MVS SDK，不需要相机厂商以外的任何中间件。

## 快速开始

```bash
uv sync                                  # torch/cv2 命中 uv 缓存，只下载 rerun-sdk

uv run hik-person --list                 # 枚举相机
uv run hik-person --duration 30          # 开 rerun 窗口跑 30 秒
uv run hik-person --no-rerun --max-frames 300 --save-dir ~/Pictures/hik-person
uv run python -m hik_person --device cpu --imgsz 480   # 也支持 -m 形式
uv run hik-person --rerun-save /tmp/hik.rrd --max-frames 100   # 不弹窗口，落盘
rerun /tmp/hik.rrd                                    # 事后回放
```

`--device` 留空时自动选择：有 CUDA 用 `cuda:0`，否则 `cpu`。
`--list` 会把虚拟相机一起列出来（本机是 `[1] ... sn=Vir62953321`），
用 `--serial` 选物理那台。

注意：弹窗模式（默认）下 rerun viewer 是另一个进程，会继承当前终端；
在脚本/管道里跑就用 `--rerun-save` 落盘，别让 viewer 把 stdout 占住。

## 运行参数

| 参数 | 默认 | 说明 |
|---|---|---|
| `--serial` | `DA2360141` | 相机序列号 |
| `--index` | 0 | 没有 `--serial` 时按枚举序号选 |
| `--weights` | `models/best.pt` | YOLO 权重 |
| `--conf` / `--iou` / `--imgsz` | 0.35 / 0.50 / 640 | 检测参数 |
| `--device` | 自动 | `''` / `cuda:0` / `cpu` |
| `--detect-hz` | 15 | 推理与可视化频率；多余的帧只取流不推理 |
| `--camera-fps` | 15 | 相机帧率上限，0 = 保持设备设置（该机型上限 249 fps） |
| `--brightness` | 110 | 测光目标（平均灰度 0-255）。0 = 关闭测光，按下面的值直接用 |
| `--exposure-us` | 8000 | 曝光 us。0 = 用设备当前值；测光开启时作为起始值 |
| `--gain-db` | -1 | 增益 dB（该机型 0~17）。**0 是合法值**；-1 = 用设备当前值 |
| `--no-rerun` | – | 不开 rerun，只在终端打印结果 |
| `--rerun-save` | – | 把记录写成 `.rrd` 而不弹窗口（无桌面 / 留证据） |
| `--save-dir` | – | 检测到人时保存标注图（最多 1 张/秒） |
| `--duration` / `--max-frames` | 0 | 跑够时长/帧数后退出，0 = 不限 |

## 曝光与增益（测光）

默认会先做一次**闭环测光**：自己量画面平均灰度，自己调曝光和增益。

为什么不用相机的 `ExposureAuto`：这台 MV-CS016-10UC **没有 `Brightness`（目标灰度）特性**
（探测过，取不到），相机的自动曝光无法指定目标值，只能用它的内部目标。

策略是**先加曝光，曝光顶到上限后再用增益补**（增益涨噪声，曝光不涨）：

```
测光: 目标灰度 110，曝光上限 60000 us（受 --camera-fps 15 约束）
  测光  1: 曝光   8000.0 us  增益  0.00 dB  ->  灰度    4.4  (目标 110)
  测光  2: 曝光  60000.0 us  增益 10.38 dB  ->  灰度  119.3  (目标 110)
  -> 曝光 60000 us，增益 10.38 dB，灰度 119.3
```

几个要点：

* **曝光上限与帧率绑定**：`上限 = 0.9 / --camera-fps`（`--camera-fps 15` → 60 ms），
  所以调曝光不会偷偷把帧率带下去。想要高帧率就拿亮度换：
  `--camera-fps 249` → 上限 3614 us，此时增益顶到 16.98 dB 也可能不够，程序会如实警告
  并停下（不会白跑迭代）：

  ```
  警告: 没达到目标灰度（14.6 / 目标 110）：曝光与增益都已到边界。光线不够，
        或者 --camera-fps 太高压住了曝光上限
  ```

* 想自己上手动值就 `--brightness 0 --exposure-us 5000 --gain-db 8`。
* **相机会记住上次的曝光/增益**（MVS 存在相机里，不随进程退出丢），
  所以“起始值”取决于上一轮跑成什么样；要干净复现就显式传 `--gain-db 0`。
* 目标灰度 110 是针对这个暗场景的默认值；光线正常时测光通常 1~2 次迭代就收敛。

已知限制：这台相机出厂 `ADCBitDepth=Bits_12`，实测请求 249 fps 只能得到 165.4 fps
（8 bit 读数才能到 249）。本项目还没暴露 BitDepth 这类通用特性改写，需要时可以加。

## 颜色链路

相机原生输出 `BayerRG8`（1440x1080）。本包所有帧在内部统一是 **BGR** 顺序
（OpenCV 的默认顺序），靠两条约定保证不串色：

1. **转换交给 SDK**。`MV_CC_SetBayerCvtQuality(1)` 配高质量插值，
   再用 `MV_CC_ConvertPixelTypeEx` 转到 `PixelType_Gvsp_BGR8_Packed`（= 35127317）。
   走 SDK 就不会碰到 Bayer 排布的命名歧义。
2. **若要自己解 Bayer，注意命名差一个像素**。MVS/GenICam 对 Bayer 排布的命名与
   OpenCV 相差一个像素：MVS 的 `BayerRG8` 必须用 `cv2.COLOR_BayerBG2BGR`。
   以 SDK 输出为参考，两者几乎是同一个东西（BayerBG 与它的 corr 0.998，
   而 BayerRG 把 R/B 对调，corr 0.988）。

   ⚠️ 这个实测的**区分度取决于画面里的颜色**：对着几乎中性灰的场景，R/B 互换在
   通道均值上只差 0.03（104.93 ↔ 104.96），光看均值看不出来。
   结论性最强的是单元测试里那张合成马赛克（R=200 / G=50 / B=100，分得开，
   见 `tests/test_colors.py`）；`tools/check_colors.py` 只是现场 sanity check，
   对着彩色物体或白纸上的红物体会明显得多。

3. **进 rerun 前换一次 RGB**。`rr.Image` 把三通道数据按 RGB 解读，
   直接把 BGR 喂进去就会红蓝互换，见 `colors.bgr_to_rgb()`。

验证：

```bash
uv run python tools/check_colors.py --save-dir /tmp/hik-color-check
# 打印三条路径的 B/G/R 均值与相关系数，并存 3 张图供肉眼确认
```

## 不需要任何 .sh

MVS SDK 的像素转换 / GenICam producer 库是 `dlopen` 时才加载的，而 glibc 只在
**进程启动时**读一次 `LD_LIBRARY_PATH`，所以环境变量必须在那之前就位。
`mvs.ensure_runtime_environment()` 在环境不满足时用 `os.execve` 换一个全新进程，
加载器自然能看到正确的搜索路径 —— 直接 `uv run` 即可，不需要 shell 包装脚本。
（要手动接管时也可以 `source /opt/MVS/bin/set_env_path.sh`。）

## 开发

```bash
uv run ruff check .            # lint
uv run ruff format --check .   # 格式
uv run pytest                  # 单元测试，不需要相机
uv run python tools/bench.py   # 分环节计时（取流 / 转换 / 推理）
```

本机 `PYTHONPATH` 里还带着别的项目的 `site-packages`（`~/.bashrc` 里 source 了
ROS 的 setup），pytest 会自动加载那些插件并在收集阶段崩掉。`pyproject.toml` 里用
`-p no:launch_testing -p no:launch_ros` 显式禁掉了它们；`-p no:<name>` 对不存在的
插件是空操作，所以在别的机器上也无害。更彻底的做法是 `PYTHONPATH= uv run pytest`。

## 结构

```
src/hik_person/
├── colors.py   像素格式映射 + Bayer/BGR/RGB 转换
├── mvs.py      MVS 运行时环境、枚举、取流、SDK 内 BGR 转换
├── detect.py   ultralytics YOLO 人体检测
├── viz.py      rerun 记录（Image / Boxes2D / Scalars / TextLog）
└── __main__.py CLI 与主循环
tools/
├── check_colors.py  颜色链路对比验证
└── bench.py         分环节计时
models/best.pt       训练好的权重
```

## 实测性能（MV-CS016-10UC 1440x1080 + RTX 4060 Laptop）

`uv run python tools/bench.py` 可复现：

| 环节 | 耗时 |
|---|---|
| SDK 转 BGR（纯计算，不含等帧） | 1.1 ms |
| BGR → RGB（rerun 需要） | 0.08 ms |
| YOLO 推理（`imgsz 640`, `cuda:0`） | 4.3 ms |
| 取一帧（含等帧） | 33 ms = 相机帧间隔 |

单帧计算总共约 5.5 ms，所以瓶颈在相机节拍，不在模型。主循环按 `--detect-hz` 限速，
并且只处理最新的帧（不会积压）：相机开到 249 fps 也只按 15 Hz 推理。

`--camera-fps` 建议取 `--detect-hz` 的 2 倍以上，例如 `--camera-fps 30 --detect-hz 15`
（实测约 13 fps 稳定、每两帧推理一次）。两者相等时帧到达时刻正好压在周期边界上，
代码里留了 10% 容差来避开这个抖动。

## 许可证

AGPL-3.0-or-later，全文见 [LICENSE](LICENSE)。

- 依赖 [ultralytics](https://github.com/ultralytics/ultralytics)（AGPL-3.0），
  本项目因此采用同一许可证。对外分发、或以网络服务形式提供时，需按 AGPL 提供源码。
- `models/best.pt` 为本项目使用的检测权重，随本仓库一同提供。
- 运行需要海康 **MVS SDK**（`/opt/MVS`）。它是海康的专有软件，不在本仓库内，
  也不由本许可证覆盖；使用它需遵守海康自己的许可条款。

## 故障排查

* `MV_CC_OpenDevice failed` —— 相机被别的进程占用（MVS 对 USB3 不强制互斥）：
  关掉 MVS 客户端，或停掉另一个正在取流的进程。
* 枚举不到设备 —— 检查 USB3 线缆，以及 `/opt/MVS` 驱动与权限。
* 颜色不对（红蓝互换）—— 检查是否绕过了 `colors.bgr_to_rgb()`。
  快速自检：对着明显偏红的物体比较 `frame.bgr[..., 2].mean()`（R）与
  `frame.bgr[..., 0].mean()`（B），R 应更大。
* rerun 窗口没起来 —— 加 `--no-rerun` 先确认取流与检测本身正常。
* 画面偏暗 / 检测不出人 —— 看测光输出：如果报了“曝光与增益都已到边界”，就是现场光线不够，
  或 `--camera-fps` 定得太高压住了曝光上限；把 `--camera-fps` 降低（例如 15）再跑，
  或把 `--brightness` 调低到这个场景能达到的值。
* `MV_CC_OpenDevice failed: 0x80000203` —— 相机已被占用（例如上一次运行还在、或开了 MVS 客户端）。
