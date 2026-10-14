# hik-person-py

HIKROBOT（海康）工业相机 → YOLO 人体检测 → [Rerun](https://rerun.io) 可视化。
取流直接用 MVS SDK。

**参数都在 [`config.toml`](config.toml) 里，代码里不写死。** 命令行只做临时覆盖。

## 快速开始

```bash
uv sync                              # torch/cv2 命中 uv 缓存，只下载 rerun-sdk

uv run hik-person --list             # 枚举相机
uv run hik-person --show-config      # 打印合并后的生效参数
uv run hik-person                    # 按 config.toml 跑（开 rerun 窗口）
uv run hik-person --duration 30      # 临时覆盖某一项
uv run hik-person --no-rerun --rerun-save out.rrd --max-frames 300
uv run hik-person --task pose --weights models/yolo26n-pose.pt   # 试姿态估计
```

## 参数文件

优先级：**命令行 > `config.toml` > 代码里的兜底默认值**。用 `--config PATH` 换文件。

```toml
[camera]
serial = "DB1505178"          # 本机 MV-CB016-10UC-S
camera_fps = 30.0             # 采集帧率上限
brightness = 110.0            # 测光目标灰度；0 = 关测光
exposure_us = 8000.0          # 测光起始值；0 = 用设备当前值
gain_db = -1.0                # -1 = 用设备当前值（0 dB 是合法最小值）
feature = ["ADCBitDepth=Bits_8"]   # 任意 GenICam 特性

[detector]
# detect = 人体检测框；pose = 关键点/骨架（切任务时 weights 也要一起改）
task = "detect"
weights = "models/best.pt"    # 相对路径按项目根解析；pose 用 models/yolo26n-pose.pt
conf = 0.35
imgsz = 640
kpt_conf = 0.5                # 仅 pose：关键点置信度阈值

[run]
detect_hz = 15.0
save_dir = ""

[rerun]
enabled = true
save = ""                     # 非空则写 .rrd 且不弹窗口
```

写错的键名**会警告**而不是被静默忽略（参数文件最怕"改了没生效还不知道"）：

```
$ uv run hik-person --config /tmp/typo.toml --show-config
warning: [camera].camera_fps 不合法（需要数字，收到 'fast'），已忽略
warning: 未知配置项 [camera].serail，已忽略
warning: 未知配置项 [rerun].enable，已忽略
```

命令行覆盖（`--help` 有全部）：`--serial --index --camera-fps --exposure-us --gain-db
--brightness --brightness-tolerance --exposure-fps-headroom --read-timeout-ms --feature
--weights --conf --iou --imgsz --device --detect-hz --duration --max-frames --save-dir
--rerun/--no-rerun --rerun-save`。收到命令行没给的项一律保留配置文件/兜底值，所以
`--gain-db 0` 和"没给增益"是两件事。

## 相机

`--list` 会列出全部设备；`--index` 只在**物理**相机里数，虚拟相机不占号：

```
[0] usb   MV-CS016-10UC    sn=DA2360141
[1] usb   MV-CB016-10UC-S  sn=DB1505178
[2] usb   MV-CS016-10UC    sn=Vir62953321  (虚拟, 打不开)
```

`--feature NAME=VALUE` 可以写任意 GenICam 特性（可重复），按值的形态自动挑写入方式
（布尔/整型/浮点/枚举），全失败时把每种尝试的错误码一起报出来：

```bash
uv run hik-person --feature ADCBitDepth=Bits_8 --camera-fps 249 --brightness 0
```

这台 **MV-CB016-10UC-S** 的实测 A/B（同一台、同一组其它参数，只换这一项）：

| `ADCBitDepth` | `resulting_fps` |
|---|---|
| `Bits_8` | **248.89** |
| `Bits_12` | **165.37** |

也就是 1440×1080 USB3 想跑到 249 fps 必须走 8 bit 读数；`Bits_12` 会卡在 165.4。
`Bits_8` 不改变载荷（BayerRG8 本来就是 8 bit/像素），代价是动态范围。

## 任务：检测框 or 关键点

`[detector].task` 决定跑哪条任务线，两者共用取流、测光、人数统计与 rerun 布局：

| `task` | `weights` | 输出 |
|---|---|---|
| `detect`（默认） | `models/best.pt` | 人体检测框 |
| `pose` | `models/yolo26n-pose.pt` | 检测框 + 17 个 COCO 关键点与骨架 |

```bash
uv run hik-person --task pose --weights models/yolo26n-pose.pt
```

姿态线用 ultralytics 的 `yolo26n-pose`（`kpt_shape=[17,3]`，单类 person）。骨架表在
`pose.SKELETON_17`（19 根骨头，0-based）——注意 ultralytics 自己的 skeleton 是 **1-based**
（那边用 `sk[0]-1` 取），抄表时很容易多一根或少一根，所以单元测试里卡了骨头数、索引范围、
每个关键点至少出现在一根骨头里。

## 曝光与增益（测光）

默认先做一次**闭环测光**：量画面平均灰度，自己调曝光和增益。

为什么不用相机的 `ExposureAuto`：这台相机**没有 `Brightness`（目标灰度）特性**，
相机的自动曝光无法指定目标值。所以策略是**先动曝光，曝光顶到上限后再用增益补**
（增益涨噪声，曝光不涨）：

```
测光: 目标灰度 110，曝光上限 30000 us（= 0.9 / camera_fps 30）
  测光  1: 曝光   8000.0 us  增益 16.98 dB  ->  灰度  172.9  (目标 110)
  测光  2: 曝光   5089.3 us  增益 16.98 dB  ->  灰度  138.6  (目标 110)
  测光  3: 曝光   4040.4 us  增益 16.98 dB  ->  灰度  118.8  (目标 110)
  -> 曝光 4040 us，增益 16.98 dB，灰度 118.8
```

* **曝光上限与帧率绑定**：`exposure_fps_headroom / camera_fps`，所以调亮不会偷偷把帧率带下去。
  要高帧率就得拿亮度换，程序会如实警告并停止，而不是假装成功：
  ```
  警告: 没达到目标灰度（14.6 / 目标 110）：曝光与增益都已到边界。光线不够，
        或者 camera_fps 太高压住了曝光上限
  ```
* 想自己上手动值就 `brightness = 0` + `exposure_us` / `gain_db`。
* **相机会记住上次的曝光/增益**（存在相机里，不随进程退出丢），所以"起始值"取决于上一轮
  跑成什么样；要干净复现就显式给值。

## 颜色链路

相机原生输出 `BayerRG8`。本包内部统一是 **BGR**（OpenCV 的默认顺序），靠两条约定保证不串色：

1. **转换交给 SDK**：`MV_CC_SetBayerCvtQuality(1)` 配高质量插值，再用
   `MV_CC_ConvertPixelTypeEx` 转到 `PixelType_Gvsp_BGR8_Packed`（= 35127317）。走 SDK
   就不会碰到 Bayer 排布的命名歧义。
2. **进 rerun 前换一次 RGB**：`rr.Image` 把三通道数据按 RGB 解读，直接把 BGR 喂进去就会
   红蓝互换，见 `colors.bgr_to_rgb()`。

如果要自己解 Bayer，注意 MVS/GenICam 对 Bayer 排布的命名与 OpenCV **相差一个像素**：
MVS 的 `BayerRG8` 必须用 `cv2.COLOR_BayerBG2BGR`（用 `COLOR_BayerRG2BGR` 会把 R/B 对调，
以 SDK 输出为参考：BG 的 corr 0.998，RG 0.988）。这个判据的区分度取决于画面里的颜色 ——
对着几乎中性灰的场景，R/B 互换在通道均值上只差 0.03；结论性最强的是单元测试里那张合成
马赛克（R=200 / G=50 / B=100）。

验证：

```bash
uv run python tools/check_colors.py --save-dir /tmp/hik-color-check
```

## 可视化（rerun）

布局由 blueprint 显式指定，不依赖 rerun 的自动推断：

| 实体 | 内容 |
|---|---|
| `camera/annotated` | 标注图：模型画的框（pose 时含骨架），不在像素上额外烧字 |
| `camera/annotated/boxes` | 同一批框的 `Boxes2D`，可悬停看标签与置信度 |
| `camera/annotated/keypoints` | 仅 pose：关键点 `Points2D`，标签是 `nose`/`left_shoulder` 等 |
| `camera/annotated/skeleton` | 仅 pose：骨架 `LineStrips2D`（两端都过 `kpt_conf` 才连） |
| `person/count` | 人数曲线（右侧 TimeSeriesView） |
| `person/events` | 人数变化日志（带各框置信度） |

* 弹窗模式下 viewer 是**独立进程**，会继承当前终端；在脚本/管道里跑就用 `--rerun-save`，
  别让 viewer 把 stdout 占住。
* `--rerun-save` 每帧写一张 1440×1080 的图，实测约 **8.5 MB/帧**（15 Hz 就是 128 MB/s）。
  长跑别这么干：用 `--no-rerun --save-dir` 只留检测到人的帧（最多 1 张/秒，见
  `save_min_interval`）。

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
uv run pytest                  # 69 个单元测试，都不需要相机
uv run python tools/bench.py   # 分环节计时（等帧 / 转换 / 推理）
```

工具（`tools/*.py`）同样读 `config.toml`，也可以 `--config PATH` 指别的文件。

本机 `PYTHONPATH` 里还带着别的项目的 `site-packages`（`~/.bashrc` 里 source 了 ROS 的
setup），pytest 会自动加载那些插件并在收集阶段崩掉。`pyproject.toml` 里用
`-p no:launch_testing -p no:launch_ros` 显式禁掉了它们；`-p no:<name>` 对不存在的插件
是空操作，所以在别的机器上也无害。更彻底的做法是 `PYTHONPATH= uv run pytest`。

## 结构

```
config.toml              所有参数都在这里
src/hik_person/
├── config.py    config.toml 读取 / 命令行覆盖 / 路径解析
├── colors.py    像素格式映射 + Bayer/BGR/RGB 转换
├── mvs.py       MVS 运行时环境、枚举、取流、测光、特性改写
├── detect.py    ultralytics YOLO 人体检测
├── pose.py      ultralytics YOLO 姿态（COCO-17 骨架）
├── viz.py       rerun 记录与布局（Image / Boxes2D / Points2D / LineStrips2D / Scalars）
└── __main__.py  CLI，只负责把参数接起来
tools/
├── check_colors.py  颜色链路对比验证
└── bench.py         分环节计时
models/
├── best.pt          自训练的人体检测权重
└── yolo26n-pose.pt  ultralytics 姿态权重
```

## 实测性能（1440×1080 + RTX 4060 Laptop）

`uv run python tools/bench.py` 可复现：

| 环节 | 耗时 |
|---|---|
| SDK 转 BGR（纯计算，不含等帧） | 1.1 ms |
| BGR → RGB（rerun 需要） | 0.08 ms |
| YOLO 推理（`imgsz 640`, `cuda:0`） | 4.3 ms |
| 取一帧（含等帧） | 33 ms = 相机帧间隔 |

单帧计算约 5.5 ms，瓶颈在相机节拍，不在模型。主循环按 `detect_hz` 限速，且只处理最新的
帧（不积压）：相机开到 249 fps 也只按 15 Hz 推理。

`camera_fps` 建议取 `detect_hz` 的 2 倍以上（例如 30 / 15，实测端到端 13~14 fps、每两帧
推理一次）。两者相等时帧到达时刻正好压在周期边界上，所以留了 `rate_tolerance` 容差。

## 故障排查

* `MV_CC_OpenDevice failed: 0x80000203` —— 相机被别的进程占用（MVS 对 USB3 不强制互斥）：
  关掉 MVS 客户端，或停掉另一个正在取流的进程。
* 枚举不到设备 —— 检查 USB3 线缆，以及 `/opt/MVS` 驱动与权限。
* 画面偏暗 / 检测不出人 —— 看测光输出：报"曝光与增益都已到边界"就是现场光线不够，或者
  `camera_fps` 定得太高压住了曝光上限；把 `camera_fps` 调低再跑，或把 `brightness` 降到
  这个场景能达到的值。
* 颜色不对（红蓝互换）—— 检查是否绕过了 `colors.bgr_to_rgb()`。快速自检：对着明显偏红的
  物体比较 `frame.bgr[..., 2].mean()`（R）与 `frame.bgr[..., 0].mean()`（B），R 应更大。
* 改了 `config.toml` 好像没生效 —— 先 `--show-config` 看合并后的值，再确认写到正确的段里
  （键名写错会有 warning）。
* rerun 窗口没起来 —— 加 `--no-rerun` 先确认取流与检测本身正常。

## 许可证

AGPL-3.0-or-later，全文见 [LICENSE](LICENSE)。

- 依赖 [ultralytics](https://github.com/ultralytics/ultralytics)（AGPL-3.0），
  本项目因此采用同一许可证。对外分发、或以网络服务形式提供时，需按 AGPL 提供源码。
- `models/best.pt` 为本项目使用的检测权重，随本仓库一同提供。
- 运行需要海康 **MVS SDK**（`/opt/MVS`）。它是海康的专有软件，不在本仓库内，
  也不由本许可证覆盖；使用它需遵守海康自己的许可条款。
