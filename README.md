# hik-person-py

HIKROBOT（海康）工业相机 → YOLO 姿态（骨架）→ [Rerun](https://rerun.io) 可视化。
取流直接用 MVS SDK。

**参数都在 [`config.toml`](config.toml) 里，代码里不写死。** 命令行只做临时覆盖。

## 快速开始

```bash
uv sync                              # torch/cv2 命中 uv 缓存，只下载 rerun-sdk

uv run hik-person --list             # 枚举相机
uv run hik-person --show-config      # 打印合并后的生效参数
uv run hik-person                    # 按 config.toml 跑（开 rerun 窗口）
uv run hik-person --duration 30      # 临时覆盖某一项
uv run hik-person --no-rerun --max-frames 300     # 只看结果，不开窗口不写文件
uv run hik-person --weights models/yolo26n-pose.pt   # 换 n 档姿态权重（快一倍）
uv run hik-person --no-pose3d        # 关掉 3D 小窗，不加载深度模型
```

## 参数文件

优先级：**命令行 > `config.toml` > 代码里的兜底默认值**。用 `--config PATH` 换文件。

```toml
[camera]
serial = "DA2360141"          # 本机 MV-CS016-10UC；另一台用 --serial 指
camera_fps = 30.0             # 采集帧率上限
exposure_mode = "sdk"         # sdk = 相机自动曝光/增益；manual = 用下面的固定值
brightness = 110.0            # AE 目标灰度（写进相机的 Brightness 特性）
auto_gain_max_db = 16.98      # AE 允许的最大增益
exposure_start_us = 8000.0    # 开 AE 前先钉住的曝光起点（0 = 用相机现存值）
exposure_us = 8000.0          # manual 模式：0 = 用设备当前值
gain_db = -1.0                # manual 模式：-1 = 用设备当前值（0 dB 是合法最小值）
feature = ["ADCBitDepth=Bits_8"]   # 任意 GenICam 特性（写别的机型的符号名也会自动改写）

[detector]
# 只有一条任务线：姿态模型给检测框 + 17 个 COCO 关键点与骨架
weights = "models/yolo26m-pose.pt"   # 相对路径按项目根解析；n/s/m/l/x 同一 API
conf = 0.35
imgsz = 640
kpt_conf = 0.5                # 关键点置信度阈值

[run]
detect_hz = 15.0
save_dir = ""

[rerun]
enabled = true
save = ""                     # 非空则写 .rrd 且不弹窗口
show_values = true            # 右列顶部的数值行（每人一行）
show_metrics = true           # 右列的指标曲线（速度、摔倒）
fall_light_flash = 2.0        # 摔倒报警灯的闪烁频率 Hz（0 = 常亮）
show_count = false            # 右列的人数曲线（数据一直在，默认不占格子）

[pose3d]
# 单目 3D 骨架：yolo26*-depth 给 17 个关键点补一个 z，rerun 右下角开个小窗画 3D
enabled = true
weights = "models/yolo26m-depth.pt"   # n/s/m/l/x 同一 API；m 比 s 更准
imgsz = 640                   # 深度推理输入边长（输出会自动缩回原分辨率）
hz = 15.0                     # 3D 骨架更新频率（深度模型比 pose 贵，不用跟满）
fov_deg = 60.0                # 相机没标定，按水平视场角估内参
depth_view = true             # 左列再放一块估计深度图（Turbo 伪彩）
eye = [0.0, 0.0, -1.5]        # 固定 3D 视角：眼位 / 视点 / 上方向（相机坐标系，米）
look_at = [0.0, 0.0, 2.5]
eye_up = [0.0, -1.0, 0.0]
kalman = true                 # 3D 关键点卡尔曼滤波
kalman_process = 10.0         # 过程噪声（m/s²）：越大越跟手
kalman_measurement = 0.1      # 观测噪声（m）：越大越平滑

[track]
iou_threshold = 0.3           # 跨帧关联的匹配线：越高越容易把同一个人判成两个
match_max_distance = 0.0      # >0 时配对还要满足 3D 距离（米），防交叉换 id；0 = 只看 IoU
max_missed = 5                # 连续漏检这么多帧后删轨迹（期间位置走卡尔曼预测）
max_missed_kpt = 3            # 单个关键点连续丢这么多帧就不再画
trail_length = 300            # 轨迹点数上限（0 = 不留轨迹）
trail_interval = 0.1          # 轨迹采样间隔（秒）
speed_smoothing = 0.3         # 速度指数平滑时间常数（秒），0 = 不平滑
speed_deadband = 0.15         # 低于它当"静止"（只影响显示与状态标签）

[fall]
enabled = true                # 摔倒检测：三步判据（冲击 -> 躺倒 -> 持续）
impact_speed = 1.0            # m/s：向下速度超过它算冲击
drop_height = 0.35            # m：窗口内从高点掉这么多也算冲击
impact_window = 0.7           # s：下沉判据的窗口
lying_angle = 60.0            # 度：躯干与"上"方向夹角超过它算躺倒
lying_height = 0.35           # m：低于站立参考这么多也算躺倒（角度被遮时兜底）
lying_time = 1.2              # s：躺倒连续保持多久才确认报警
confirm_window = 3.0          # s：冲击后多久内确认有效，超时当没事
recover_angle = 35.0          # 度：恢复到直立
recover_time = 1.0            # s：直立保持多久才解除报警

[world]
enabled = false               # 相机系 -> 世界（地面）系；没量过安装角度就别开
height = 1.0                  # 光心离地高度（米）
pitch_deg = 0.0               # 镜头下俯角度（度，正 = 向下）
yaw_deg = 0.0                 # 相机朝向与世界 X 的夹角（度）；装得周正就是 0
roll_deg = 0.0                # 绕光轴的旋转（度）
view = false                  # 右下角再开一个世界系 3D 视图（相机系那个照旧）

[output]
jsonl = ""                    # 非空则把人写成 JSON Lines（追加写，一行一帧）
hz = 5.0                      # 写 JSONL 的频率（Hz）
```

写错的键名**会警告**而不是被静默忽略（参数文件最怕"改了没生效还不知道"）：

```
$ uv run hik-person --config /tmp/typo.toml --show-config
warning: [camera].camera_fps 不合法（需要数字，收到 'fast'），已忽略
warning: 未知配置项 [camera].serail，已忽略
warning: 未知配置项 [rerun].enable，已忽略
```

命令行覆盖（`--help` 有全部）：`--serial --index --camera-fps --exposure-us --gain-db
--brightness --exposure-start-us --exposure-fps-headroom --read-timeout-ms --feature
--weights --conf --imgsz --device --kpt-conf --detect-hz --duration --max-frames --save-dir
--rerun/--no-rerun --rerun-save --show-values/--no-show-values --show-metrics --show-count
--fall/--no-fall --fall-light-flash
--show-speed/--no-show-speed
--track-iou-threshold --track-match-max-distance --track-max-missed --track-trail-length
--world/--no-world --world-height --world-pitch --world-yaw --world-roll --world-view
--jsonl --output-hz`。收到命令行没给的项一律保留配置文件/兜底值，所以
`--gain-db 0` 和"没给增益"是两件事。

## 相机

`--list` 会列出全部设备；`--index` 只在**物理**相机里数，虚拟相机不占号：

```
[0] usb   MV-CS016-10UC    sn=DA2360141
[1] usb   MV-CB016-10UC-S  sn=DB1505178
[2] usb   MV-CS016-10UC    sn=Vir62953321  (虚拟, 打不开)
```
`--feature NAME=VALUE` 可以写任意 GenICam 特性（可重复），按值的形态自动挑写入方式
（布尔/整型/浮点/枚举）。写不进去也不能只报一个错误码 —— 它会探清楚这个特性是什么
类型、认哪些符号。

```bash
uv run hik-person --feature ADCBitDepth=Bits_8 --camera-fps 249 --brightness 0
```

### `ADCBitDepth`：8 bit 读数解锁 249 fps

想跑到 249 fps 必须走 8 bit 读数；12 bit 会卡在 165.4（1440×1080 USB3 的读出上限）。
8 bit 不改变载荷（`BayerRG8` 本来就是 8 bit/像素），代价是动态范围。两台机型实测：

> 量这个要把曝光压短（用 `MVCC_FLOATVALUE` 固定到 500 us）。自动曝光开着时
> `ResultingFrameRate` 是 `min(读出上限, 1/曝光)`，量到的是曝光不是读出能力 ——
> 开着 AE 量出来的是 168 fps（≈ 1/6 ms），很容易误判成"这台机型 8 bit 也没用"。

| `ADCBitDepth` | `MV-CB016-10UC-S` | `MV-CS016-10UC` |
|---|---|---|
| 8 bit（数值 0） | **249.10 fps** | **249.10 fps** |
| 12 bit（数值 3） | 165.37 fps | 165.37 fps |

#### 符号名随机型不同 —— 但程序会自己换

同一个设置，两个型号的符号名不一样（数值都是 0/3）：

| 机型 | 8 bit | 12 bit |
|---|---|---|
| `MV-CB016-10UC-S` | `Bits_8` | `Bits_12` |
| `MV-CS016-10UC` | `ADCBitDepth_8` | `ADCBitDepth_12` |

所以写错机型的名字不再需要手改 config：能确定是**同一个设置**就自动换成这台相机认的
名字，并明确打印实际写了什么（免得不声不响地换掉用户的意图）：

```
$ uv run hik-person --feature ADCBitDepth=ADCBitDepth_8     # config 是 CB 机的名字，现在插的是 CS 机
feature: ADCBitDepth=ADCBitDepth_8 (以 enum 写入；这台相机不叫 ADCBitDepth_8，
已自动换成同一个设置的另一个名字)
```

自动改写故意做得很保守，两个条件都要求**名字里的数字不变**（8 bit 与 12 bit 不能搞混）：

1. 去掉节点名后两边完全一样（`Auto` 对 `ADCBitDepth_Auto`）；
2. 残部有一头是另一头的后缀，且尾部数字一样（`Bits_8` 对 `ADCBitDepth_8`）。

命中多个候选（`Foo_8`/`Bar_8` 都对得上）、或数字对不上（`Bits_10` 对 `_8`）就不改，
而是报错让人自己拍板：

```
$ uv run hik-person --feature ADCBitDepth=Bits_10
error: 设置特性 ADCBitDepth=Bits_10 失败（按 enum=0x80000004 依次试过）：ADCBitDepth 是枚举特性，
这台相机只认这些符号：ADCBitDepth_8 / ADCBitDepth_12（符号名随机型不同，别照抄别的相机的配置）
```

不依赖符号名的话，也可以直接写数值（`ADCBitDepth=0` = 8 bit），两台机型都认；
代价是看不出这是什么设置。

错误码的含义（程序探到类型后已经把常见情况分好了，可以直接对表）：

| 错误码 | 含义 | 常见原因 |
|---|---|---|
| `0x80000109` | `MV_E_GC_NODE_NOT_FOUND` | 特性名拼写错，或节点不是这个类型 |
| `0x80000004` | `MV_E_PARAMETER` | **节点在，但符号名不在它的枚举表里** |
| `0x80000025` | `MV_E_PARAMETER_RANGE` | 数值超出该特性的范围 |

## 模型：YOLO 姿态

`[detector].weights` 指到哪个姿态权重就跑哪个：输出都是检测框 + 17 个 COCO 关键点与骨架，
差别只在精度和延迟（本机 RTX 4060 Laptop，`imgsz 640`）：

| `weights` | 延迟 | 说明 |
|---|---|---|
| `models/yolo26n-pose.pt` | 7.4 ms | n 档，最快 |
| `models/yolo26m-pose.pt`（默认） | 16.6 ms | m 档，精度明显更好，仍远低于 15 Hz 的 66 ms 预算 |

```bash
uv run hik-person --weights models/yolo26n-pose.pt
```

权重是 ultralytics 的 pose 任务（`kpt_shape=[17,3]`，单类 person）。骨架表在
`pose.SKELETON_17`（19 根骨头，0-based）——注意 ultralytics 自己的 skeleton 是 **1-based**
（那边用 `sk[0]-1` 取），抄表时很容易多一根或少一根，所以单元测试里卡了骨头数、索引范围、
每个关键点至少出现在一根骨头里。

## 曝光与增益

默认 `exposure_mode = "sdk"`：直接用**相机自己的**自动曝光与自动增益，程序不插手。

```
相机: Brightness = 110（int）
相机: AutoExposureTimeUpperLimit = 30000（int）
相机: AutoGainUpperLimit = 16.98（float）
曝光/增益: 相机自动（目标灰度 110，曝光上限 30000 us，增益上限 16.98 dB）
  -> 曝光 23140 us，增益 0.00 dB
camera: ... exposure_us=23140.0, gain_db=0.0, resulting_fps=30.0
```

三个特性要一起给对，否则会得到“又暗又是噪点”的结果：

| 特性 | 类型 | 作用 |
|---|---|---|
| `Brightness` | **Int** | AE 目标灰度（0-255） |
| `AutoExposureTimeUpperLimit` | **Int** | AE 曝光上限；不放开，AE 只能用增益变亮 |
| `AutoGainUpperLimit` | **Float** | AE 增益上限（dB） |

* **曝光上限 = `exposure_fps_headroom / camera_fps`**，默认 `0.9/30 s = 30000 us`：
  既让 AE 有足够的调亮余地，又不会把帧率悄悄带下去。
* 想自己上手动值就用 `exposure_mode = "manual"` + `exposure_us` / `gain_db`。
* **相机会记住上次的曝光/增益**（存在相机里，不随进程退出丢），换进程不一定回到同一状态。
  所以 `sdk` 模式下会先把自动关掉、把曝光钉到 `exposure_start_us`（默认 8000 us），
  再开 `Continuous` —— 换相机、或别人用 MVS 客户端动过这台相机，每轮的起跑线也一样。
  `exposure_start_us = 0` = 不重置，用相机现存值。
* 启动时那行 `-> 曝光 …` 是**等 AE 收敛后**才读的（连续几帧不动才算稳）：AE 是逐帧
  爬的，刚开 `Continuous` 就读会得到一个没意义的半路值（实测换相机后读到过 2697 us，
  而它实际一路爬到了 15000 us 上限）；等满帧数还不稳会标「未收敛」。

踩过的坑（都跟错类型有关，写错时 SDK 只给一个 `0x80000109`，看着像“不支持”）：

1. 这三项是 **Int / Int / Float**，不是统一的 Float。我一开始用 `GetFloatValue` 查
   `Brightness`，报错后误判“这台相机没有目标灰度特性”，于是写了个量灰度自己调曝光的
   闭环——而真正的问题只是类型用错，那个闭环反而把曝光拉到 30 ms + 增益拉满造成过曝。
   现在闭环已删掉。
2. 出厂 `AutoExposureTimeUpperLimit = 3828 us`（不是 30 ms），AE 被卡在这个天花板里，
   只好把增益顶到设备上限 16.98 dB（满噪点），画面还很暗（灰度 52）。把它放到 30000 us 后，
   AE 立刻改用曝光：**23140 us / 0.00 dB / 灰度 103**。

不确定相机到底支持什么、当前值是什么，就 dump 一份特性表（不需要开 MVS 客户端）：

```bash
uv run python -c "
from hik_person.mvs import Camera, load_sdk
with Camera(load_sdk(), serial='DB1505178') as c:
    print(c.cam.MV_CC_FeatureSave('/tmp/cam_features.mvcfg'))"
grep -iE 'auto|bright|exposure|gain' /tmp/cam_features.mvcfg
```

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

**rerun 只当显示器：默认不往磁盘写任何文件。** 只有显式给 `--rerun-save` 才会写 `.rrd`。
但 viewer 本身得把收到的数据留在内存里才能显示和回拖，这是它的工作原理，无法关闭；
能做的只有给它一个上限，到顶就丢最旧数据：

| 参数 | 默认 | 说明 |
|---|---|---|
| `[rerun].memory_limit` | `2GB` | viewer 内存上限（rerun 自己的默认是 `75%`，本机约 23 GB） |
| `[rerun].server_memory_limit` | `512MB` | 同进程的 gRPC 缓冲（给后到的 viewer 补数据） |
| `[rerun].jpeg_quality` | `75` | 图像送进 viewer 前先压过；`0` = 送原图 |

实测（1440×1080、15 Hz、同尺寸窗口）：

| 送什么 | viewer 内存增长 | 每帧 |
|---|---|---|
| `rr.Image` 原图 | **+70 MB/s**（= 15 Hz × 4.67 MB） | 4.67 MB |
| `.compress(jpeg_quality=75)` | **+4 MB/s** | ~0.3 MB |

所以不改这两个默认值，跑几分钟 viewer 就会涨到几个 GB；而 rerun 落盘的 `.rrd` 看着小，
是因为文件里又压了一道 —— **占内存的是 viewer 里那份解压后的数据**。

### 框只画一遍（已修）

以前一帧的信息被画了两遍、传了两遍：模型 `result.plot()` 先把框/骨架烧进整图（1440×1080
的绘制 + 整图拷贝），rerun 又用 `Boxes2D`/`Points2D`/`LineStrips2D` 重画一遍。现在
`camera/image` 只送相机原图，框/骨架一律由 rerun 结构化地画；`result.plot()` 改成
`annotate()`，只在要存盘（`--save-dir`）时才调，推理主循环里不再有这项开销。

### 画面自己缩放（已修）

不设 `visual_bounds` 时，rerun 的 2D 视图按“**视图内数据的包围盒**”自动取景。
检测框/关键点一伸出画面（人半出画就是这种），取景就跟着放大缩小，看着像画面在自己缩放。
现在 `visual_bounds` 按图像尺寸钉死（第一帧时确定，分辨率变了才重发 blueprint），
修复后同一段数据在两种情况下屏上尺寸逐像素相同（1168×875 px）。

布局与实体：

| 实体 | 内容 |
|---|---|
| `camera/image` | 相机**原图**（不在像素上烧框/骨架，交给 rerun 按下面的结构化实体画） |
| `camera/image/boxes` | 框的 `Boxes2D`，可悬停看标签与置信度 |
| `camera/image/keypoints` | 仅 pose：关键点 `Points2D`，标签是 `nose`/`left_shoulder` 等 |
| `camera/image/skeleton` | 仅 pose：骨架 `LineStrips2D`（两端都过 `kpt_conf` 才连） |
| `camera/depth` | 仅 pose3d：估计深度图 `DepthImage`（Turbo 伪彩，送前先降采样） |
| `person/count` | 人数曲线（右侧 TimeSeriesView） |
| `person/events` | 人数变化日志（带各框置信度） |
| `camera/pose3d/skeleton` | 仅 pose+pose3d：3D 骨架 `LineStrips3D` |
| `camera/pose3d/positions` | 每个人的 3D 位置球，标签 = `#id 距离 · 地面速度` |
| `world/...` | 开了 `[world]` 才有：骨架/位置的世界系副本（`RIGHT_HAND_Z_UP`） |
| `person/values` | 每人一行的极简数值（距/速/状态/有效关节），右列顶部 |
| `person/metrics/<组>/<指标>/<id>` | 每个人的指标曲线：`speed` 组只画 `speed_ground`（地面速度，与数值行的「速」同源；三维模长 `speed` 只进 JSONL）、`fall` 组画 `fall_level`（等级）/ `torso_angle`（躯干角）；颜色与图例名由 id 决定 |
| `person/fall_light` | **摔倒报警灯**（一格 2D：整块颜色 + 格内文字；红=摔倒会闪 / 琥珀=疑似倒地 / 绿=正常 / 暗蓝灰=无人） |
| `person/fall_alert` | 报警灯的纯文本副本（markdown，可选中复制） |
| `person/fall` | 摔倒状态变化日志（只在跳变时记一条） |

### 单目 3D 骨架（右下角小窗）

`[pose3d].enabled = true` 时，左上/左下是原图与估计深度图，右上/右下
是人数曲线与 `Spatial3DView` 3D 骨架；关掉就用 `--no-pose3d`，不会加载深度模型。

做法：pose 权重出 17 个像素关键点，`yolo26*-depth` 出同一帧的公制深度图，在每个
关键点像素附近取窗口中位数当 z，再用针孔模型（按 `fov_deg` 估的内参）还原成**相机坐标系**
（`X` 右 / `Y` 下 / `Z` 前，即 rerun 的 `RIGHT_HAND_Y_DOWN`）。关键点要“置信度过阈值
**且**深度有效”才画，缺一个就不画。

几个刻意的选择：

* **固定视角**：不设 `eye_controls` 时 rerun 按数据包围盒自动取景，而单目深度逐帧抖动，
  画面就会一直“自动缩放”。这里把 `eye`/`look_at`/`eye_up` 钉死（`[pose3d]` 里可改），
  打开就是一个固定视角；viewer 里鼠标拖仍可自由旋转。
* **卡尔曼滤波**：每个人的每个关键点坐标（17×3 个标量）各配一个恒定速度滤波，跨帧用检测框
  IoU 贪心配对跟踪，把深度噪声导致的“呼吸”抹平。`kalman_process` 越大越跟手，
  `kalman_measurement` 越大越平滑；`--no-pose3d-kalman` 可关掉。
* **深度图**先整倍数降采样再送（1440×1080 的 float32 每帧 6 MB，小窗用不着），
  `DepthImage` 在 viewer 侧上色。

小窗里画的是骨头 `LineStrips3D` + 每个人一个位置球（相机坐标系，`RIGHT_HAND_Y_DOWN`），
颜色跟着 id 走，位置球的标签是 `#id 距离 · 地面速度`（见「跟踪与 ID」）。

**3D 视图不再记逐个关节**：17 个带名字的关节点（`Points3D` + 标签）会盖住骨架本身，
鼠标扫过去还全是坐标读数。关节数据仍然在（`person.joints`、JSONL 里的 `valid_keypoints`），
想逐个关节看名字与坐标就看 2D 关键点那层（`camera/image/keypoints`）。

**它只是看着像 3D，不是测量**：深度是模型从单张图估的，绝对尺度会有出入（实测
`bus.jpg` 里一个人的垂直跨度会比真身高大一截），而且相机没标定，只按一个假定的视场角
近似。看骨架结构和姿态足够，别拿它量距离。

深度模型选型（同一张 `bus.jpg`，以 `yolo26m-depth@768` 为参考；本机 RTX 4060 Laptop）：

| 配置 | 延迟 | abs_rel vs m@768 ↓ | δ1 vs m@768 ↑ |
|---|---|---|---|
| n@320 | 4.5 ms | 0.129 | 0.751 |
| s@640 | 10.0 ms | 0.082 | 0.948 |
| n@640 | 7.7 ms | 0.061 | 0.965 |
| **m@640（默认）** | 17.4 ms | 0.036 | 0.998 |

`[pose3d].weights` / `imgsz` 想换就换：默认 m@640；省算力可以退到 s@640 / n@640。
注意精度主要是被 **imgsz** 卡住的（权重在 768 训练/评测，320 远没喂饱），所以
`imgsz` 降到 320 掉得比换个更大的模型还快。另外官方的绝对尺度是全局标定，可以用 `YOLO(...).calibrate()`
拿本相机几个实测点重标（文档：5 点/图、100+ 图可到全量拟合的 1% 内）。

* 弹窗模式下 viewer 是**独立进程**，会继承当前终端；在脚本/管道里跑就用 `--rerun-save`，
  别让 viewer 把 stdout 占住。
* 不想开窗口也不留文件，就用 `--no-rerun`；要只留检测到人的帧（最多 1 张/秒，见
  `save_min_interval`）就用 `--no-rerun --save-dir`。

## 跟踪与 ID：持久 id / 轨迹 / 相对量速度

2D 关键点 + 深度抬成 3D 之后，跨帧要把"第 i 个人"对上号：前后帧用检测框 IoU 贪心配对
（`[track].iou_threshold`），可选再加一条 3D 距离门限（`match_max_distance`）——
两个人交错而过时光靠 IoU 容易换 id，加上"位置不能跳太远"就稳了。配上就**沿用同一个 id**。

* **id 单调递增、永不复用**，可以直接当人的编号写进日志（`[output].jsonl` 里就是它）。
* **漏检不清空**：轨迹保留 `max_missed` 帧（15 Hz 下 5 帧 ≈ 0.33 s），期间位置由卡尔曼
  预测（框留着、2D 关键点不再画），所以短时间被遮挡的人不会丢 id，也不会在曲线里被数两遍。
* **单个关键点**连续丢 `max_missed_kpt` 帧后不再画，人的其他部位继续画。
* **轨迹**按 `trail_interval` 限流记录，最多 `trail_length` 个点（0 = 不留）。
  轨迹**不画到画面上**：3D 视图只给位置球与骨架，想另做用处直接读 `person.trail`
  （跟踪器一直在维护它，JSONL 里没有这一项）。
* **速度**优先取位置卡尔曼的速度状态，再走一道指数平滑（`speed_smoothing`）：
  单目深度抖动大，不平滑就会看到速度在 0 和 1 之间乱跳。`speed_ground` 是沿**上方向**的
  垂直分量之外的部分，"人走多快"看它；低于 `speed_deadband` 显示"静止"。
  "上方向"默认取相机 -Y（假定镜头水平）；开了 `[world]` 就换成世界 Z，镜头俯仰时才算得准。

**画面里的 id**：框、关键点、骨架、曲线都用同一个 id 决定颜色（`viz.PALETTE` 按 id 轮转），
所以同一个人在任何视图里都同色；标签写成 `#3 0.87`（id + 置信度）。右列顶部那份
`person/values` 每人一行：

```
- #1 · 距 2.35 m · 速 0.40 m/s · 行走 · 关节 14/17
- #2 · 距 1.02 m · 速 0.00 m/s · 静止 · 关节 12/17
```

深度模型比 pose 贵（m 档 22 ms vs 16.6 ms，x 档 44 ms），所以深度按 `[pose3d].hz` 单独限速：
**没到深度时刻的那几帧只做 2D 关联**（id/轨迹照常，3D 部分走预测），不会拿上一帧的旧深度
去抬这一帧的关键点。`--no-pose3d` 时整个跟踪退化成 2D-only（只有 id 与轨迹，无距离/速度）。

尺度口径（重要）：`distance` / `speed` / 世界坐标都建立在**单目估计深度**上，绝对公制值
不可全信 —— 同一次运行内做相对比较（谁更快、谁更靠前）才有意义。`valid_keypoints` /
`coverage`（有效关节比例）是尺度无关的可信度自检。d435i 那边的真深度可以直接量身高/肩宽，
这边不出这类绝对量指标。

## 摔倒检测（`[fall]`）

`fall.py` 是一条**单人时序状态机**（纯逻辑、可单测），三步判据**缺一步都不报警**，
所以不会把"弯腰系鞋带""快速蹲一下""主动慢慢躺下"当成摔倒：

| 步 | 信号 | 为什么这么定 |
|---|---|---|
| 1 冲击 | 向下速度 ≥ `impact_speed`(1.0 m/s)，或 `impact_window`(0.7 s) 内从窗口高点掉了 `drop_height`(0.35 m) | 速度滤波有滞后，所以再用"位移"兜一道 |
| 2 躺倒 | 躯干与"上"方向夹角 ≥ `lying_angle`(60°)；角度测不到时用"比站立参考低 `lying_height`(0.35 m)"兜底 | 躺着时关键点常被床/腿挡住；站立参考是自动学的 EMA，不用量相机多高 |
| 3 持续 | 躺倒连续 `lying_time`(1.2 s)，且必须在冲击后 `confirm_window`(3 s) 内 | 把"快速蹲一下又起来"排掉 |

报警后要**直立保持 `recover_time`(1 s)** 才解除（人爬起来的过程中不闪烁）。
状态：`0 正常` / `1 疑似`（有冲击或正在倒地）/ `2 摔倒`（报警）。

```bash
uv run hik-person --no-fall                  # 关掉摔倒检测（连报警灯格子一起收起来）
uv run hik-person --fall-light-flash 0       # 报警灯常亮，不闪
```

报警长什么样（与 d435i 一致）：

- **「摔倒警报」一格**（右列指标格子的第三格）：**整块颜色 + 文字画在色块里面**——
  颜色是屏幕上最先被看见的东西，也不用表情符号：

  | 状态 | 颜色 | 文字 |
  |---|---|---|
  | 摔倒（level 2） | **红**（和"整个人变红"同一个色，2 Hz 闪） | `摔倒 #1` |
  | 疑似倒地（有冲击/正在倒地） | 琥珀 | `疑似倒地 #1` |
  | 正常 | 绿 | `正常` |
  | 画面内无人 | 暗蓝灰（故意不用绿：没人在 ≠ 一切正常） | `画面内无人` |

  红字下面跟着明细行（`#1 · 躯干 90° · 4.1 m · 0.7 s`）与最近一次状态变化，
  这些字都在色块里面；同一格还有一份纯文本副本 `person/fall_alert`（可选中/复制）。
- **人变红**：2D 框/关键点/骨架与 3D 位置球都换 `ALARM_COLOR`（比"去某个框里找条线"快）
- 数值行行首插 `⚠ 摔倒 89°`（确认前只写"倒地中"；仅挨一下冲击的正常动作不插）
- 曲线：`person/metrics/fall/fall_level/<id>` 与 `torso_angle/<id>`（0/1/2 与度数）
- 日志：`person/fall` 只在状态跳变时记一条（摔倒 / 疑似 / 恢复）

**尺度提醒**：这边的深度是单目估计的，`impact_speed` / `drop_height` / `lying_height`
都是"估计尺度"下的经验值（见 `fall.py` 的说明）；躯干夹角与时序判据不受尺度影响。
调阈值请对着 `person/metrics/fall` 那两条曲线看，别照搬 d435i（真深度）实测出来的数。

## 世界坐标系与 JSON Lines 输出

### 世界坐标（`[world]`）

`[world].enabled = true` 时额外算一份"人在地面上的位置"。只需要三个数（`--world-height` /
`--world-pitch` / `--world-yaw` 可临时覆盖）：

1. `height`：光心离地多高（米）。卷尺量，或把人放到已知距离看深度读数校准。
2. `pitch_deg`：镜头往下俯了几度。手机水平仪靠在相机外壳上量，或用已知身高反推。
3. `yaw_deg` / `roll_deg`：装得周正就都是 0。

坐标系照 ROS REP-103 的习惯：相机系 `RDF`（X 右 / Y 下 / Z 前）-> 世界系 `FLU`
（X 前 / Y 左 / Z 上，原点在相机正下方的地面点）。水平放置时相机前方 2 m 的点就在世界
`(2, 0, 高度)`；镜头竖直向下时同一个点落在脚下 2 m。外参不准只会让**绝对位置**整体偏移，
不影响骨架形状与相对距离；`--world-view` 可以在右下角再开一个世界系 3D 视图对照看。

**单目方案的前提**：深度是估的，绝对尺度会飘，所以世界坐标适合"谁在谁左边、大概多远"
这类相对比较，不要当公制真值；要给底盘发坐标，先用已知身高或已知距离把尺度校一遍。

### JSON Lines（`[output]`）

`[output].jsonl` 非空时按 `[output].hz` **追加**写每帧的人（`--no-rerun` 也照写）：

```json
{"t": 3.86, "wall": 1791975404.516, "frame": 264, "device_timestamp_ms": 1791975404487,
 "count": 2,
 "people": [{"id": 3, "conf": 0.87, "distance": 2.31,
             "cam": [0.12, -0.05, 2.31], "world": [0.31, -0.12, 1.10],
             "speed": 0.41, "speed_ground": 0.41, "state": "行走",
             "velocity": [-0.02, 0.01, 0.41], "box": [301.2, 120.4, 360.8, 470.1],
             "valid_keypoints": 14, "coverage": 0.82, "age": 96, "missed": 0}]}
```

* `t` 是**相对本次运行起点**的秒（`wall` 是绝对时间，跨次对齐用它），一行一帧、写完即 flush。
* 位置测不到时 `cam` / `world` / `distance` 是 `null`，不会给一个骗人的 `(0,0,0)`。
* `speed` / `speed_ground` 与 `cam` / `world` 同样是**估计尺度**（见上）。

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
uv run pytest                  # 不需要相机，也不需要模型权重
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
├── pose.py      ultralytics YOLO 姿态（COCO-17 骨架）
├── depth.py     单目深度估计（ultralytics depth 任务）
├── geometry.py  内参 / 深度取样 / 反投影（纯函数）
├── pose3d.py    关键点 -> 3D 人（含"人在哪"的逐级兜底）
├── track.py     卡尔曼 + IoU 关联（持久 id / 轨迹 / 相对量速度）
├── world.py     相机系 -> 世界（地面）系的静态外参
├── output.py    JSON Lines 输出
├── viz.py       rerun 记录与布局（Image / Boxes2D / Points2D / LineStrips2D / Scalars）
└── __main__.py  CLI，只负责把参数接起来
tools/
├── check_colors.py  颜色链路对比验证
└── bench.py         分环节计时
models/
├── yolo26m-pose.pt   姿态权重（默认）
├── yolo26m-depth.pt  单目深度权重（默认，[pose3d]）
├── yolo26n-pose.pt   n 档姿态权重（更快）
└── yolo26s-depth.pt  s 档深度权重（更快）
```

## 实测性能（1440×1080 + RTX 4060 Laptop）

`uv run python tools/bench.py` 可复现：

| 环节 | 耗时 |
|---|---|
| SDK 转 BGR（纯计算，不含等帧） | 1.1 ms |
| BGR → RGB（rerun 需要） | 0.08 ms |
| YOLO pose 推理（`yolo26m-pose`, `imgsz 640`, `cuda:0`） | 16.6 ms |
| 深度推理（`yolo26m-depth`, `imgsz 640`，按 `pose3d.hz` 单独限速） | 22.0 ms |
| 取一帧（含等帧） | 33 ms = 相机帧间隔 |

单帧计算约 18 ms（再叠上按 `pose3d.hz` 跑的深度模型约 39 ms），仍低于 15 Hz 的 66 ms
预算，瓶颈在相机节拍而不在模型。主循环按 `detect_hz` 限速，且只处理最新的帧（不积压）：
相机开到 249 fps 也只按 15 Hz 推理。

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
* `设置特性 X=Y 失败` —— 报错已经把尽力查到的类型/可选符号列出来了；`0x80000004` 是
  "节点在、符号名不对"。同义符号名（`Bits_8` 对 `ADCBitDepth_8`）会先自动改写，所以还
  报这个错就说明数字对不上或有多个候选，照着报错里的符号名写全名即可。
* `MV_CC_OpenDevice failed: 0x80000203` 反复出现、但没有任何进程在取流 —— 上一把程序
  刚退出时相机的访问权还没放完，等一两秒重试就好；一直不行就拔插一次 USB。
* 改了 `config.toml` 好像没生效 —— 先 `--show-config` 看合并后的值，再确认写到正确的段里
  （键名写错会有 warning）。
* rerun 窗口没起来 —— 加 `--no-rerun` 先确认取流与检测本身正常。

## 许可证

AGPL-3.0-or-later，全文见 [LICENSE](LICENSE)。

- 依赖 [ultralytics](https://github.com/ultralytics/ultralytics)（AGPL-3.0），
  本项目因此采用同一许可证。对外分发、或以网络服务形式提供时，需按 AGPL 提供源码。
- `models/yolo26m-pose.pt` / `models/yolo26m-depth.pt` 为默认权重（ultralytics 官方姿态/深度模型）；n-pose / s-depth 也随仓库一同提供，换 `weights` 即可回退。
- 运行需要海康 **MVS SDK**（`/opt/MVS`）。它是海康的专有软件，不在本仓库内，
  也不由本许可证覆盖；使用它需遵守海康自己的许可条款。
