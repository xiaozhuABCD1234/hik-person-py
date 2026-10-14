# SPDX-License-Identifier: AGPL-3.0-or-later
"""HIKROBOT MVS SDK 取流。

MVS SDK 的像素转换 / GenICam producer 库是 dlopen 时才加载的，而 glibc 只在进程
启动时读一次 LD_LIBRARY_PATH，所以环境变量必须在进程启动前就位。
ensure_runtime_environment() 在环境不满足时用 os.execve 重启自己，
因此不需要任何 shell 包装脚本。
"""

from __future__ import annotations

import os
import sys
from collections.abc import Iterable, Sequence
from ctypes import POINTER, byref, c_bool, c_ubyte, cast, memset, sizeof
from dataclasses import dataclass, field

import numpy as np

from .colors import pixel_type_name

MVS_ROOT = os.environ.get("MVCAM_SDK_PATH") or "/opt/MVS"
MVS_LIB_DIR = os.path.join(MVS_ROOT, "lib", "64" if sys.maxsize > 2**32 else "32")
MVIMPORT_DIR = "/opt/MVS/Samples/64/Python/MvImport"

_REEXEC_FLAG = "HIK_PERSON_MVS_ENV"


class MvsError(RuntimeError):
    pass


def library_path_configured() -> bool:
    """SDK 库目录是否已在 LD_LIBRARY_PATH 里。"""
    return MVS_LIB_DIR in os.environ.get("LD_LIBRARY_PATH", "").split(":")


def library_path_hint() -> str:
    return (
        f"'{MVS_LIB_DIR}' 不在 LD_LIBRARY_PATH 里，SDK 无法在运行时 dlopen "
        f"像素转换/GenICam producer 库。正常入口 "
        f"(uv run hik-person / python -m hik_person) 会自动处理；"
        f"也可以先 source {MVS_ROOT}/bin/set_env_path.sh"
    )


def _apply_env(env: dict) -> dict:
    """补齐 MVS SDK 需要的运行时环境变量。"""
    lib = os.path.join(MVS_ROOT, "lib")
    env.setdefault("MVCAM_SDK_PATH", MVS_ROOT)
    env.setdefault("MVCAM_COMMON_RUNENV", lib)
    env.setdefault("MVCAM_SOFTWARE_LIBENV", lib)
    env.setdefault("MVCAM_GENICAM_CLPROTOCOL", os.path.join(lib, "CLProtocol"))
    env.setdefault("ALLUSERSPROFILE", os.path.join(MVS_ROOT, "MVFG"))
    # 已在列表里就不要重复加
    entries = [p for p in env.get("LD_LIBRARY_PATH", "").split(":") if p]
    if MVS_LIB_DIR not in entries:
        entries.insert(0, MVS_LIB_DIR)
    env["LD_LIBRARY_PATH"] = ":".join(entries)
    return env


def ensure_runtime_environment() -> None:
    """补齐 MVS 运行时环境；必要时用 os.execve 换一个新进程。

    glibc 只在进程启动时读一次 LD_LIBRARY_PATH，而 MVS SDK 的像素转换 /
    GenICam producer 库是 dlopen 时才加载的，所以在同一个进程里改 os.environ
    没有用。execve 出去就是一个全新进程，加载器自然能看到新的搜索路径，
    于是不需要任何 .sh 包装脚本。
    """
    if library_path_configured():
        return
    if not sys.argv or sys.argv[0].startswith("-"):
        # python -c ... 这类调用没法安全重启，直接给出提示
        raise MvsError(library_path_hint())
    if os.environ.get(_REEXEC_FLAG) == "1":
        # 已经重启过一次还是不满足 -> 报错，避免 execve 死循环
        raise MvsError(library_path_hint())

    env = _apply_env(dict(os.environ))
    env[_REEXEC_FLAG] = "1"
    os.execve(sys.executable, [sys.executable, *sys.argv], env)
    raise AssertionError("unreachable")  # pragma: no cover


def load_sdk():
    """导入 MVS Python 封装（等价于 sample 里的 MvImport 用法）。"""
    ensure_runtime_environment()

    # MvCameraControl_class.py 在导入时(第 77 行)就执行 check_sys_and_update_dll()，
    # 用 MVCAM_COMMON_RUNENV 拼出 libMvCameraControl.so 的绝对路径 -> 必须先设好
    lib = os.path.join(MVS_ROOT, "lib")
    os.environ.setdefault("MVCAM_SDK_PATH", MVS_ROOT)
    os.environ.setdefault("MVCAM_COMMON_RUNENV", lib)
    os.environ.setdefault("MVCAM_SOFTWARE_LIBENV", lib)
    os.environ.setdefault("MVCAM_GENICAM_CLPROTOCOL", os.path.join(lib, "CLProtocol"))
    os.environ.setdefault("ALLUSERSPROFILE", os.path.join(MVS_ROOT, "MVFG"))

    if MVIMPORT_DIR not in sys.path:
        # MvImport 内部是绝对导入（from PixelType_header import *），
        # 需要该目录本身出现在 sys.path 上
        sys.path.insert(0, MVIMPORT_DIR)
    import MvCameraControl_class as sdk

    return sdk


def _decode(chars) -> str:
    raw = memoryview(chars).tobytes().split(b"\x00", 1)[0]
    for encoding in ("utf-8", "gbk", "latin-1"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("latin-1", errors="replace")


@dataclass
class Device:
    index: int
    model: str
    serial: str
    layer: str
    info: object = field(repr=False, default=None)

    @property
    def virtual(self) -> bool:
        """MVS 虚拟相机的序列号是 ``Vir`` 开头（实测 Vir62953321）。"""
        return self.serial.startswith("Vir")


@dataclass
class Frame:
    """一帧 BGR 图像 + 相机元数据。"""

    bgr: np.ndarray
    width: int
    height: int
    frame_number: int
    pixel_type: int
    host_timestamp_ms: int
    exposure_us: float
    gain_db: float


def list_devices(sdk) -> list[Device]:
    dev_list = sdk.MV_CC_DEVICE_INFO_LIST()
    ret = sdk.MvCamera.MV_CC_EnumDevices(sdk.MV_GIGE_DEVICE | sdk.MV_USB_DEVICE, dev_list)
    if ret != 0:
        raise MvsError(f"MV_CC_EnumDevices failed: 0x{ret & 0xFFFFFFFF:08x}")
    devices: list[Device] = []
    for i in range(dev_list.nDeviceNum):
        info = cast(dev_list.pDeviceInfo[i], POINTER(sdk.MV_CC_DEVICE_INFO)).contents
        if info.nTLayerType == sdk.MV_USB_DEVICE:
            usb = info.SpecialInfo.stUsb3VInfo
            devices.append(
                Device(i, _decode(usb.chModelName), _decode(usb.chSerialNumber), "usb", info)
            )
        elif info.nTLayerType in (sdk.MV_GIGE_DEVICE, sdk.MV_GENTL_GIGE_DEVICE):
            gige = info.SpecialInfo.stGigEInfo
            devices.append(
                Device(i, _decode(gige.chModelName), _decode(gige.chSerialNumber), "gige", info)
            )
        else:
            devices.append(Device(i, "<unknown>", "", f"0x{int(info.nTLayerType):x}", info))
    return devices


# GenICam 特性类型 -> 报错时给用户看的说法（SDK 没有“查类型”的接口，见 Camera._node_kind）
_NODE_KIND_LABELS = {
    "bool": "布尔（Bool）",
    "int": "整型（Int）",
    "float": "浮点（Float）",
    "enum": "枚举（Enum）",
}


def parse_feature_spec(spec: str) -> tuple[str, str]:
    """把 ``--feature NAME=VALUE`` 解析成 ``(NAME, VALUE)``。"""
    name, sep, value = spec.partition("=")
    name, value = name.strip(), value.strip()
    if not sep or not name or not value:
        raise ValueError(f"特性写法应为 NAME=VALUE，收到 {spec!r}")
    return name, value


def feature_setter_order(value: str) -> tuple[str, ...]:
    """按值的形态决定先试哪种 setter。

    SDK 没有“查特性类型”的接口，而 GenICam 特性分枚举/整型/浮点/布尔四种，
    写错类型会直接报错，所以按值的形态排队试，并把失败码收集起来一次性报出。
    纯函数，便于单测。
    """
    text = value.strip().lower()
    if text in ("true", "false"):
        return ("bool", "int", "enum")
    try:
        int(text)
    except ValueError:
        pass
    else:
        return ("int", "enum_int", "float")
    try:
        float(text)
    except ValueError:
        return ("enum",)
    return ("float", "enum_int")


def _norm_symbol(text: str) -> str:
    """符号名归一化：只留小写字母数字（``Bits_8``/``bits8`` 视为同一个）。"""
    return "".join(ch for ch in text.lower() if ch.isalnum())


def symbolic_candidates(name: str, value: str, entries: Sequence[str]) -> list[str]:
    """从枚举符号表里猜用户想写的是哪一个（纯函数，便于单测）。

    同一个概念在不同机型上符号名不同（``Bits_8`` 对 ``ADCBitDepth_8``，数值都是 0），
    所以先把符号名和值都归一化，再用两个判据：① 两串完全一样；② 把两边的节点名都去掉
    后，残部有一头是另一头的后缀（两个方向都拍，所以哪边的配置抄错了都能认出来）。

    只是**提示**（报错里列出最像的），敢不敢真拿它去写由 ``alias_symbolic`` 再筛一道。
    """
    want = _norm_symbol(value)
    if not want:
        return []
    for entry in entries:
        if _norm_symbol(entry) == want:
            return [entry]
    value_body = _norm_symbol(value.replace(name, ""))
    if not value_body:
        return []
    return [
        entry
        for entry in entries
        if (body := _norm_symbol(entry.replace(name, "")))
        and (value_body.endswith(body) or body.endswith(value_body))
    ]


def _trailing_digits(text: str) -> str:
    """``bits8`` -> ``8``；没有尾部数字就是空串。"""
    return text[len(text.rstrip("0123456789")) :]


def alias_symbolic(name: str, value: str, entries: Sequence[str]) -> str | None:
    """这台相机上“同一个设置的另一个名字”是哪个；不敢确定就返回 None。

    同一个特性在不同机型上符号名不同（同一台 MV-CB016-10UC-S 叫 ``Bits_8``，
    MV-CS016-10UC 叫 ``ADCBitDepth_8``，数值都是 0），写错名字不该逼人改 config，
    所以能确定就自动换成这台相机认的那个名字。

    自动改写只认两种把握得住的情况，**都要求名字里的数字不变** —— 8 bit 与 12 bit
    不能搞混，所以不做“最接近”的猜：

    ① 去掉节点名后两边完全一样（``Foo`` 对 ``ADCBitDepth_Foo``）；
    ② 残部有一头是另一头的后缀，且尾部数字一样（``Bits_8`` 对 ``ADCBitDepth_8``）。

    命中多个候选（``Foo_8``/``Bar_8`` 都对得上）也不猜。纯函数，便于单测。
    """
    candidates = symbolic_candidates(name, value, entries)
    if len(candidates) != 1:
        return None
    body = _norm_symbol(candidates[0].replace(name, ""))
    value_body = _norm_symbol(value.replace(name, ""))
    digits = _trailing_digits(body)
    if body == value_body or (digits and digits == _trailing_digits(value_body)):
        return candidates[0]
    return None


def _feature_setters() -> dict:
    """kind -> (调用的 SDK 方法, 把字符串值转成参数)。"""
    return {
        "bool": lambda cam, name, value: cam.MV_CC_SetBoolValue(
            name, value.strip().lower() == "true"
        ),
        "int": lambda cam, name, value: cam.MV_CC_SetIntValue(name, int(value)),
        "float": lambda cam, name, value: cam.MV_CC_SetFloatValue(name, float(value)),
        "enum": lambda cam, name, value: cam.MV_CC_SetEnumValueByString(name, value),
        "enum_int": lambda cam, name, value: cam.MV_CC_SetEnumValue(name, int(value)),
    }


@dataclass(frozen=True)
class FeatureWrite:
    """一次 GenICam 特性写入的结果。

    ``value`` 是**实际写进去**的值：符号名用了别的机型的叫法时，会被自动改写成这台
    相机认的那个名字（``requested`` 保留用户写的原样，两个不一样就说明改写过了）。
    """

    kind: str
    value: str
    requested: str = ""

    @property
    def aliased(self) -> bool:
        return self.value != self.requested

    def summary(self) -> str:
        """给日志用的一行。"""
        if self.aliased:
            return (
                f"{self.value} (以 {self.kind} 写入；这台相机不叫 {self.requested}，"
                f"已自动换成同一个设置的另一个名字)"
            )
        return f"{self.value} (以 {self.kind} 写入)"


class Camera:
    """最小可用的 MVS 取流封装。"""

    def __init__(self, sdk, serial: str | None = None, index: int = 0):
        self.sdk = sdk
        self.serial = serial
        self.index = index
        self.cam = None
        self.device: Device | None = None
        self.notes: list[str] = []
        self.applied_features: list[str] = []
        self._dst = None
        self._param = None

    # ------------------------------------------------------------- 生命周期
    def open(self) -> Camera:
        sdk = self.sdk
        sdk.MvCamera.MV_CC_Initialize()
        devices = list_devices(sdk)
        if not devices:
            raise MvsError("没有枚举到设备；检查 USB3 连接与 MVS 驱动")
        physical = [d for d in devices if not d.virtual]
        if not physical:
            raise MvsError(
                "只枚举到 MVS 虚拟相机（序列号 Vir 开头），没有物理相机；"
                "虚拟相机在 Linux 上打不开（MV_E_ACCESS_DENIED）"
            )

        if self.serial:
            match = [d for d in devices if d.serial == self.serial]
            if not match:
                raise MvsError(
                    f"序列号 {self.serial!r} 不存在；可用: "
                    + ", ".join(f"{d.serial}({d.model})" for d in devices)
                )
            self.device = match[0]
        else:
            # --index 只在物理相机里数，虚拟相机不占号
            self.device = physical[self.index]

        self.cam = sdk.MvCamera()
        ret = self.cam.MV_CC_CreateHandle(self.device.info)
        if ret != 0:
            raise MvsError(f"MV_CC_CreateHandle failed: 0x{ret & 0xFFFFFFFF:08x}")
        ret = self.cam.MV_CC_OpenDevice(sdk.MV_ACCESS_Exclusive, 0)
        if ret != 0:
            raise MvsError(
                f"MV_CC_OpenDevice failed: 0x{ret & 0xFFFFFFFF:08x} "
                f"(同一台相机被别的进程占用？关掉 MVS 客户端或另一个取流进程)"
            )
        return self

    def close(self) -> None:
        if self.cam is not None:
            self.cam.MV_CC_StopGrabbing()
            self.cam.MV_CC_CloseDevice()
            self.cam.MV_CC_DestroyHandle()
            self.cam = None
        self.sdk.MvCamera.MV_CC_Finalize()

    def __enter__(self) -> Camera:
        return self.open()

    def __exit__(self, *exc) -> bool:
        self.close()
        return False

    # ----------------------------------------------------------------- 配置
    def configure(
        self,
        *,
        exposure_us: float = 0.0,
        gain_db: float = -1.0,
        fps: float = 0.0,
        exposure_auto: str | None = "Off",
        gain_auto: str | None = "Off",
        auto_brightness: float | None = None,
        auto_exposure_upper_us: float | None = None,
        auto_gain_upper_db: float | None = None,
        exposure_start_us: float | None = None,
        bayer_quality: int = 1,
        grab_strategy: str = "LatestImagesOnly",
        buffer_count: int = 3,
        feature_overrides: Sequence[str] = (),
    ) -> dict:
        """按固定顺序配置相机；越靠后写入的越优先（显式参数压过通用特性改写）。

        顺序：自动曝光/增益 -> 触发 -> Bayer 质量 -> 取流策略/缓冲
              -> feature_overrides -> AE 参数 -> 曝光/增益 -> 帧率
        帧率放最后：改 ADCBitDepth 之类的特性会把 AcquisitionFrameRate 重置掉。

        ``exposure_auto``/``gain_auto`` 设成 "Continuous" 就是用相机自己的自动
        曝光/增益；此时 ``auto_*`` 三个参数交给相机：目标灰度、曝光上限、增益上限。
        ``exposure_start_us`` 是 AE 的**起点**：相机自己记着上次的曝光/增益（换相机、
        或别人用 MVS 客户端动过之后，起跑线都不一样），给了这个值就先关自动、把曝光
        钉到它，再开 Continuous，保证每轮从同一个起点开始。0/None = 不重置。
        注意这三个特性的类型不一样（写错会拿到 0x80000109，像是不支持）：

        ============================  ======  ====================================
        Brightness                    Int     AE 目标灰度（0-255）
        AutoExposureTimeUpperLimit    Int     AE 曝光上限，不放开的话 AE 只能拉增益
        AutoGainUpperLimit            Float   AE 增益上限（dB）
        ============================  ======  ====================================
        """
        sdk, cam = self.sdk, self.cam
        self.notes = []
        self.applied_features = []

        def hard(ret, what):
            if ret != 0:
                raise MvsError(f"{what} failed: 0x{ret & 0xFFFFFFFF:08x}")

        def soft(ret, what):
            if ret != 0:
                self.notes.append(f"{what}: 0x{ret & 0xFFFFFFFF:08x}（忽略）")

        auto_exposure_on = bool(exposure_auto) and exposure_auto != "Off"
        auto_gain_on = bool(gain_auto) and gain_auto != "Off"

        # AE 起点：相机把上次的曝光/增益存在自己身上，谁用过这台相机都会影响起跑线
        # （实测刚开 Continuous 时读到的几百 us 就是半路值）。先把自动关掉、把曝光
        # 钉到已知值，再开 Continuous，起点就固定了。顺序不能反：开着自动时相机
        # 拒绝写固定值。
        if auto_exposure_on and exposure_start_us and exposure_start_us > 0:
            hard(
                cam.MV_CC_SetEnumValueByString("ExposureAuto", "Off"),
                "ExposureAuto(起点)",
            )
            if auto_gain_on:
                hard(
                    cam.MV_CC_SetEnumValueByString("GainAuto", "Off"),
                    "GainAuto(起点)",
                )
            hard(
                cam.MV_CC_SetFloatValue("ExposureTime", float(exposure_start_us)),
                "ExposureTime(起点)",
            )
            self.notes.append(f"AE 起点: 先把曝光钉到 {exposure_start_us:g} us，再开自动曝光/增益")

        # 自动曝光/增益交给相机自己做（Continuous）：它按自己的测光取目标灰度，
        # 比自己量灰度再反算稳得多。写了自动就不能再写固定值，相机会拒绝。
        if exposure_auto:
            hard(
                cam.MV_CC_SetEnumValueByString("ExposureAuto", exposure_auto),
                "ExposureAuto",
            )
        if gain_auto:
            hard(cam.MV_CC_SetEnumValueByString("GainAuto", gain_auto), "GainAuto")

        # 自由运行
        hard(cam.MV_CC_SetEnumValue("TriggerMode", sdk.MV_TRIGGER_MODE_OFF), "TriggerMode")

        # Bayer 插值质量
        hard(cam.MV_CC_SetBayerCvtQuality(bayer_quality), "MV_CC_SetBayerCvtQuality")

        # 取流策略/缓冲是性能提示，失败只记录
        strategies = {
            "OneByOne": sdk.MV_GrabStrategy_OneByOne,
            "LatestImagesOnly": sdk.MV_GrabStrategy_LatestImagesOnly,
            "LatestImages": sdk.MV_GrabStrategy_LatestImages,
            "UpcomingImage": sdk.MV_GrabStrategy_UpcomingImage,
        }
        soft(
            cam.MV_CC_SetGrabStrategy(strategies[grab_strategy]),
            f"SetGrabStrategy({grab_strategy})",
        )
        soft(cam.MV_CC_SetImageNodeNum(buffer_count), "SetImageNodeNum")

        # 通用特性改写（例如 --feature ADCBitDepth=ADCBitDepth_8；符号名随机型）
        self.applied_features = self.apply_features(feature_overrides)

        # AE 参数：类型按值的形态自动选（Brightness/曝光上限是 Int，增益上限是 Float）
        if auto_exposure_on:
            for name, value in (
                ("Brightness", auto_brightness),
                ("AutoExposureTimeUpperLimit", auto_exposure_upper_us),
                ("AutoGainUpperLimit", auto_gain_upper_db),
            ):
                if value is not None:
                    written = self.set_feature(name, f"{value:g}")
                    self.notes.append(f"{name} = {value:g}（{written.kind}）")

        # 曝光/增益：<=0 与 <0 分别表示“保持设备值”。增益的 0 dB 是合法值（最小值），
        # 所以它不能用 0 当哨兵值。自动开着时这两个值由相机决定，不能再写。
        if not auto_exposure_on and exposure_us > 0:
            hard(cam.MV_CC_SetFloatValue("ExposureTime", exposure_us), "ExposureTime")
        if not auto_gain_on and gain_db >= 0:
            hard(cam.MV_CC_SetFloatValue("Gain", gain_db), "Gain")

        # 帧率最后写：上面的特性改写可能把 AcquisitionFrameRate 重置掉
        if fps > 0:
            hard(
                cam.MV_CC_SetBoolValue("AcquisitionFrameRateEnable", True),
                "AcquisitionFrameRateEnable",
            )
            hard(
                cam.MV_CC_SetFloatValue("AcquisitionFrameRate", fps),
                "AcquisitionFrameRate",
            )

        return self.settings()

    # -------------------------------------------------------------- 读参数
    def _enum_cur(self, key: str) -> int:
        st = self.sdk.MVCC_ENUMVALUE()
        if self.cam.MV_CC_GetEnumValue(key, st) != 0:
            return -1
        return int(st.nCurValue)

    def _int_cur(self, key: str) -> int:
        st = self.sdk.MVCC_INTVALUE()
        if self.cam.MV_CC_GetIntValue(key, st) != 0:
            return -1
        return int(st.nCurValue)

    def _float_cur(self, key: str) -> float:
        st = self.sdk.MVCC_FLOATVALUE()
        if self.cam.MV_CC_GetFloatValue(key, st) != 0:
            return float("nan")
        return float(st.fCurValue)

    # --------------------------------------------------------- 通用特性改写
    def set_feature(self, name: str, value: str) -> FeatureWrite:
        """按名字写一个 GenICam 特性；返回实际写入的方式与实际写入的值。

        失败时会先试着“改名”再报错：同一个特性在不同机型上符号名不同（同一台
        MV-CB016-10UC-S 叫 ``Bits_8``，MV-CS016-10UC 叫 ``ADCBitDepth_8``），能确定
        是同一个设置就自动换成这台相机认的名字，省掉"换个相机就得改 config"这件事。
        改写要求名字里的数字一致（见 ``alias_symbolic``），8 bit 不会被悄悄改成 12 bit。

        还是不成功就把每种尝试的错误码一起报出来，并探清楚这个特性是什么类型
        （实测两种错要分开看）：

        ==========  ==========================================================
        0x80000109  MV_E_GC_NODE_NOT_FOUND：没有这个名字的节点，或节点不是这个类型
        0x80000004  MV_E_PARAMETER：节点在，但给的符号名不在它的枚举列表里
        ==========  ==========================================================
        """
        attempts = []
        setters = _feature_setters()
        for setter_kind in feature_setter_order(value):
            ret = setters[setter_kind](self.cam, name, value)
            if ret == 0:
                return FeatureWrite(setter_kind, value, value)
            attempts.append(f"{setter_kind}=0x{ret & 0xFFFFFFFF:08x}")

        node_kind = self._node_kind(name)
        entries = self._enum_entries(name) if node_kind == "enum" else []
        if node_kind == "enum":
            alias = alias_symbolic(name, value, entries)
            if alias and setters["enum"](self.cam, name, alias) == 0:
                return FeatureWrite("enum", alias, value)

        raise MvsError(
            f"设置特性 {name}={value} 失败（按 {', '.join(attempts)} 依次试过）："
            f"{self._feature_diagnosis(name, value, node_kind, entries)}"
        )

    def _node_kind(self, name: str) -> str | None:
        """探一个 GenICam 特性的类型；没有这个节点返回 None。

        SDK 没有“查类型”的接口，只能拿四种 getter 逐个试：实测节点存在时只有对应
        类型的那一个返回 0（别的都是 0x80000109）。
        """
        sdk = self.sdk
        if self.cam.MV_CC_GetEnumValue(name, sdk.MVCC_ENUMVALUE()) == 0:
            return "enum"
        if self.cam.MV_CC_GetIntValue(name, sdk.MVCC_INTVALUE()) == 0:
            return "int"
        if self.cam.MV_CC_GetFloatValue(name, sdk.MVCC_FLOATVALUE()) == 0:
            return "float"
        if self.cam.MV_CC_GetBoolValue(name, c_bool()) == 0:
            return "bool"
        return None

    def _enum_entries(self, name: str) -> list[str]:
        """枚举特性支持的符号名（SDK 只给数值，得逐个查回符号）。"""
        sdk = self.sdk
        st = sdk.MVCC_ENUMVALUE()
        if self.cam.MV_CC_GetEnumValue(name, st) != 0:
            return []
        entries = []
        for i in range(st.nSupportedNum):
            entry = sdk.MVCC_ENUMENTRY()
            entry.nValue = st.nSupportValue[i]
            if self.cam.MV_CC_GetEnumEntrySymbolic(name, entry) == 0:
                entries.append(_decode(entry.chSymbolic))
        return entries

    def _feature_diagnosis(
        self, name: str, value: str, kind: str | None, entries: list[str]
    ) -> str:
        """给 set_feature 的失败补一句人话：这特性是什么类型、能写什么值。

        ``kind``/``entries`` 由 set_feature 探好传进来（它自己也要用，不重复查）。
        """
        if kind is None:
            return (
                f"这台相机上没有读到 {name} 这个特性：检查拼写，或它需要先满足别的条件"
                f"（比如先关掉自动模式 / 停止取流）；也可以 dump 一份特性表看有什么"
            )
        if kind != "enum":
            return (
                f"{name} 是{_NODE_KIND_LABELS[kind]}特性，不能按枚举符号写 {value!r}"
                f"（枚举/整型/浮点/布尔要分别用各自的写法，类型写错也是这个错）"
            )
        if not entries:
            return f"{name} 是枚举特性，但读不到它的符号表"
        known = f"{name} 是枚举特性，这台相机只认这些符号：{' / '.join(entries)}"
        guesses = symbolic_candidates(name, value, entries)
        if guesses:
            return f"{known}；想写的可能是 {' 或 '.join(guesses)}（不能确定，没有自动改写）"
        return f"{known}（符号名随机型不同，别照抄别的相机的配置）"

    def apply_features(self, specs: Iterable[str]) -> list[str]:
        """批量应用 ``NAME=VALUE``；返回可打印的摘要。"""
        applied = []
        for spec in specs:
            name, value = parse_feature_spec(spec)
            written = self.set_feature(name, value)
            applied.append(f"{name}={written.summary()}")
        return applied

    def settings(self) -> dict:
        pixel_type = self._enum_cur("PixelFormat")
        return {
            "model": self.device.model if self.device else "?",
            "serial": self.device.serial if self.device else "?",
            "width": self._int_cur("Width"),
            "height": self._int_cur("Height"),
            "pixel_format": pixel_type_name(pixel_type) if pixel_type >= 0 else "?",
            "exposure_us": round(self._float_cur("ExposureTime"), 1),
            "gain_db": round(self._float_cur("Gain"), 3),
            "resulting_fps": round(self._float_cur("ResultingFrameRate"), 2),
        }

    # ----------------------------------------------------------------- 取流
    def start(self) -> None:
        ret = self.cam.MV_CC_StartGrabbing()
        if ret != 0:
            raise MvsError(f"MV_CC_StartGrabbing failed: 0x{ret & 0xFFFFFFFF:08x}")

    # ------------------------------------------------------------ 曝光/增益
    def wait_auto_settled(
        self,
        *,
        max_frames: int = 120,
        stable_frames: int = 8,
        rel_tol: float = 0.01,
    ) -> tuple[dict, bool]:
        """取帧等相机自己的 AE 收敛；返回 ``(参数读数, 是否收敛)``。

        AE 是逐帧爬的：刚开 ``Continuous`` 时读到的曝光只是起点或半路值（实测从
        几百 us 爬到上限要几十帧），所以“开完就读”会打印出一个没意义的数字。
        这里一边取帧一边看 ``ExposureTime``/``Gain`` 还动不动：连续 ``stable_frames``
        帧的相对变化都在 ``rel_tol``（曝光）/``0.05 dB``（增益）内就认为收敛；
        等满 ``max_frames`` 帧还不稳就返回未收敛，让调用方自己决定怎么说。
        """
        prev: tuple[float, float] | None = None
        stable = 0
        for _ in range(max_frames):
            self.read()
            cur = self.settings()
            now = (cur["exposure_us"], cur["gain_db"])
            moved = (
                prev is None
                or abs(now[0] - prev[0]) > rel_tol * max(abs(prev[0]), 1.0)
                or abs(now[1] - prev[1]) > 0.05
            )
            stable = 0 if moved else stable + 1
            if stable >= stable_frames:
                return cur, True
            prev = now
        return self.settings(), False

    def read(self, timeout_ms: int = 1000) -> Frame | None:
        """取一帧（含 SDK 转 BGR8）；没有数据返回 None。

        注意 ``Frame.exposure_us``/``gain_db`` 来自帧头元数据（``fExposureTime``/
        ``fGain``），**有些机型恒为 0**（实测 MV-CS016-10UC），别拿它判断当前曝光；
        要读当前值用 ``settings()``（读节点）。
        """
        sdk = self.sdk
        out = sdk.MV_FRAME_OUT()
        memset(byref(out), 0, sizeof(out))
        ret = self.cam.MV_CC_GetImageBuffer(out, timeout_ms)
        if ret != 0 or not out.pBufAddr:
            return None
        try:
            info = out.stFrameInfo
            return Frame(
                bgr=self._to_bgr(out, info),
                width=int(info.nWidth),
                height=int(info.nHeight),
                frame_number=int(info.nFrameNum),
                pixel_type=int(info.enPixelType),
                host_timestamp_ms=int(info.nHostTimeStamp),
                exposure_us=float(info.fExposureTime),
                gain_db=float(info.fGain),
            )
        finally:
            self.cam.MV_CC_FreeImageBuffer(out)

    def read_raw(self, timeout_ms: int = 1000):
        """取一帧未转换的原始缓冲。

        返回 (raw, width, height, pixel_type, frame_len)，raw 是 uint8 的副本；
        用于跟 OpenCV 去马赛克做逐帧对比（tools/check_colors.py）。
        """
        sdk = self.sdk
        out = sdk.MV_FRAME_OUT()
        memset(byref(out), 0, sizeof(out))
        ret = self.cam.MV_CC_GetImageBuffer(out, timeout_ms)
        if ret != 0 or not out.pBufAddr:
            return None
        try:
            info = out.stFrameInfo
            length = int(info.nFrameLen)
            raw = np.ctypeslib.as_array(
                cast(out.pBufAddr, POINTER(c_ubyte)), shape=(length,)
            ).copy()
            return (raw, int(info.nWidth), int(info.nHeight), int(info.enPixelType), length)
        finally:
            self.cam.MV_CC_FreeImageBuffer(out)

    def _to_bgr(self, out, info) -> np.ndarray:
        return self.convert_to_bgr(
            out.pBufAddr,
            int(info.nWidth),
            int(info.nHeight),
            int(info.enPixelType),
            int(info.nFrameLen),
        )

    def convert_to_bgr(
        self, src_ptr, width: int, height: int, src_pixel_type: int, src_len: int
    ) -> np.ndarray:
        """SDK 内部转 BGR8（相机原生 Bayer 排布交给 SDK 处理）。"""
        sdk = self.sdk
        need = width * height * 3
        if self._dst is None or len(self._dst) != need:
            self._dst = (c_ubyte * need)()
            param = sdk.MV_CC_PIXEL_CONVERT_PARAM_EX()
            memset(byref(param), 0, sizeof(param))
            param.nWidth = width
            param.nHeight = height
            param.enDstPixelType = sdk.PixelType_Gvsp_BGR8_Packed  # 35127317
            param.pDstBuffer = cast(self._dst, POINTER(c_ubyte))
            param.nDstBufferSize = need
            self._param = param

        param = self._param
        param.pSrcData = src_ptr
        param.nSrcDataLen = src_len
        param.enSrcPixelType = src_pixel_type
        param.nDstLen = 0
        ret = self.cam.MV_CC_ConvertPixelTypeEx(param)
        if ret != 0:
            raise MvsError(
                f"MV_CC_ConvertPixelTypeEx({pixel_type_name(src_pixel_type)}"
                f" -> BGR8_Packed) failed: 0x{ret & 0xFFFFFFFF:08x}"
            )
        # 缓冲是复用的：拷出来，调用方拿到的数组与下一次取流解耦。
        # 注意必须用 frombuffer + reshape：np.ctypeslib.as_array 对 ctypes *数组*
        # 会忽略传入的 shape，直接返回一维缓冲。
        return np.frombuffer(self._dst, dtype=np.uint8).reshape(height, width, 3).copy()
