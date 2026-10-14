# SPDX-License-Identifier: AGPL-3.0-or-later
"""HIKROBOT MVS SDK 取流。

MVS SDK 的像素转换 / GenICam producer 库是 dlopen 时才加载的，而 glibc 只在进程
启动时读一次 LD_LIBRARY_PATH，所以环境变量必须在进程启动前就位。
ensure_runtime_environment() 在环境不满足时用 os.execve 重启自己，
因此不需要任何 shell 包装脚本。
"""

from __future__ import annotations

import math
import os
import sys
from ctypes import POINTER, byref, c_ubyte, cast, memset, sizeof
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


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def next_exposure_step(
    mean: float,
    target: float,
    exposure_us: float,
    gain_db: float,
    *,
    exposure_min_us: float,
    exposure_max_us: float,
    gain_min_db: float,
    gain_max_db: float,
) -> tuple[float, float]:
    """测光的下一步：先动曝光，曝光到上限（受帧率约束）后再用增益补。

    亮度近似为 ``k * 曝光 * 10**(增益/20)``，所以需要的倍数直接取 ``target / mean``；
    曝光被夹住时，剩下的倍数换成 dB：``20 * log10(剩余倍数)``。

    纯函数，方便单测。
    """
    ratio = target / max(mean, 1.0)
    needed = exposure_us * ratio

    if mean < target:
        if needed <= exposure_max_us:
            return max(exposure_min_us, needed), gain_db
        # 曝光顶到上限，剩余亮度用增益补
        residual = ratio * exposure_us / exposure_max_us
        gain = min(gain_max_db, gain_db + 20.0 * math.log10(max(residual, 1e-6)))
        return exposure_max_us, _clamp(gain, gain_min_db, gain_max_db)

    if needed >= exposure_min_us:
        return needed, gain_db
    # 曝光已经到下限（帧率上限要求足够短的曝光），只能降增益
    residual = ratio * exposure_us / exposure_min_us
    gain = max(gain_min_db, gain_db + 20.0 * math.log10(max(residual, 1e-6)))
    return exposure_min_us, _clamp(gain, gain_min_db, gain_max_db)


class Camera:
    """最小可用的 MVS 取流封装。"""

    def __init__(self, sdk, serial: str | None = None, index: int = 0):
        self.sdk = sdk
        self.serial = serial
        self.index = index
        self.cam = None
        self.device: Device | None = None
        self.notes: list[str] = []
        self._dst = None
        self._param = None

    # ------------------------------------------------------------- 生命周期
    def open(self) -> Camera:
        sdk = self.sdk
        sdk.MvCamera.MV_CC_Initialize()
        devices = list_devices(sdk)
        if not devices:
            raise MvsError("没有枚举到设备；检查 USB3 连接与 MVS 驱动")

        if self.serial:
            match = [d for d in devices if d.serial == self.serial]
            if not match:
                raise MvsError(
                    f"序列号 {self.serial!r} 不存在；可用: "
                    + ", ".join(f"{d.serial}({d.model})" for d in devices)
                )
            self.device = match[0]
        else:
            self.device = devices[self.index]

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
        bayer_quality: int = 1,
        grab_strategy: str = "LatestImagesOnly",
        buffer_count: int = 3,
    ) -> dict:
        sdk, cam = self.sdk, self.cam
        self.notes = []

        def hard(ret, what):
            if ret != 0:
                raise MvsError(f"{what} failed: 0x{ret & 0xFFFFFFFF:08x}")

        def soft(ret, what):
            if ret != 0:
                self.notes.append(f"{what}: 0x{ret & 0xFFFFFFFF:08x}（忽略）")

        # 先关自动曝光/增益，再写固定值
        if exposure_auto:
            hard(
                cam.MV_CC_SetEnumValueByString("ExposureAuto", exposure_auto),
                "ExposureAuto",
            )
        if gain_auto:
            hard(cam.MV_CC_SetEnumValueByString("GainAuto", gain_auto), "GainAuto")
        # 曝光/增益：<=0 与 <0 分别表示“保持设备值”。增益的 0 dB 是合法值（最小值），
        # 所以它不能用 0 当哨兵值。
        if exposure_us > 0:
            hard(cam.MV_CC_SetFloatValue("ExposureTime", exposure_us), "ExposureTime")
        if gain_db >= 0:
            hard(cam.MV_CC_SetFloatValue("Gain", gain_db), "Gain")

        # 自由运行
        hard(cam.MV_CC_SetEnumValue("TriggerMode", sdk.MV_TRIGGER_MODE_OFF), "TriggerMode")

        if fps > 0:
            hard(
                cam.MV_CC_SetBoolValue("AcquisitionFrameRateEnable", True),
                "AcquisitionFrameRateEnable",
            )
            hard(
                cam.MV_CC_SetFloatValue("AcquisitionFrameRate", fps),
                "AcquisitionFrameRate",
            )

        # Bayer 插值质量 = 驱动的 bayer_conversion_quality: 1
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
    def _float_range(self, key: str) -> tuple[float, float, float] | None:
        st = self.sdk.MVCC_FLOATVALUE()
        if self.cam.MV_CC_GetFloatValue(key, st) != 0:
            return None
        return float(st.fCurValue), float(st.fMin), float(st.fMax)

    def set_exposure_us(self, microseconds: float) -> None:
        ret = self.cam.MV_CC_SetFloatValue("ExposureTime", float(microseconds))
        if ret != 0:
            raise MvsError(f"设置 ExposureTime={microseconds:g}us 失败: 0x{ret & 0xFFFFFFFF:08x}")

    def set_gain_db(self, db: float) -> None:
        ret = self.cam.MV_CC_SetFloatValue("Gain", float(db))
        if ret != 0:
            raise MvsError(f"设置 Gain={db:g}dB 失败: 0x{ret & 0xFFFFFFFF:08x}")

    def brightness(self, frames: int = 3, timeout_ms: int = 1000) -> float:
        """取若干帧原始平面的平均灰度（0-255），用于测光。

        直接用未转换的 Bayer/灰度平面，不做去马赛克，测光够用且很快。
        """
        values = []
        for _ in range(frames):
            raw = self.read_raw(timeout_ms)
            if raw is not None:
                values.append(float(raw[0].mean()))
        if not values:
            raise MvsError("测光失败：取不到帧")
        return sum(values) / len(values)

    def auto_expose(
        self,
        *,
        target: float = 110.0,
        tolerance: float = 12.0,
        exposure_upper_us: float | None = None,
        gain_max_db: float | None = None,
        max_iters: int = 10,
        settle_frames: int = 2,
        measure_frames: int = 3,
        verbose=None,
    ) -> dict:
        """闭环测光：调整曝光，曝光到上限后再用增益补。

        本机型没有 Brightness（目标灰度）特性，相机的 ExposureAuto/GainAuto 无法指定
        目标值，所以只能自己量灰度、自己算。曝光上限默认由调用方按帧率约束传进来
        （曝光越长，能达到的最高帧率越低）；增益上限默认取设备的 Gain 最大值。

        返回 {exposure_us, gain_db, brightness, iterations, reached, reason, history}；
        reason 是 converged / at_limits / max_iters。
        """
        cam = self.cam

        def must(ret, what):
            if ret != 0:
                raise MvsError(f"{what} failed: 0x{ret & 0xFFFFFFFF:08x}")

        # 必须先关掉自动曝光/增益，否则手动写的值会被相机自己覆盖回去
        must(cam.MV_CC_SetEnumValueByString("ExposureAuto", "Off"), "ExposureAuto=Off")
        must(cam.MV_CC_SetEnumValueByString("GainAuto", "Off"), "GainAuto=Off")

        exposure_range = self._float_range("ExposureTime")
        gain_range = self._float_range("Gain")
        if exposure_range is None or gain_range is None:
            raise MvsError("读不到 ExposureTime / Gain 的取值范围")
        exposure_min_us, exposure_max_dev = exposure_range[1], exposure_range[2]
        gain_min_db, gain_max_dev = gain_range[1], gain_range[2]

        exposure_upper_us = (
            exposure_max_dev if exposure_upper_us is None else float(exposure_upper_us)
        )
        exposure_upper_us = _clamp(exposure_upper_us, exposure_min_us, exposure_max_dev)
        gain_ceiling = gain_max_dev if gain_max_db is None else min(gain_max_db, gain_max_dev)

        exposure_us = _clamp(exposure_range[0], exposure_min_us, exposure_upper_us)
        gain_db = _clamp(gain_range[0], gain_min_db, gain_ceiling)

        history: list[dict] = []
        reached = False
        reason = "max_iters"
        for _ in range(max_iters):
            self.set_exposure_us(exposure_us)
            self.set_gain_db(gain_db)
            for _ in range(settle_frames):
                self.read_raw()  # 丢掉切换设置后的过渡帧
            mean = self.brightness(measure_frames)
            history.append(
                {
                    "exposure_us": round(exposure_us, 1),
                    "gain_db": round(gain_db, 2),
                    "brightness": round(mean, 1),
                }
            )
            if verbose:
                verbose(
                    f"  测光 {len(history):2d}: 曝光 {exposure_us:8.1f} us  "
                    f"增益 {gain_db:5.2f} dB  ->  灰度 {mean:6.1f}  (目标 {target:g})"
                )
            if abs(mean - target) <= tolerance:
                reached = True
                reason = "converged"
                break
            new_exposure, new_gain = next_exposure_step(
                mean,
                target,
                exposure_us,
                gain_db,
                exposure_min_us=exposure_min_us,
                exposure_max_us=exposure_upper_us,
                gain_min_db=gain_min_db,
                gain_max_db=gain_ceiling,
            )
            if abs(new_exposure - exposure_us) < 1e-9 and abs(new_gain - gain_db) < 1e-9:
                # 曝光和增益都已经顶到边界，再迭代只是白跑
                reason = "at_limits"
                break
            exposure_us, gain_db = new_exposure, new_gain

        return {
            "exposure_us": exposure_us,
            "gain_db": gain_db,
            "brightness": history[-1]["brightness"] if history else float("nan"),
            "iterations": len(history),
            "reached": reached,
            "reason": reason,
            "history": history,
        }

    def read(self, timeout_ms: int = 1000) -> Frame | None:
        """取一帧（含 SDK 转 BGR8）；没有数据返回 None。"""
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
