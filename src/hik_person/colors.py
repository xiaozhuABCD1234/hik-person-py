# SPDX-License-Identifier: AGPL-3.0-or-later
"""像素格式与颜色转换。

相机原生输出 BayerRG8，转换统一走 SDK：MV_CC_SetBayerCvtQuality(1) 配高质量插值，
MV_CC_ConvertPixelTypeEx 转到 PixelType_Gvsp_BGR8_Packed。也就是本包内部统一使用
BGR 顺序（OpenCV 的默认顺序）；走 SDK 就不会碰到 Bayer 排布的命名歧义。

只有需要自己解 Bayer 原始缓冲时要注意：MVS/GenICam 对 Bayer 排布的命名与 OpenCV
相差一个像素，MVS 的 BayerRG8 必须用 cv2.COLOR_BayerBG2BGR。tools/check_colors.py
在同一帧上实测过（以 SDK 输出为参考）：
    COLOR_BayerBG2BGR -> 各通道均值一致, corr ~0.99
    COLOR_BayerRG2BGR -> B/R 恰好对调, corr ~0.92
"""

from __future__ import annotations

import cv2
import numpy as np

# 数值来自 /opt/MVS/Samples/64/Python/MvImport/PixelType_header.py
PIXEL_TYPES = {
    17301505: "Mono8",
    17301513: "BayerRG8",
    35127316: "RGB8_Packed",
    35127317: "BGR8_Packed",
}
MONO8 = 17301505
BAYER_RG8 = 17301513
RGB8_PACKED = 35127316
BGR8_PACKED = 35127317


def pixel_type_name(value: int) -> str:
    return PIXEL_TYPES.get(int(value), f"0x{int(value) & 0xFFFFFFFF:08x}")


def bayer_rg8_to_bgr(raw: np.ndarray, width: int, height: int) -> np.ndarray:
    """MVS BayerRG8 -> BGR（OpenCV 去马赛克）。

    必须用 OpenCV 的 BG 约定，见模块 docstring 的实测结论。
    """
    plane = np.ascontiguousarray(raw[: width * height].reshape(height, width))
    return cv2.cvtColor(plane, cv2.COLOR_BayerBG2BGR)


def bgr_to_rgb(bgr: np.ndarray) -> np.ndarray:
    """BGR -> RGB。

    rerun 的 rr.Image 把 3 通道数据当 RGB 解读，所以本包内部的 BGR 帧在进 rerun 前
    必须换一次，否则红蓝互换。
    """
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
