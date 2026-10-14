"""验证颜色链路：MVS BayerRG8 -> BGR，三条路径在同一帧上对比。

  1. SDK  MV_CC_ConvertPixelTypeEx(BGR8_Packed)   <- 本包默认取流路径
  2. cv2  COLOR_BayerBG2BGR                       <- MVS/GenICam 命名下的正确约定
  3. cv2  COLOR_BayerRG2BGR                       <- 错误约定，应表现为 B/R 互换

以 1 为参考：2 应几乎重合，3 应将 B/R 对调。

    uv run python tools/check_colors.py
    uv run python tools/check_colors.py --save-dir /tmp/hik-color-check
"""

from __future__ import annotations

import argparse
import sys
from ctypes import POINTER, c_ubyte
from pathlib import Path

import cv2
import numpy as np

from hik_person.colors import BAYER_RG8, pixel_type_name
from hik_person.mvs import Camera, load_sdk


def stats(image: np.ndarray, reference: np.ndarray) -> dict:
    means = image.reshape(-1, 3).mean(axis=0)
    corr = float(
        np.corrcoef(image.reshape(-1).astype(np.float32), reference.reshape(-1).astype(np.float32))[
            0, 1
        ]
    )
    return {"B": means[0], "G": means[1], "R": means[2], "corr": corr}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--serial", default="DA2360141")
    parser.add_argument("--save-dir", default="")
    args = parser.parse_args()

    sdk = load_sdk()
    save_dir = Path(args.save_dir) if args.save_dir else None
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)

    with Camera(sdk, serial=args.serial) as cam:
        cam.configure(exposure_us=8000.0, fps=15.0)
        cam.start()

        got = None
        for _ in range(100):
            got = cam.read_raw()
            if got is not None:
                break
        if got is None:
            print("error: 没有取到帧", file=sys.stderr)
            return 1
        raw, width, height, pixel_type, frame_len = got

        sdk_bgr = cam.convert_to_bgr(
            raw.ctypes.data_as(POINTER(c_ubyte)), width, height, pixel_type, frame_len
        )

    print(
        f"frame : {width}x{height}  {pixel_type_name(pixel_type)}  "
        f"frame_len={frame_len}  (expect {width * height} for 8-bit Bayer/Mono)"
    )

    if pixel_type != BAYER_RG8:
        print(
            f"注意: 相机当前不是 BayerRG8（{pixel_type_name(pixel_type)}），"
            f"下面的 cv2 对比只对 Bayer 有意义"
        )
        if save_dir:
            cv2.imwrite(str(save_dir / "sdk_bgr8.jpg"), sdk_bgr)
        return 0

    plane = np.ascontiguousarray(raw[: width * height].reshape(height, width))

    print()
    print("候选路径对比（参考 = SDK 输出，即本包默认取流路径）")
    print(f"{'path':34s} {'B mean':>8s} {'G mean':>8s} {'R mean':>8s} {'corr':>7s}")
    reference = sdk_bgr
    candidates = (
        ("SDK ConvertPixelTypeEx(BGR8_Packed)", "sdk_bgr8", sdk_bgr),
        (
            "cv2 COLOR_BayerBG2BGR (right)",
            "cv2_bayerbg",
            cv2.cvtColor(plane, cv2.COLOR_BayerBG2BGR),
        ),
        (
            "cv2 COLOR_BayerRG2BGR (wrong)",
            "cv2_bayerrg",
            cv2.cvtColor(plane, cv2.COLOR_BayerRG2BGR),
        ),
    )
    for name, stem, image in candidates:
        s = stats(image, reference)
        print(f"{name:34s} {s['B']:8.2f} {s['G']:8.2f} {s['R']:8.2f} {s['corr']:7.3f}")
        if save_dir:
            ok = cv2.imwrite(str(save_dir / f"{stem}.jpg"), image)
            if not ok:
                print(f"  warning: imwrite 失败 {stem}.jpg (shape={image.shape})")

    print()
    print(
        "判定: BayerBG 的 corr 应接近 1（与 SDK 输出一致），"
        "BayerRG 的 B/R 均值应与 SDK 完全对调、corr 明显更低。"
    )
    if save_dir:
        print(f"图片已存到 {save_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
