# SPDX-License-Identifier: AGPL-3.0-or-later
"""分环节计时：等帧 / SDK 转换 / YOLO 推理 / rerun 前的颜色转换。

uv run python tools/bench.py
uv run python tools/bench.py --frames 60 --imgsz 480
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from ctypes import POINTER, c_ubyte

from hik_person.colors import bgr_to_rgb, pixel_type_name
from hik_person.config import default_config_path, load_settings, resolve_path
from hik_person.detect import PersonDetector
from hik_person.mvs import Camera, load_sdk


def timed(fn, repeat):
    samples = []
    for _ in range(repeat):
        t = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t) * 1000.0)
    return samples


def report(name: str, samples: list[float]) -> None:
    print(
        f"{name:34s} median {statistics.median(samples):7.2f} ms"
        f"   mean {statistics.fmean(samples):7.2f} ms"
        f"   max {max(samples):7.2f} ms"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(default_config_path()))
    parser.add_argument("--serial", default=None, help="默认取 config.toml 的 [camera].serial")
    parser.add_argument("--weights", default=None, help="默认取 config.toml 的 [detector].weights")
    parser.add_argument("--frames", type=int, default=40)
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--camera-fps", type=float, default=None)
    args = parser.parse_args()

    settings, warnings = load_settings(args.config)
    for warning in warnings:
        print(f"warning: {warning}", file=sys.stderr)

    sdk = load_sdk()
    detector = PersonDetector(
        resolve_path(args.weights or settings.weights),
        device=args.device if args.device is not None else (settings.device or None),
        imgsz=args.imgsz or settings.imgsz,
    )
    camera_fps = args.camera_fps if args.camera_fps is not None else settings.camera_fps
    serial = args.serial or settings.serial
    print(f"device={detector.device} imgsz={detector.imgsz} frames={args.frames}")

    with Camera(sdk, serial=serial or None) as cam:
        cam.configure(
            exposure_us=settings.exposure_us,
            gain_db=settings.gain_db,
            fps=camera_fps,
            feature_overrides=settings.feature,
        )
        cam.start()

        # 先抓一帧原始缓冲，用来单独量“纯转换”的耗时（不含等帧）
        raw = None
        for _ in range(200):
            raw = cam.read_raw()
            if raw is not None:
                break
        if raw is None:
            print("error: 没取到帧")
            return 1
        buf, width, height, pixel_type, frame_len = raw
        ptr = buf.ctypes.data_as(POINTER(c_ubyte))
        print(f"frame={width}x{height} {pixel_type_name(pixel_type)} camera_fps={camera_fps:g}")

        report(
            "SDK 转 BGR（纯计算, 不含等帧）",
            timed(
                lambda: cam.convert_to_bgr(ptr, width, height, pixel_type, frame_len), args.frames
            ),
        )
        bgr = cam.convert_to_bgr(ptr, width, height, pixel_type, frame_len)
        report("BGR -> RGB（rerun 需要）", timed(lambda: bgr_to_rgb(bgr), args.frames))
        report("取一帧（含等帧）", timed(lambda: cam.read(), args.frames))

        detector(bgr)  # warmup
        report("YOLO 推理", timed(lambda: detector(bgr), min(args.frames, 20)))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
