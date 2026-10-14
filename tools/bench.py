"""分环节计时：等帧 / SDK 转换 / YOLO 推理 / rerun 前的颜色转换。

uv run python tools/bench.py
uv run python tools/bench.py --frames 60 --imgsz 480
"""

from __future__ import annotations

import argparse
import statistics
import time
from ctypes import POINTER, c_ubyte

from hik_person.colors import bgr_to_rgb, pixel_type_name
from hik_person.detect import PersonDetector
from hik_person.mvs import Camera, load_sdk

DEFAULT_WEIGHTS = "models/best.pt"


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
    parser.add_argument("--serial", default="DA2360141")
    parser.add_argument("--weights", default=DEFAULT_WEIGHTS)
    parser.add_argument("--frames", type=int, default=40)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="")
    parser.add_argument("--camera-fps", type=float, default=30.0)
    args = parser.parse_args()

    sdk = load_sdk()
    detector = PersonDetector(args.weights, device=args.device or None, imgsz=args.imgsz)
    print(f"device={detector.device} imgsz={detector.imgsz} frames={args.frames}")

    with Camera(sdk, serial=args.serial) as cam:
        cam.configure(exposure_us=8000.0, fps=args.camera_fps)
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
        print(
            f"frame={width}x{height} {pixel_type_name(pixel_type)} camera_fps={args.camera_fps:g}"
        )

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
