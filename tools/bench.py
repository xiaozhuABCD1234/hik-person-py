# SPDX-License-Identifier: AGPL-3.0-or-later
"""分环节计时：等帧 / SDK 转换 / YOLO 推理 / 深度模型 / 跟踪（抬 3D）。

    uv run python tools/bench.py
    uv run python tools/bench.py --frames 60 --imgsz 480
    uv run python tools/bench.py --no-depth          # 不量深度模型（不加载它）

看的是“哪个环节占多少”，用来决定该换权重还是换策略：x 档 pose/depth 一帧合计能吃掉
15 Hz 的 66 ms 预算，而跟踪本身只有 1 ms 级（主要是 17×3 个标量卡尔曼 + JSONL）。
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from ctypes import POINTER, c_ubyte

import numpy as np

from hik_person.colors import bgr_to_rgb, pixel_type_name
from hik_person.config import default_config_path, load_settings, resolve_path
from hik_person.geometry import Intrinsics
from hik_person.mvs import Camera, load_sdk
from hik_person.pose import NUM_KEYPOINTS, PoseEstimator, PoseInstance
from hik_person.track import PersonTracker


def bench_poses(instances, width: int, height: int) -> list[PoseInstance]:
    """拿真检测来计时；画面里没人时造一个，保证量的是跟踪本身而不是“空列表有多快”。"""
    if instances:
        return list(instances)
    points = np.stack(
        [
            np.full(NUM_KEYPOINTS, width * 0.5),
            np.linspace(height * 0.25, height * 0.8, NUM_KEYPOINTS),
        ],
        axis=1,
    ).astype("float32")
    return [
        PoseInstance(
            conf=0.8,
            xyxy=(width * 0.4, height * 0.2, width * 0.6, height * 0.85),
            keypoints=points,
            kpt_conf=np.full(NUM_KEYPOINTS, 0.9, dtype="float32"),
        )
    ]


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
    parser.add_argument("--no-depth", action="store_true", help="不量深度模型（也就不加载它）")
    args = parser.parse_args()

    settings, warnings = load_settings(args.config)
    for warning in warnings:
        print(f"warning: {warning}", file=sys.stderr)

    sdk = load_sdk()
    estimator = PoseEstimator(
        resolve_path(args.weights or settings.weights),
        device=args.device if args.device is not None else (settings.device or None),
        imgsz=args.imgsz or settings.imgsz,
        kpt_conf=settings.kpt_conf,
    )
    camera_fps = args.camera_fps if args.camera_fps is not None else settings.camera_fps
    serial = args.serial or settings.serial
    print(f"device={estimator.device} imgsz={estimator.imgsz} frames={args.frames}")

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

        estimator(bgr)  # warmup
        report("YOLO pose 推理", timed(lambda: estimator(bgr), min(args.frames, 20)))

        instances, _ = estimator(bgr)
        poses = bench_poses(instances, width, height)
        if instances:
            print(f"跟踪计时用真实检测：{len(instances)} 人")
        else:
            print("跟踪计时用合成的假人（画面里没人，否则量出来是空列表有多快）")

        intrinsics = Intrinsics.from_fov(width, height, settings.pose3d_fov_deg)
        # 单目深度的耗时无法在这里造：用一张常量深度图，只量“抬 3D + 滤波 + 关联”
        depth = np.full((height, width), 2.5, dtype=np.float32)
        tracker = PersonTracker(
            kalman=settings.pose3d_kalman,
            process_noise=settings.pose3d_kalman_process,
            measurement_noise=settings.pose3d_kalman_measurement,
            iou_threshold=settings.track_iou_threshold,
            match_max_distance=settings.track_match_max_distance,
            max_missed=settings.track_max_missed,
            max_missed_kpt=settings.track_max_missed_kpt,
            trail_length=settings.track_trail_length,
            trail_interval=settings.track_trail_interval,
            speed_smoothing=settings.track_speed_smoothing,
            speed_deadband=settings.track_speed_deadband,
        )
        clock = [time.monotonic()]

        def track_once():
            clock[0] += 1.0 / 15.0
            tracker.update(poses, depth, intrinsics, threshold=settings.kpt_conf, t=clock[0])

        track_once()  # warmup
        report("跟踪：抬 3D + 卡尔曼 + 关联", timed(track_once, args.frames))

        if not args.no_depth:
            depth_path = resolve_path(settings.pose3d_weights)
            if not depth_path.is_file():
                print(f"跳过深度模型计时：找不到 {depth_path}")
            else:
                from hik_person.depth import DepthEstimator

                depth_estimator = DepthEstimator(
                    depth_path, device=estimator.device, imgsz=settings.pose3d_imgsz
                )
                depth_estimator(bgr)  # warmup
                report(
                    f"深度模型（imgsz {settings.pose3d_imgsz}）",
                    timed(lambda: depth_estimator(bgr), min(args.frames, 20)),
                )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
