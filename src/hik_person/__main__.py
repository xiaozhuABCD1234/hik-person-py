# SPDX-License-Identifier: AGPL-3.0-or-later
"""hik-person: HIKROBOT MVS 相机 -> YOLO 人体检测 -> rerun。

参数都放在 config.toml（见 `--show-config` 看生效值）；命令行只用来临时覆盖。

    uv run hik-person --list            # 枚举相机
    uv run hik-person                   # 按 config.toml 跑
    uv run hik-person --show-config     # 打印合并后的生效参数
    uv run hik-person --duration 30
    uv run hik-person --no-rerun --max-frames 300
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import fields

import numpy as np

from .colors import pixel_type_name
from .config import (
    EXPOSURE_MODES,
    TASKS,
    Settings,
    apply_cli,
    default_config_path,
    load_settings,
    resolve_path,
)
from .mvs import (
    Camera,
    MvsError,
    ensure_runtime_environment,
    list_devices,
    load_sdk,
    parse_feature_spec,
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="hik-person",
        description="HIKROBOT MVS 相机 -> YOLO 人体检测 -> rerun。"
        "参数默认读 config.toml，命令行只做临时覆盖",
    )
    p.add_argument(
        "--config",
        default=str(default_config_path()),
        help="参数文件（默认项目根的 config.toml）",
    )
    p.add_argument("--show-config", action="store_true", help="打印生效参数后退出")
    p.add_argument("--list", action="store_true", help="枚举相机后退出")

    # 下面这些 default=None 表示"命令行没给"，值交给 config.toml / 兜底默认值
    p.add_argument("--serial", default=None, help="覆盖 [camera].serial")
    p.add_argument("--index", type=int, default=None, help="覆盖 [camera].index")
    p.add_argument("--camera-fps", type=float, default=None, help="覆盖 [camera].camera_fps")
    p.add_argument("--exposure-us", type=float, default=None, help="覆盖 [camera].exposure_us")
    p.add_argument("--gain-db", type=float, default=None, help="覆盖 [camera].gain_db")
    p.add_argument(
        "--exposure-mode",
        default=None,
        choices=EXPOSURE_MODES,
        help="覆盖 [camera].exposure_mode：sdk（相机自动）/manual",
    )
    p.add_argument(
        "--brightness", type=float, default=None, help="覆盖 [camera].brightness（AE 目标灰度）"
    )
    p.add_argument("--auto-gain-max-db", type=float, default=None)
    p.add_argument(
        "--exposure-start-us",
        type=float,
        default=None,
        help="覆盖 [camera].exposure_start_us：开 AE 前先钉住的曝光起点（0 = 用相机现存值）",
    )
    p.add_argument("--exposure-fps-headroom", type=float, default=None)
    p.add_argument("--read-timeout-ms", type=int, default=None)
    p.add_argument(
        "--feature",
        action="append",
        default=None,
        metavar="NAME=VALUE",
        help="覆盖 [camera].feature（给了就整体替换，可重复）。"
        "例：--feature ADCBitDepth=Bits_8 解锁 249 fps（写别的机型的符号名会自动改写）",
    )
    p.add_argument("--weights", default=None, help="覆盖 [detector].weights")
    p.add_argument("--task", default=None, choices=TASKS, help="覆盖 [detector].task")
    p.add_argument("--conf", type=float, default=None, help="覆盖 [detector].conf")
    p.add_argument("--iou", type=float, default=None, help="覆盖 [detector].iou")
    p.add_argument("--imgsz", type=int, default=None, help="覆盖 [detector].imgsz")
    p.add_argument("--device", default=None, help="覆盖 [detector].device：''/cuda:0/cpu")
    p.add_argument("--kpt-conf", type=float, default=None, help="覆盖 [detector].kpt_conf")
    p.add_argument("--detect-hz", type=float, default=None, help="覆盖 [run].detect_hz")
    p.add_argument("--duration", type=float, default=None, help="覆盖 [run].duration")
    p.add_argument("--max-frames", type=int, default=None, help="覆盖 [run].max_frames")
    p.add_argument("--save-dir", default=None, help="覆盖 [run].save_dir")

    rerun_group = p.add_mutually_exclusive_group()
    rerun_group.add_argument(
        "--rerun", dest="rerun_enabled", action="store_true", default=None, help="开 rerun viewer"
    )
    rerun_group.add_argument(
        "--no-rerun",
        dest="rerun_enabled",
        action="store_false",
        default=None,
        help="不开 rerun，只打印",
    )
    p.add_argument(
        "--rerun-save",
        default=None,
        help="覆盖 [rerun].save：写成 .rrd 且不弹窗口（默认不写文件）",
    )
    p.add_argument(
        "--rerun-memory-limit",
        default=None,
        help='覆盖 [rerun].memory_limit，如 "512MB"/"75%%"',
    )
    p.add_argument(
        "--rerun-keep-history",
        action="store_true",
        default=None,
        help="覆盖 [rerun].keep_history=true：保留时间轴（内存会一直涨）",
    )
    p.add_argument(
        "--rerun-jpeg-quality",
        type=int,
        default=None,
        help="覆盖 [rerun].jpeg_quality：1-100，0 = 送原图（更吃内存）",
    )

    pose3d_group = p.add_mutually_exclusive_group()
    pose3d_group.add_argument(
        "--pose3d",
        dest="pose3d_enabled",
        action="store_true",
        default=None,
        help="覆盖 [pose3d].enabled=true：rerun 右下角开 3D 骨架小窗",
    )
    pose3d_group.add_argument(
        "--no-pose3d",
        dest="pose3d_enabled",
        action="store_false",
        default=None,
        help="覆盖 [pose3d].enabled=false：关掉 3D 小窗，不加载深度模型",
    )
    p.add_argument("--pose3d-weights", default=None, help="覆盖 [pose3d].weights")
    p.add_argument("--pose3d-imgsz", type=int, default=None, help="覆盖 [pose3d].imgsz")
    p.add_argument("--pose3d-hz", type=float, default=None, help="覆盖 [pose3d].hz")
    p.add_argument("--pose3d-fov-deg", type=float, default=None, help="覆盖 [pose3d].fov_deg")
    p.add_argument(
        "--pose3d-depth-view",
        dest="pose3d_depth_view",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="覆盖 [pose3d].depth_view：左列再放一块估计深度图",
    )
    p.add_argument(
        "--pose3d-kalman",
        dest="pose3d_kalman",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="覆盖 [pose3d].kalman：3D 关键点卡尔曼滤波（默认开）",
    )
    return p


def print_settings(settings: Settings, config_path: str) -> None:
    print(f"# {config_path}")
    for spec in fields(settings):
        print(f"{spec.name} = {getattr(settings, spec.name)!r}")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        settings, config_warnings = load_settings(args.config)
    except (OSError, ValueError) as exc:  # TOMLDecodeError 也是 ValueError
        print(f"error: 读参数文件失败: {exc}", file=sys.stderr)
        return 2
    settings = apply_cli(settings, args)
    for warning in config_warnings:
        print(f"warning: {warning}", file=sys.stderr)

    if args.show_config:
        print_settings(settings, args.config)
        return 0

    if settings.exposure_mode not in EXPOSURE_MODES:
        print(
            f"error: [camera].exposure_mode = {settings.exposure_mode!r} 不认识，"
            f"可选 {EXPOSURE_MODES}",
            file=sys.stderr,
        )
        return 2

    if settings.task not in TASKS:
        print(
            f"error: [detector].task = {settings.task!r} 不认识，可选 {TASKS}",
            file=sys.stderr,
        )
        return 2

    for spec in settings.feature:
        try:
            parse_feature_spec(spec)
        except ValueError as exc:
            print(f"error: [camera].feature 里的 {exc}", file=sys.stderr)
            return 2

    try:
        ensure_runtime_environment()  # 必要时 execve 重启自己，补齐 MVS 环境
        sdk = load_sdk()
    except MvsError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.list:
        for device in list_devices(sdk):
            tag = "  (虚拟, 打不开)" if device.virtual else ""
            print(f"[{device.index}] {device.layer:5s} {device.model}  sn={device.serial}{tag}")
        sdk.MvCamera.MV_CC_Finalize()
        return 0

    weights_value = settings.weights or Settings().weights
    weights_path = resolve_path(weights_value)
    if not weights_path.is_file():
        print(
            f"error: 找不到权重 {weights_path}（[detector].weights = {weights_value!r}）",
            file=sys.stderr,
        )
        return 2

    try:
        if settings.task == "pose":
            from .pose import PoseEstimator

            estimator = PoseEstimator(
                weights_path,
                device=settings.device or None,
                conf=settings.conf,
                imgsz=settings.imgsz,
                kpt_conf=settings.kpt_conf,
            )
        else:
            from .detect import PersonDetector

            estimator = PersonDetector(
                weights_path,
                device=settings.device or None,
                conf=settings.conf,
                iou=settings.iou,
                imgsz=settings.imgsz,
            )
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    view = None
    if settings.rerun_enabled:
        from .viz import RerunView

        rrd = settings.rerun_save or None
        view = RerunView(
            spawn=not rrd,
            save_path=rrd,
            kpt_conf=settings.kpt_conf,
            keep_history=settings.rerun_keep_history,
            memory_limit=settings.rerun_memory_limit,
            server_memory_limit=settings.rerun_server_memory_limit,
            jpeg_quality=settings.rerun_jpeg_quality,
            port=settings.rerun_port,
            show_pose3d=settings.pose3d_enabled and settings.task == "pose",
            show_depth=(
                settings.pose3d_enabled and settings.pose3d_depth_view and settings.task == "pose"
            ),
            eye=settings.pose3d_eye,
            look_at=settings.pose3d_look_at,
            eye_up=settings.pose3d_eye_up,
        )
        if rrd:
            print(f"rerun 记录 -> {rrd}")
            print(f"  打开方式: rerun {rrd}   （或拖进 rerun viewer）")
        else:
            print(
                f"rerun: viewer 已启动（记忆历史 {settings.rerun_keep_history}，"
                f"内存上限 {settings.rerun_memory_limit}，图像 JPEG {settings.rerun_jpeg_quality}）"
            )
            print("  不要窗口就用 --no-rerun")

    save_dir = resolve_path(settings.save_dir) if settings.save_dir else None
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)

    # 单目 3D 骨架（只在 pose + rerun 都在时才有意义）
    depth_estimator = None
    tracker = None
    if view is not None and settings.pose3d_enabled and settings.task == "pose":
        from .depth import DepthEstimator
        from .pose import NUM_KEYPOINTS
        from .pose3d import Intrinsics, lift_keypoints, to_display, valid_mask
        from .track import Pose3DTracker

        depth_path = resolve_path(settings.pose3d_weights)
        if not depth_path.is_file():
            print(
                f"error: 找不到深度权重 {depth_path}（[pose3d].weights = "
                f"{settings.pose3d_weights!r}）；不想用 3D 小窗就加 --no-pose3d",
                file=sys.stderr,
            )
            return 2
        depth_estimator = DepthEstimator(
            depth_path,
            device=settings.device or None,
            imgsz=settings.pose3d_imgsz,
        )
        if settings.pose3d_kalman:
            tracker = Pose3DTracker(
                process_noise=settings.pose3d_kalman_process,
                measurement_noise=settings.pose3d_kalman_measurement,
            )

    print(f"config: {args.config}")
    print(
        f"run   : {settings.task}, detect {settings.detect_hz:g} Hz"
        f"（相机 {settings.camera_fps:g} fps 上限）, imgsz {settings.imgsz}, "
        f"conf {settings.conf:g}, 曝光 {settings.exposure_mode}, "
        f"feature {settings.feature or '[]'}"
    )
    if depth_estimator is not None:
        extras = []
        if settings.pose3d_depth_view:
            extras.append("深度图")
        if settings.pose3d_kalman:
            extras.append(
                f"卡尔曼 q={settings.pose3d_kalman_process:g} "
                f"r={settings.pose3d_kalman_measurement:g}"
            )
        print(
            f"pose3d: {settings.pose3d_weights} imgsz {settings.pose3d_imgsz}, "
            f"{settings.pose3d_hz:g} Hz, 视场角 {settings.pose3d_fov_deg:g}°"
            + (f"，{'，'.join(extras)}" if extras else "")
        )

    period = 1.0 / max(settings.detect_hz, 0.1)
    # 帧到达时刻正好压在周期边界上时纯靠抖动决定去留，会白丢约一半的帧
    # （实测 15/15 与 30/15 都会），所以留一点容差。倍数在 config.toml 里。
    min_gap = period * settings.rate_tolerance
    # 深度模型的频率与 detect 分开：它比 pose 贵，没必要跟满
    period_3d = 1.0 / settings.pose3d_hz if settings.pose3d_hz > 0 else 0.0
    t_last_3d = 0.0
    processed = skipped = 0
    last_count = -1
    t_start = t_last = t_last_save = time.monotonic()

    try:
        with Camera(sdk, serial=settings.serial or None, index=settings.index) as cam:
            auto = settings.exposure_mode == "sdk"
            # 相机 AE 的曝光上限：按帧率留 0.9 的余量，否则 AE 只能靠拉增益变亮
            auto_exposure_upper = (
                round(settings.exposure_fps_headroom * 1e6 / settings.camera_fps)
                if settings.camera_fps > 0
                else None
            )
            cam.configure(
                exposure_us=settings.exposure_us,
                gain_db=settings.gain_db,
                fps=settings.camera_fps,
                # Continuous = 交给相机自己的自动曝光/自动增益
                exposure_auto="Continuous" if auto else "Off",
                gain_auto="Continuous" if auto else "Off",
                auto_brightness=settings.brightness if auto and settings.brightness > 0 else None,
                auto_exposure_upper_us=auto_exposure_upper if auto else None,
                auto_gain_upper_db=settings.auto_gain_max_db if auto else None,
                exposure_start_us=settings.exposure_start_us if auto else None,
                feature_overrides=settings.feature,
            )
            for item in cam.applied_features:
                print(f"feature: {item}")
            for note in cam.notes:
                print(f"相机: {note}")
            cam.start()

            if auto:
                # 相机自己的 AE：等它自己收敛，程序不插手
                print(
                    f"曝光/增益: 相机自动（目标灰度 {settings.brightness:g}，"
                    f"曝光上限 {auto_exposure_upper} us，增益上限 "
                    f"{settings.auto_gain_max_db:g} dB）"
                    if auto_exposure_upper is not None
                    else "曝光/增益: 相机自动"
                )
                settled, converged = cam.wait_auto_settled()
                tail = "" if converged else "（还在变，未收敛）"
                print(
                    f"  -> 曝光 {settled['exposure_us']:.0f} us，"
                    f"增益 {settled['gain_db']:.2f} dB{tail}"
                )
            elif settings.exposure_mode == "manual":
                print(f"曝光/增益: 固定 {settings.exposure_us:.0f} us / {settings.gain_db:g} dB")

            camera_settings = cam.settings()
            print("camera: " + ", ".join(f"{k}={v}" for k, v in camera_settings.items()))
            print(
                f"model : {estimator.weights} task={settings.task} classes={estimator.names} "
                f"device={estimator.device}"
            )
            print(f"running: {settings.task} {settings.detect_hz:g} Hz, Ctrl-C 退出")

            while True:
                frame = cam.read(timeout_ms=settings.read_timeout_ms)
                if frame is None:
                    continue

                now = time.monotonic()
                # 还没到检测时刻：这帧只用来保持取流节奏，不做推理
                # （注意不能把这个判断放在 read 之前，否则会先把等到的那一帧丢掉）
                if now - t_last < min_gap:
                    skipped += 1
                    continue
                t_last = now

                # 不在这里 result.plot()：rerun 侧用结构化框/骨架重画，只有存盘时才
                # 需要标注图，到那时再 estimator.annotate(result) 按需生成。
                instances, result = estimator(frame.bgr)
                processed += 1
                count = len(instances)

                if view is not None:
                    # 送相机原图；框/关键点/骨架由 rerun 用结构化数据画
                    if settings.task == "pose":
                        view.log_frame(frame, frame.bgr, poses=instances)
                    else:
                        view.log_frame(frame, frame.bgr, detections=instances)
                if count and count != last_count:
                    print(
                        f"[{time.strftime('%H:%M:%S')}] person x{count} "
                        f"({', '.join(f'{d.conf:.2f}' for d in instances)})"
                    )
                last_count = count

                now = time.monotonic()
                if save_dir and count and now - t_last_save >= settings.save_min_interval:
                    import cv2

                    t_last_save = now
                    path = save_dir / time.strftime("person_%Y%m%d_%H%M%S.jpg")
                    cv2.imwrite(str(path), estimator.annotate(result))
                    print(f"saved {path}")

                if depth_estimator is not None and now - t_last_3d >= period_3d:
                    t_last_3d = now
                    # 每一轮都算深度：深度图本身也要看，不只给骨架用
                    depth = depth_estimator(frame.bgr)
                    if settings.pose3d_depth_view:
                        view.log_depth(depth)
                    intrinsics = Intrinsics.from_fov(
                        frame.width, frame.height, settings.pose3d_fov_deg
                    )
                    if instances:
                        boxes = [pose.xyxy for pose in instances]
                        confs = np.stack([pose.kpt_conf for pose in instances])
                        lifted = np.stack(
                            [
                                lift_keypoints(pose.keypoints, depth, intrinsics)
                                for pose in instances
                            ]
                        )
                    else:
                        boxes = []
                        confs = np.zeros((0, NUM_KEYPOINTS), np.float32)
                        lifted = np.zeros((0, NUM_KEYPOINTS, 3), np.float32)
                    if tracker is not None:
                        items = tracker.update(
                            boxes, lifted, confs, threshold=settings.kpt_conf, t=now
                        )
                    else:
                        items = [
                            to_display(
                                lifted[i], valid_mask(lifted[i], confs[i], settings.kpt_conf)
                            )
                            for i in range(len(boxes))
                        ]
                    view.log_poses3d(items)

                if processed % 50 == 0:
                    print(
                        f"... {processed} frames, "
                        f"{processed / (now - t_start):.1f} fps 端到端, "
                        f"{skipped} 帧跳推理, 最近一帧 {frame.width}x{frame.height} "
                        f"{pixel_type_name(frame.pixel_type)}"
                    )

                if settings.max_frames and processed >= settings.max_frames:
                    break
                if settings.duration and now - t_start >= settings.duration:
                    break
    except KeyboardInterrupt:
        print("\ninterrupted")
    except MvsError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    elapsed = time.monotonic() - t_start
    print(f"done: {processed} frames / {elapsed:.1f}s ({processed / max(elapsed, 1e-6):.2f} fps)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
