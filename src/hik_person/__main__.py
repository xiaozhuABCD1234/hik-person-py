"""hik-person: HIKROBOT MVS 相机 -> YOLO 人体检测 -> rerun。

uv run hik-person --list
uv run hik-person --duration 30
uv run hik-person --no-rerun --max-frames 300
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from .colors import pixel_type_name
from .mvs import Camera, MvsError, ensure_runtime_environment, list_devices, load_sdk

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # src/hik_person/__main__.py -> 项目根
DEFAULT_WEIGHTS = PROJECT_ROOT / "models" / "best.pt"


def resolve_weights(explicit: str | None) -> str:
    if explicit:
        return explicit
    if DEFAULT_WEIGHTS.is_file():
        return str(DEFAULT_WEIGHTS)
    raise FileNotFoundError(
        f"找不到 {DEFAULT_WEIGHTS}；用 --weights 指定，或把 best.pt 复制到 models/"
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="hik-person",
        description="HIKROBOT MVS 相机 -> YOLO 人体检测 -> rerun；"
        "颜色统一为 BGR（SDK 转 BGR8_Packed）",
    )
    p.add_argument("--list", action="store_true", help="枚举相机后退出")
    p.add_argument("--serial", default="DA2360141", help="相机序列号")
    p.add_argument("--index", type=int, default=0, help="按枚举序号选相机")
    p.add_argument("--weights", default=None, help="默认 models/best.pt")
    p.add_argument("--conf", type=float, default=0.35)
    p.add_argument("--iou", type=float, default=0.50)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", default="", help="'', cuda:0, cpu")
    p.add_argument(
        "--detect-hz", type=float, default=15.0, help="推理/可视化频率（相机更快就跳帧）"
    )
    p.add_argument("--camera-fps", type=float, default=15.0, help="相机帧率上限，0 = 保持设备设置")
    p.add_argument("--exposure-us", type=float, default=8000.0, help="曝光(us)，0 = 保持设备设置")
    p.add_argument("--gain-db", type=float, default=0.0, help="增益(dB)，0 = 保持设备设置")
    p.add_argument("--no-rerun", action="store_true", help="不开 rerun，只打印")
    p.add_argument(
        "--rerun-save",
        default="",
        help="把 rerun 记录写成 .rrd 文件（不弹窗口，适合无桌面或留证据）",
    )
    p.add_argument("--save-dir", default="", help="检测到人时保存标注图")
    p.add_argument("--duration", type=float, default=0.0, help="跑多少秒，0 = 不限")
    p.add_argument("--max-frames", type=int, default=0, help="处理多少帧后退出")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        ensure_runtime_environment()  # 必要时 execve 重启自己，补齐 MVS 环境
        sdk = load_sdk()
    except MvsError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.list:
        for device in list_devices(sdk):
            print(f"[{device.index}] {device.layer:5s} {device.model}  sn={device.serial}")
        sdk.MvCamera.MV_CC_Finalize()
        return 0

    from .detect import PersonDetector

    try:
        weights = resolve_weights(args.weights)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    detector = PersonDetector(
        weights,
        device=args.device or None,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
    )

    view = None
    if not args.no_rerun:
        from .viz import RerunView

        rrd = args.rerun_save or None
        view = RerunView(spawn=not rrd, save_path=rrd)
        if rrd:
            print(f"rerun 记录 -> {rrd}")
            print(f"  打开方式: rerun {rrd}   （或拖进 rerun viewer）")

    save_dir = Path(args.save_dir) if args.save_dir else None
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)

    period = 1.0 / max(args.detect_hz, 0.1)
    # 容忍 10%：相机帧率正好是 detect_hz 的整数倍时，帧到达时刻会恰好压在周期边界上，
    # 那时纯靠抖动决定处理还是丢掉，会白丢约一半的帧（实测 15/15 与 30/15 都会）。
    min_gap = period * 0.9
    processed = skipped = 0
    last_count = -1
    t_start = t_last = t_last_save = time.monotonic()

    try:
        with Camera(sdk, serial=args.serial or None, index=args.index) as cam:
            settings = cam.configure(
                exposure_us=args.exposure_us, gain_db=args.gain_db, fps=args.camera_fps
            )
            for note in cam.notes:
                print(f"warning: {note}", file=sys.stderr)
            print("camera: " + ", ".join(f"{k}={v}" for k, v in settings.items()))
            print(
                f"model : {detector.weights} classes={detector.names} "
                f"filter={detector.classes} device={detector.device}"
            )
            cam.start()
            print(f"running: detect {args.detect_hz:g} Hz, Ctrl-C 退出")

            while True:
                frame = cam.read(timeout_ms=1000)
                if frame is None:
                    continue

                now = time.monotonic()
                # 还没到检测时刻：这帧只用来保持取流节奏，不做推理
                # （注意不能把 due 判断放在 read 之前，否则会先把等到的那一帧丢掉）
                if now - t_last < min_gap:
                    skipped += 1
                    continue
                t_last = now

                detections, annotated = detector(frame.bgr)
                processed += 1
                count = len(detections)

                if view is not None:
                    view.log_frame(frame, detections, annotated)
                if count and count != last_count:
                    print(
                        f"[{time.strftime('%H:%M:%S')}] person x{count} "
                        f"({', '.join(f'{d.conf:.2f}' for d in detections)})"
                    )
                last_count = count

                now = time.monotonic()
                if save_dir and count and now - t_last_save >= 1.0:
                    import cv2

                    t_last_save = now
                    path = save_dir / time.strftime("person_%Y%m%d_%H%M%S.jpg")
                    cv2.imwrite(str(path), annotated)
                    print(f"saved {path}")

                if processed % 50 == 0:
                    print(
                        f"... {processed} frames, "
                        f"{processed / (now - t_start):.1f} fps 端到端, "
                        f"{skipped} 帧跳推理, 最近一帧 {frame.width}x{frame.height} "
                        f"{pixel_type_name(frame.pixel_type)}"
                    )

                if args.max_frames and processed >= args.max_frames:
                    break
                if args.duration and now - t_start >= args.duration:
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
