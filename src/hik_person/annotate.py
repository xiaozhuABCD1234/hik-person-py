# SPDX-License-Identifier: AGPL-3.0-or-later
"""在标注图上画人数横幅（检测与姿态两条任务线共用）。"""

from __future__ import annotations

import cv2
import numpy as np

BANNER_WIDTH = 330
BANNER_HEIGHT = 56


def draw_count_banner(bgr: np.ndarray, count: int) -> np.ndarray:
    """把人数写在画面左上角：截图和存盘 JPEG 也能直接看到人数。

    返回新数组，不改调用方传进来的那张。有人是绿色，0 人偏灰 —— 一眼能看出当前有没有人。
    """
    annotated = np.ascontiguousarray(bgr).copy()
    cv2.rectangle(annotated, (0, 0), (BANNER_WIDTH, BANNER_HEIGHT), (0, 0, 0), -1)
    cv2.putText(
        annotated,
        f"person: {count}",
        (16, 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.3,
        (0, 255, 0) if count else (150, 150, 150),
        2,
        cv2.LINE_AA,
    )
    return annotated
