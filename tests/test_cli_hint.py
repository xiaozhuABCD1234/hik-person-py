# SPDX-License-Identifier: AGPL-3.0-or-later
"""权重缺失时的提示：默认权重是 x 档且不入库，得让人知道去哪下、怎么退回。"""

from __future__ import annotations

from hik_person.__main__ import weights_download_hint
from hik_person.config import ASSETS_URL, Settings


def test_hint_points_at_the_official_asset_url() -> None:
    hint = weights_download_hint(Settings().weights, "--weights models/yolo26m-pose.pt")
    assert ASSETS_URL in hint
    assert "yolo26x-pose.pt" in hint
    assert "--weights models/yolo26m-pose.pt" in hint


def test_hint_follows_the_given_path_and_fallback() -> None:
    hint = weights_download_hint("models/custom.pt", "自备权重")
    assert f"{ASSETS_URL}/custom.pt" in hint
    assert hint.endswith("自备权重")
