# SPDX-License-Identifier: AGPL-3.0-or-later
"""MVS SDK 需要的运行时环境变量 —— 纯函数，不需要相机、不加载 SDK。"""

from __future__ import annotations

from hik_person.mvs import MVS_LIB_DIR, MVS_ROOT, _apply_env, library_path_configured


def test_apply_env_matches_driver_environment_hook() -> None:
    """/opt/MVS 下 SDK 需要的几个变量。"""
    env = _apply_env({})
    assert env["MVCAM_SDK_PATH"] == MVS_ROOT
    assert env["MVCAM_COMMON_RUNENV"] == f"{MVS_ROOT}/lib"
    assert env["MVCAM_SOFTWARE_LIBENV"] == f"{MVS_ROOT}/lib"
    assert env["MVCAM_GENICAM_CLPROTOCOL"] == f"{MVS_ROOT}/lib/CLProtocol"
    assert env["ALLUSERSPROFILE"] == f"{MVS_ROOT}/MVFG"


def test_apply_env_prepends_library_dir() -> None:
    """SDK 库目录要 prepend，且不丢掉已有条目。"""
    env = _apply_env({"LD_LIBRARY_PATH": "/somewhere/else"})
    assert env["LD_LIBRARY_PATH"] == f"{MVS_LIB_DIR}:/somewhere/else"


def test_apply_env_is_idempotent() -> None:
    once = _apply_env({})
    twice = _apply_env(dict(once))
    assert once["LD_LIBRARY_PATH"] == twice["LD_LIBRARY_PATH"]
    assert twice["LD_LIBRARY_PATH"].count(MVS_LIB_DIR) == 1


def test_library_path_configured(monkeypatch) -> None:
    monkeypatch.setenv("LD_LIBRARY_PATH", f"/usr/lib:{MVS_LIB_DIR}")
    assert library_path_configured()

    monkeypatch.setenv("LD_LIBRARY_PATH", "/usr/lib")
    assert not library_path_configured()

    monkeypatch.delenv("LD_LIBRARY_PATH", raising=False)
    assert not library_path_configured()
