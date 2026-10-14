# SPDX-License-Identifier: AGPL-3.0-or-later
"""`--feature` 解析与写入顺序的单元测试 —— 纯函数，不需要相机。"""

from __future__ import annotations

import pytest

from hik_person.mvs import feature_setter_order, parse_feature_spec


def test_parse_simple() -> None:
    assert parse_feature_spec("ADCBitDepth=Bits_8") == ("ADCBitDepth", "Bits_8")


def test_parse_strips_spaces() -> None:
    assert parse_feature_spec("  Gamma = 1.2 ") == ("Gamma", "1.2")


def test_parse_keeps_equals_in_value() -> None:
    """只按第一个 = 切分，值里还能有 =。"""
    assert parse_feature_spec("A=b=c") == ("A", "b=c")


@pytest.mark.parametrize("spec", ["ADCBitDepth", "=Bits_8", "ADCBitDepth=", "", "   "])
def test_parse_rejects_bad_specs(spec: str) -> None:
    with pytest.raises(ValueError):
        parse_feature_spec(spec)


def test_bool_value_tries_bool_first() -> None:
    assert feature_setter_order("true")[0] == "bool"
    assert feature_setter_order("FALSE")[0] == "bool"


def test_integer_value_tries_int_first() -> None:
    assert feature_setter_order("3")[0] == "int"
    assert "enum_int" in feature_setter_order("3")  # 数值枚举也要能写


def test_float_value_tries_float_first() -> None:
    assert feature_setter_order("1.25")[0] == "float"


def test_symbolic_value_only_tries_enum() -> None:
    """像 Bits_8 这种只能按符号名写枚举，不该去试数值 setter（会白报错）。"""
    assert feature_setter_order("Bits_8") == ("enum",)
