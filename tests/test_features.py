# SPDX-License-Identifier: AGPL-3.0-or-later
"""`--feature` 解析、写入顺序与失败诊断的单元测试 —— 不需要真相机。"""

from __future__ import annotations

import pytest

from hik_person.mvs import (
    Camera,
    MvsError,
    alias_symbolic,
    feature_setter_order,
    parse_feature_spec,
    symbolic_candidates,
)

NOT_FOUND = 0x80000109  # MV_E_GC_NODE_NOT_FOUND
BAD_PARAM = 0x80000004  # MV_E_PARAMETER：节点在，但符号名不在枚举表里


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


# ------------------------------------------------------- 枚举符号名的提示（纯函数）
def test_candidates_matches_after_normalizing() -> None:
    assert symbolic_candidates("Gamma", "bits8", ["Bits_8"]) == ["Bits_8"]


def test_candidates_ignores_node_name_prefix() -> None:
    """别的机型写 Bits_8，本机叫 ADCBitDepth_8 —— 去掉节点名后对得上。"""
    entries = ["ADCBitDepth_8", "ADCBitDepth_12"]
    assert symbolic_candidates("ADCBitDepth", "Bits_8", entries) == ["ADCBitDepth_8"]


def test_candidates_also_works_when_written_the_other_way_around() -> None:
    """反向也认：CS 机的符号名写到 CB 机上（Bits_8 / Bits_12）。"""
    entries = ["Bits_8", "Bits_12"]
    assert symbolic_candidates("ADCBitDepth", "ADCBitDepth_8", entries) == ["Bits_8"]


def test_candidates_empty_when_nothing_matches() -> None:
    """猜不出来就什么都别说，不要把不相干的符号硬推给用户。"""
    assert symbolic_candidates("ADCBitDepth", "Bits_10", ["ADCBitDepth_8"]) == []


# ------------------------------------------------- 自动改写（同一个设置的另一个名字）
def test_alias_rewrites_both_directions() -> None:
    """两台相机的符号名都能互相认：CS 写 Bits_8、CB 写 ADCBitDepth_8 都能跑。"""
    assert (
        alias_symbolic("ADCBitDepth", "Bits_8", ["ADCBitDepth_8", "ADCBitDepth_12"])
        == "ADCBitDepth_8"
    )
    assert alias_symbolic("ADCBitDepth", "ADCBitDepth_8", ["Bits_8", "Bits_12"]) == "Bits_8"


def test_alias_never_changes_the_number() -> None:
    """安全性关键：只有 12bit 的节点绝不能因为名字像就被写成别的档位（2 != 12）。"""
    assert alias_symbolic("ADCBitDepth", "Bits_12", ["ADCBitDepth_2"]) is None
    assert alias_symbolic("ADCBitDepth", "Bits_10", ["ADCBitDepth_8"]) is None


def test_alias_refuses_when_several_symbols_match() -> None:
    """两个符号都像的时候不猜，交给报错去说。"""
    assert alias_symbolic("X", "Bits_8", ["Left_Bits_8", "Right_Bits_8"]) is None


def test_alias_accepts_a_pure_rename_without_digits() -> None:
    """只差节点名前缀的重命名（名字里没数字）也是同一个设置。"""
    assert alias_symbolic("ADCBitDepth", "Auto", ["ADCBitDepth_Auto"]) == "ADCBitDepth_Auto"


# --------------------------------------------------------- 失败诊断（假相机）
class _Struct:
    """占位符，只需要能被 sdk 那样调用。"""


class _EnumEntry:
    def __init__(self) -> None:
        self.nValue = 0
        self.chSymbolic = b""


class _EnumValue:
    def __init__(self) -> None:
        self.nCurValue = 0
        self.nSupportedNum = 0
        self.nSupportValue: list[int] = []


class _FakeSdk:
    MVCC_ENUMVALUE = _EnumValue
    MVCC_INTVALUE = _Struct
    MVCC_FLOATVALUE = _Struct
    MVCC_ENUMENTRY = _EnumEntry


class _FakeCam:
    """只实现 set_feature / 类型探测会碰到的那几个方法。

    复现真实相机的关键行为：只有对应类型的 getter 返回 0；枚举节点写一个不在表里
    的符号名给 0x80000004（节点不存在、或类型不对给 0x80000109）。
    """

    def __init__(self, kind: str | None, symbols: tuple[str, ...] = ()) -> None:
        self.kind = kind
        self.symbols = list(symbols)
        self.values = [0, 3][: len(symbols)]

    def _get(self, kind: str, key, out):
        if self.kind != kind:
            return NOT_FOUND
        if kind == "enum":
            out.nCurValue = self.values[0] if self.values else 0
            out.nSupportedNum = len(self.symbols)
            out.nSupportValue = list(self.values)
        return 0

    def MV_CC_GetEnumValue(self, key, out):
        return self._get("enum", key, out)

    def MV_CC_GetIntValue(self, key, out):
        return self._get("int", key, out)

    def MV_CC_GetFloatValue(self, key, out):
        return self._get("float", key, out)

    def MV_CC_GetBoolValue(self, key, out):
        return self._get("bool", key, out)

    def MV_CC_GetEnumEntrySymbolic(self, key, entry):
        entry.chSymbolic = self.symbols[self.values.index(entry.nValue)].encode()
        return 0

    def MV_CC_SetEnumValueByString(self, key, value):
        if self.kind != "enum":
            return NOT_FOUND
        return 0 if value in self.symbols else BAD_PARAM

    def MV_CC_SetIntValue(self, key, value):
        return 0 if self.kind == "int" else NOT_FOUND

    def MV_CC_SetFloatValue(self, key, value):
        return 0 if self.kind == "float" else NOT_FOUND

    def MV_CC_SetBoolValue(self, key, value):
        return 0 if self.kind == "bool" else NOT_FOUND


def make_camera(kind: str | None, symbols: tuple[str, ...] = ()) -> Camera:
    camera = Camera(_FakeSdk())
    camera.cam = _FakeCam(kind, symbols)
    return camera


def test_enum_written_successfully_returns_kind() -> None:
    camera = make_camera("enum", ("ADCBitDepth_8", "ADCBitDepth_12"))
    written = camera.set_feature("ADCBitDepth", "ADCBitDepth_8")
    assert written.kind == "enum"
    assert written.value == "ADCBitDepth_8"
    assert not written.aliased
    assert written.summary() == "ADCBitDepth_8 (以 enum 写入)"


def test_enum_symbol_from_other_model_is_rewritten_automatically() -> None:
    """换成另一台相机的符号名：自动改写成这台相机认的名字，而不是报错让人去改 config。"""
    camera = make_camera("enum", ("ADCBitDepth_8", "ADCBitDepth_12"))
    written = camera.set_feature("ADCBitDepth", "Bits_8")
    assert written.kind == "enum"
    assert written.value == "ADCBitDepth_8"
    assert written.requested == "Bits_8"
    assert written.aliased
    assert "已自动换成同一个设置的另一个名字" in written.summary()


def test_apply_features_summary_says_what_was_really_written() -> None:
    camera = make_camera("enum", ("Bits_8", "Bits_12"))
    assert camera.apply_features(["ADCBitDepth=ADCBitDepth_8"]) == [
        "ADCBitDepth=Bits_8 (以 enum 写入；这台相机不叫 ADCBitDepth_8，"
        "已自动换成同一个设置的另一个名字)"
    ]


def test_enum_without_close_match_lists_symbols_but_does_not_guess() -> None:
    camera = make_camera("enum", ("ADCBitDepth_8", "ADCBitDepth_12"))
    with pytest.raises(MvsError) as err:
        camera.set_feature("ADCBitDepth", "Bits_10")
    text = str(err.value)
    assert "ADCBitDepth_8 / ADCBitDepth_12" in text
    assert "想写的可能是" not in text


def test_non_enum_node_reports_its_type() -> None:
    """值写了符号名，但节点其实是 Int：要说清类型，别只说“拼写错了”。"""
    camera = make_camera("int")
    with pytest.raises(MvsError) as err:
        camera.set_feature("Brightness", "Bits_8")
    text = str(err.value)
    assert "Brightness 是整型（Int）特性" in text
    assert "不能按枚举符号写" in text


def test_unknown_node_reports_not_found() -> None:
    camera = make_camera(None)
    with pytest.raises(MvsError) as err:
        camera.set_feature("NoSuchNode", "Bits_8")
    assert "没有读到 NoSuchNode 这个特性" in str(err.value)


def test_ambiguous_symbol_is_reported_not_rewritten() -> None:
    camera = make_camera("enum", ("Left_Bits_8", "Right_Bits_8"))
    with pytest.raises(MvsError) as err:
        camera.set_feature("X", "Bits_8")
    text = str(err.value)
    assert "Left_Bits_8 / Right_Bits_8" in text
    assert "不能确定，没有自动改写" in text
