from pytest import raises

from slide.utils import parse_join_type


def test_parse_join_types():
    assert "cross" == parse_join_type("CROss")
    assert "inner" == parse_join_type("join")
    assert "inner" == parse_join_type("Inner")
    assert "left_outer" == parse_join_type("left")
    assert "left_outer" == parse_join_type("left  outer")
    assert "right_outer" == parse_join_type("right")
    assert "right_outer" == parse_join_type("right_ outer")
    assert "full_outer" == parse_join_type("full")
    assert "full_outer" == parse_join_type(" outer ")
    assert "full_outer" == parse_join_type("full_outer")
    assert "left_anti" == parse_join_type("anti")
    assert "left_anti" == parse_join_type("left anti")
    assert "left_semi" == parse_join_type("semi")
    assert "left_semi" == parse_join_type("left semi")
    raises(
        NotImplementedError,
        lambda: parse_join_type("right semi"),
    )
