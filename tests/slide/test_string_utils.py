from slide._string_utils import LikeExpr, LikeExprShortcut
import re


def test_like_expr():
    def test(expr, shortcut, *parts):
        le = LikeExpr(expr)
        assert list(parts) == [x[1] for x in le.tokens]
        assert shortcut == le.shortcut

    def test_re(expr, re_expr, matches, no_matches):
        le = LikeExpr(expr)
        if not isinstance(re_expr, str):
            assert le.re() in re_expr
        else:
            assert re_expr == le.re()
        tc = re.compile(le.re())
        for m in matches:
            assert tc.match(m) is not None
        for m in no_matches:
            assert tc.match(m) is None

    test("", LikeExprShortcut.EMPTY)
    test(" ", LikeExprShortcut.SIMPLE, " ")
    test("  ", LikeExprShortcut.SIMPLE, "  ")
    test("a", LikeExprShortcut.SIMPLE, "a")
    test("abc", LikeExprShortcut.SIMPLE, "abc")
    test("ab\\%\\_\\\\c", LikeExprShortcut.SIMPLE, "ab%_\\c")

    test("%", LikeExprShortcut.ANY, "%")
    test("%%%%", LikeExprShortcut.ANY, "%")

    test("Ab%", LikeExprShortcut.START, "Ab", "%")
    test("ab\\%%", LikeExprShortcut.START, "ab%", "%")
    test("ab\\%c%", LikeExprShortcut.START, "ab%c", "%")
    test("\\%ab\\%c%", LikeExprShortcut.START, "%ab%c", "%")
    test("\\b\\%c%", LikeExprShortcut.START, "b%c", "%")

    test("%%b\\%c", LikeExprShortcut.END, "%", "b%c")
    test("%%b\\", LikeExprShortcut.END, "%", "b\\")

    test("x%%b\\%c", LikeExprShortcut.START_END, "x", "%", "b%c")
    test("\\%%%b\\%c", LikeExprShortcut.START_END, "%", "%", "b%c")

    test("%%b\\%\\_c%%", LikeExprShortcut.CONTAIN, "%", "b%_c", "%")

    test("%_", LikeExprShortcut.NOT_EMPTY, "%", "_")
    test("%%_", LikeExprShortcut.NOT_EMPTY, "%", "_")
    test("_%", LikeExprShortcut.NOT_EMPTY, "_", "%")
    test("_%%", LikeExprShortcut.NOT_EMPTY, "_", "%")
    test("%%%%_%%", LikeExprShortcut.NOT_EMPTY, "%", "_", "%")

    test("__", LikeExprShortcut.NA, "_", "_")
    test("_a_b%%cd%", LikeExprShortcut.NA, "_", "a", "_", "b", "%", "cd", "%")

    test_re("", "^$", [""], [" ", "a"])
    test_re("__", "^..$", ["ab", "  "], ["abc", "a"])
    test_re(
        "a%bc\\%.?",
        [r"^a.*bc%\.\?$", r"^a.*bc\%\.\?$"],
        ["abc%.?", "afffbc%.?"],
        ["abc"],
    )
