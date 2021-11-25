from enum import Enum
from typing import Dict, Iterable, Tuple, List
import re


class LikeExprShortcut(Enum):
    SIMPLE = 0
    ANY = 1
    START = 2
    END = 3
    START_END = 4
    CONTAIN = 5
    NOT_EMPTY = 6
    EMPTY = 7
    NA = 20


_SHORTCUT_MAP: Dict[str, LikeExprShortcut] = {
    "": LikeExprShortcut.EMPTY,
    "s": LikeExprShortcut.SIMPLE,
    "%": LikeExprShortcut.ANY,
    "s%": LikeExprShortcut.START,
    "%s": LikeExprShortcut.END,
    "s%s": LikeExprShortcut.START_END,
    "%s%": LikeExprShortcut.CONTAIN,
    "%_": LikeExprShortcut.NOT_EMPTY,
    "_%": LikeExprShortcut.NOT_EMPTY,
    "%_%": LikeExprShortcut.NOT_EMPTY,
}


class LikeExpr:
    def __init__(self, expr: str):
        self._tokens = list(self._get_clean_tokens(expr))
        self._pattern = "".join(x[0] for x in self._tokens)
        self._shortcut = _SHORTCUT_MAP.get(self._pattern, LikeExprShortcut.NA)

    @property
    def tokens(self) -> List[Tuple[str, str]]:
        return self._tokens

    @property
    def shortcut(self) -> LikeExprShortcut:
        return self._shortcut

    def re(self) -> str:
        def f_():
            yield "^"
            for t in self._tokens:
                if t[0] == "%":
                    yield ".*"
                elif t[0] == "_":
                    yield "."
                else:
                    yield re.escape(t[1])
            yield "$"

        return "".join(f_())

    def _get_clean_tokens(self, expr: str) -> Iterable[Tuple[str, str]]:
        last = None
        for x in self._get_tokens(expr):
            if x[0] == "%":
                if last == "%":
                    continue
            yield x
            last = x[0]

    def _get_tokens(self, expr: str) -> Iterable[Tuple[str, str]]:
        i = 0
        while i < len(expr):
            c = expr[i]
            if c == "%" or c == "_":
                yield (c, c)
                i += 1
            else:
                end = i
                res = ""
                while end < len(expr):
                    c = expr[end]
                    if c == "%" or c == "_":
                        break
                    if c == "\\":
                        res += expr[i:end]
                        if end + 1 < len(expr):
                            res += expr[end + 1]
                            i = end + 2
                            end = i
                            continue
                        else:
                            res += c
                            end = i = len(expr)
                            break
                    end += 1
                if end > i:
                    res += expr[i:end]
                i = end
                yield ("s", res)
