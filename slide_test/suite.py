import json
import pickle
from datetime import date, datetime
from typing import Any
from unittest import TestCase

import numpy as np
import pandas as pd
import pyarrow as pa
from pytest import raises
from slide.exceptions import SlideCastError, SlideInvalidOperation
from slide.utils import SlideUtils
from slide_test.utils import assert_duck_eq, assert_pdf_eq, make_rand_df
from triad import Schema
from triad.utils.pyarrow import expression_to_schema


class SlideTestSuite(object):
    """Pandas-like utils test suite.
    Any new :class:`~slide.utils.SlideUtils` should pass this test suite.
    """

    class Tests(TestCase):
        @classmethod
        def setUpClass(cls):
            # register_default_sql_engine(lambda engine: engine.sql_engine)
            cls._utils = cls.make_utils(cls)
            pass

        def make_utils(self) -> SlideUtils:
            raise NotImplementedError

        @property
        def utils(self) -> SlideUtils:
            return self._utils  # type: ignore

        @classmethod
        def tearDownClass(cls):
            # cls._engine.stop()
            pass

        def to_pd(self, data: Any) -> pd.DataFrame:
            raise NotImplementedError

        def to_df(
            self,
            data: Any,
            columns: Any = None,
            enforce_type: bool = True,
            null_safe: bool = False,
        ):
            raise NotImplementedError

        def test_is_series(self):
            df = self.to_df([["a", 1]], "a:str,b:long")
            assert self.utils.is_series(df["a"])
            assert not self.utils.is_series(None)
            assert not self.utils.is_series(1)
            assert not self.utils.is_series("abc")

        def test_to_series(self):
            s1 = self.utils.to_series(pd.Series([0, 1], name="x"))
            s2 = self.utils.to_series(pd.Series([2, 3], name="x"), "y")
            s3 = self.utils.to_series([4, 5], "z")
            assert self.utils.is_series(s1)
            assert self.utils.is_series(s2)
            assert self.utils.is_series(s3)

            df = self.utils.cols_to_df([s1, s2, s3])
            assert_pdf_eq(
                self.to_pd(df), pd.DataFrame(dict(x=[0, 1], y=[2, 3], z=[4, 5]))
            )

        def test_to_constant_series(self):
            s = self.utils.to_series(pd.Series([0, 1], name="x"))
            s1 = self.utils.to_constant_series("a", s, name="y")
            s2 = self.utils.to_constant_series(None, s, name="z", dtype="float64")
            df = self.utils.cols_to_df([s, s1, s2])
            assert_pdf_eq(
                self.to_pd(df),
                pd.DataFrame(dict(x=[0, 1], y=["a", "a"], z=[None, None])),
            )

        def test_get_col_pa_type(self):
            df = self.to_df(
                [["a", 1, 1.1, True, datetime.now()]],
                "a:str,b:long,c:double,d:bool,e:datetime",
            )
            assert pa.types.is_string(self.utils.get_col_pa_type(df["a"]))
            assert pa.types.is_string(self.utils.get_col_pa_type("a"))
            assert pa.types.is_int64(self.utils.get_col_pa_type(df["b"]))
            assert pa.types.is_integer(self.utils.get_col_pa_type(123))
            assert pa.types.is_float64(self.utils.get_col_pa_type(df["c"]))
            assert pa.types.is_floating(self.utils.get_col_pa_type(1.1))
            assert pa.types.is_boolean(self.utils.get_col_pa_type(df["d"]))
            assert pa.types.is_boolean(self.utils.get_col_pa_type(False))
            assert pa.types.is_timestamp(self.utils.get_col_pa_type(df["e"]))
            assert pa.types.is_timestamp(self.utils.get_col_pa_type(datetime.now()))

        def test_unary_arithmetic_op(self):
            pdf = pd.DataFrame([[2.0], [0.0], [None], [-3.0]], columns=["a"])
            df = self.to_df(pdf)
            df["a"] = self.utils.unary_arithmetic_op(df["a"], "+")
            assert_pdf_eq(self.to_pd(df), pdf)
            df["a"] = self.utils.unary_arithmetic_op(df["a"], "-")
            pdf = pd.DataFrame([[-2.0], [0.0], [None], [3.0]], columns=["a"])
            assert_pdf_eq(self.to_pd(df), pdf)
            df["a"] = self.utils.unary_arithmetic_op(-10.1, "-")
            pdf = pd.DataFrame([[10.1], [10.1], [10.1], [10.1]], columns=["a"])
            assert_pdf_eq(self.to_pd(df), pdf)
            raises(
                NotImplementedError,
                lambda: self.utils.unary_arithmetic_op(df["a"], "]"),
            )

        def test_binary_arithmetic_op(self):
            def test_(pdf: pd.DataFrame, op: str):
                df = self.to_df(pdf)
                df["d"] = self.utils.binary_arithmetic_op(pdf.a, pdf.b, op)
                df["e"] = self.utils.binary_arithmetic_op(pdf.a, 1.0, op)
                df["f"] = self.utils.binary_arithmetic_op(1.0, pdf.b, op)
                df["g"] = self.utils.binary_arithmetic_op(1.0, 2.0, op)
                df["h"] = self.utils.binary_arithmetic_op(1.0, pdf.c, op)
                df["i"] = self.utils.binary_arithmetic_op(pdf.a, pdf.c, op)

                assert_duck_eq(
                    self.to_pd(df[list("defghi")]),
                    f"""
                    SELECT
                        a{op}b AS d, a{op}1.0 AS e, 1.0{op}b AS f,
                        1.0{op}2.0 AS g, 1.0{op}c AS h, a{op}c AS i
                    FROM pdf
                    """,
                    pdf=pdf,
                    check_order=False,
                )

            pdf = pd.DataFrame(
                dict(
                    a=[1.0, 2.0, 3.0, 4.0],
                    b=[2.0, 2.0, 0.1, 2.0],
                    c=[1.0, None, 1.0, float("nan")],
                )
            )
            test_(pdf, "+")
            test_(pdf, "-")
            test_(pdf, "*")
            test_(pdf, "/")

            # Integer division and dividing by 0 do not have consistent behaviors
            # on different SQL engines. So we can't unify.
            # SELECT 1.0/0.0 AS x, 1/2 AS y

        def test_comparison_op_num(self):
            def test_(pdf: pd.DataFrame, op: str):
                df = self.to_df(pdf)
                df["d"] = self.utils.comparison_op(pdf.a, pdf.b, op)
                df["e"] = self.utils.comparison_op(pdf.a, 2.0, op)
                df["f"] = self.utils.comparison_op(2.0, pdf.b, op)
                df["g"] = self.utils.comparison_op(2.0, 3.0, op)
                df["h"] = self.utils.comparison_op(2.0, pdf.c, op)
                df["i"] = self.utils.comparison_op(pdf.a, pdf.c, op)
                df["j"] = self.utils.comparison_op(pdf.c, pdf.c, op)

                assert_duck_eq(
                    self.to_pd(df[list("defghij")]),
                    f"""
                    SELECT
                        a{op}b AS d, a{op}2.0 AS e, 2.0{op}b AS f,
                        2.0{op}3.0 AS g, 2.0{op}c AS h, a{op}c AS i,
                        c{op}c AS j
                    FROM pdf
                    """,
                    pdf=pdf,
                    check_order=False,
                )

                assert self.utils.comparison_op(None, None, op) is None

            pdf = pd.DataFrame(
                dict(
                    a=[1.0, 2.0, 3.0, 4.0],
                    b=[2.0, 2.0, 0.1, 2.0],
                    c=[2.0, None, 2.0, float("nan")],
                )
            )
            test_(pdf, "<")
            test_(pdf, "<=")
            test_(pdf, "==")
            test_(pdf, "!=")
            test_(pdf, ">")
            test_(pdf, ">=")

        def test_comparison_op_str(self):
            def test_(pdf: pd.DataFrame, op: str):
                df = self.to_df(pdf)
                df["d"] = self.utils.comparison_op(pdf.a, pdf.b, op)
                df["e"] = self.utils.comparison_op(pdf.a, "y", op)
                df["f"] = self.utils.comparison_op("y", pdf.b, op)
                df["g"] = self.utils.comparison_op("y", "z", op)
                df["h"] = self.utils.comparison_op("y", pdf.c, op)
                df["i"] = self.utils.comparison_op(pdf.a, pdf.c, op)
                df["j"] = self.utils.comparison_op(pdf.c, pdf.c, op)

                assert_duck_eq(
                    self.to_pd(df[list("defghij")]),
                    f"""
                    SELECT
                        a{op}b AS d, a{op}'y' AS e, 'y'{op}b AS f,
                        'y'{op}'z' AS g, 'y'{op}c AS h, a{op}c AS i,
                        c{op}c AS j
                    FROM pdf
                    """,
                    pdf=pdf,
                    check_order=False,
                )

            pdf = pd.DataFrame(
                dict(
                    a=["xx", None, "x"],
                    b=[None, "t", "tt"],
                    c=["zz", None, "z"],
                )
            )
            test_(pdf, "<")
            test_(pdf, "<=")
            test_(pdf, "==")
            test_(pdf, "!=")
            test_(pdf, ">")
            test_(pdf, ">=")

        def test_comparison_op_time(self):
            t = datetime(2019, 1, 1)
            x = datetime(2020, 1, 1)
            y = datetime(2020, 1, 2)
            z = datetime(2020, 1, 3)

            def test_(pdf: pd.DataFrame, op: str):
                df = self.to_df(pdf)
                df["d"] = self.utils.comparison_op(pdf.a, pdf.b, op)
                df["e"] = self.utils.comparison_op(pdf.a, y, op)
                df["f"] = self.utils.comparison_op(y, pdf.b, op)
                df["g"] = self.utils.comparison_op(y, z, op)
                df["h"] = self.utils.comparison_op(y, pdf.c, op)
                df["i"] = self.utils.comparison_op(pdf.a, pdf.c, op)
                df["j"] = self.utils.comparison_op(pdf.c, pdf.c, op)

                assert_duck_eq(
                    self.to_pd(df[list("defghij")]),
                    f"""
                    SELECT
                        a{op}b AS d, a{op}'{y}' AS e, '{y}'{op}b AS f,
                        '{y}'{op}'{z}' AS g, '{y}'{op}c AS h, a{op}c AS i,
                        c{op}c AS j
                    FROM pdf
                    """,
                    pdf=pdf,
                    check_order=False,
                )

            pdf = pd.DataFrame(
                dict(
                    a=[x, None, x],
                    b=[None, t, t],
                    c=[z, z, None],
                )
            )
            test_(pdf, "<")
            test_(pdf, "<=")
            test_(pdf, "==")
            test_(pdf, "!=")
            test_(pdf, ">")
            test_(pdf, ">=")

        def test_binary_logical_op(self):
            def test_(pdf: pd.DataFrame, op: str):
                df = self.to_df(pdf)
                df["d"] = self.utils.binary_logical_op(pdf.a, pdf.b, op)
                df["e"] = self.utils.binary_logical_op(pdf.a, True, op)
                df["f"] = self.utils.binary_logical_op(True, pdf.b, op)
                df["g"] = self.utils.binary_logical_op(pdf.a, False, op)
                df["h"] = self.utils.binary_logical_op(False, pdf.b, op)
                df["i"] = self.utils.binary_logical_op(True, False, op)
                df["j"] = self.utils.binary_logical_op(True, None, op)
                df["k"] = self.utils.binary_logical_op(False, None, op)
                df["l"] = self.utils.binary_logical_op(None, None, op)

                assert_duck_eq(
                    self.to_pd(df[list("defghijkl")]),
                    f"""
                    SELECT
                        a {op} b AS d, a {op} TRUE AS e, TRUE {op} b AS f,
                        a {op} FALSE AS g, FALSE {op} b AS h, TRUE {op} FALSE AS i,
                        TRUE {op} NULL AS j, FALSE {op} NULL AS k, NULL {op} NULL AS l
                    FROM pdf
                    """,
                    pdf=pdf,
                    check_order=False,
                )

            pdf = pd.DataFrame(
                dict(
                    a=[True, False, True, False, True, False, None],
                    b=[False, True, True, False, None, None, None],
                )
            )
            test_(pdf, "and")
            test_(pdf, "or")

        def test_logical_not(self):
            def test_(pdf: pd.DataFrame):
                df = self.to_df(pdf)
                df["c"] = self.utils.logical_not(pdf.a)
                df["e"] = self.utils.logical_not(True)
                df["f"] = self.utils.logical_not(False)
                df["g"] = self.utils.logical_not(None)

                assert_duck_eq(
                    self.to_pd(df[list("cefg")]),
                    """
                    SELECT
                        NOT a AS c, NOT TRUE AS e,
                        NOT FALSE AS f, NOT NULL AS g
                    FROM pdf
                    """,
                    pdf=pdf,
                    check_order=False,
                )

            pdf = pd.DataFrame(dict(a=[True, False, None]))
            test_(pdf)

        def test_filter_df(self):
            def test_(pdf: pd.DataFrame):
                df = self.to_df(pdf)
                assert_duck_eq(
                    self.to_pd(self.utils.filter_df(df, df["a"])),
                    """
                    SELECT * FROM pdf WHERE a
                    """,
                    pdf=pdf,
                    check_order=False,
                )

            test_(pd.DataFrame(dict(a=[True, False], b=[1.0, 2.0])))
            test_(pd.DataFrame(dict(a=[False, False], b=[1.0, 2.0])))
            test_(pd.DataFrame(dict(a=[1.0, 0.0, None], b=[1.0, 2.0, 3.0])))
            test_(pd.DataFrame(dict(a=[float("nan"), 0.0, None], b=[1.0, 2.0, 3.0])))

            pdf = pd.DataFrame([[1], [2]], columns=["a"])
            df = self.to_df(pdf)
            assert_duck_eq(
                self.to_pd(self.utils.filter_df(df, True)),
                """
                SELECT * FROM pdf WHERE TRUE
                """,
                pdf=pdf,
                check_order=False,
            )
            assert_duck_eq(
                self.to_pd(self.utils.filter_df(df, False)),
                """
                SELECT * FROM pdf WHERE FALSE
                """,
                pdf=pdf,
                check_order=False,
            )

        def test_is_value(self):
            assert self.utils.is_value(None, None, True)
            assert not self.utils.is_value(None, None, False)
            assert not self.utils.is_value(None, True, True)
            assert self.utils.is_value(None, True, False)
            assert not self.utils.is_value(None, False, True)
            assert self.utils.is_value(None, False, False)

            assert self.utils.is_value(float("nan"), None, True)
            assert not self.utils.is_value(float("nan"), None, False)

            assert self.utils.is_value(pd.NaT, None, True)
            assert not self.utils.is_value(pd.NaT, None, False)

            assert not self.utils.is_value("abc", None, True)
            assert self.utils.is_value("abc", None, False)

            assert not self.utils.is_value(True, None, True)
            assert self.utils.is_value(True, None, False)
            assert self.utils.is_value(True, True, True)
            assert not self.utils.is_value(True, True, False)
            assert not self.utils.is_value(True, False, True)
            assert self.utils.is_value(True, False, False)

            assert not self.utils.is_value(-1.1, None, True)
            assert self.utils.is_value(-1.1, None, False)
            assert self.utils.is_value(-1.1, True, True)
            assert not self.utils.is_value(-1.1, True, False)
            assert not self.utils.is_value(-1.1, False, True)
            assert self.utils.is_value(-1.1, False, False)

            assert not self.utils.is_value(False, None, True)
            assert self.utils.is_value(False, None, False)
            assert not self.utils.is_value(False, True, True)
            assert self.utils.is_value(False, True, False)
            assert self.utils.is_value(False, False, True)
            assert not self.utils.is_value(False, False, False)

            assert not self.utils.is_value(0, None, True)
            assert self.utils.is_value(0, None, False)
            assert not self.utils.is_value(0, True, True)
            assert self.utils.is_value(0, True, False)
            assert self.utils.is_value(0, False, True)
            assert not self.utils.is_value(0, False, False)

            with raises(NotImplementedError):
                self.utils.is_value(0, "x", False)

            pdf = pd.DataFrame(dict(a=[True, False, None]))

            df = self.to_df(pdf)
            df["h"] = self.utils.is_value(df["a"], None, True)
            df["i"] = self.utils.is_value(df["a"], None, False)
            df["j"] = self.utils.is_value(df["a"], True, True)
            df["k"] = self.utils.is_value(df["a"], True, False)
            df["l"] = self.utils.is_value(df["a"], False, True)
            df["m"] = self.utils.is_value(df["a"], False, False)

            assert_pdf_eq(
                self.to_pd(df[list("hijklm")]),
                pd.DataFrame(
                    dict(
                        h=[False, False, True],
                        i=[True, True, False],
                        j=[True, False, False],
                        k=[False, True, True],
                        l=[False, True, False],
                        m=[True, False, True],
                    ),
                ),
                check_order=False,
            )

        def test_is_in(self):
            assert self.utils.is_in(None, [None, 1], True) is None
            assert self.utils.is_in(None, [None, 1], False) is None
            assert self.utils.is_in(None, ["a", "b"], True) is None
            assert self.utils.is_in(None, ["a", "b"], False) is None

            assert self.utils.is_in(True, [False, True], True)
            assert not self.utils.is_in(True, [False, True], False)
            assert self.utils.is_in(False, [None, False], True)
            assert not self.utils.is_in(False, [None, False], False)

            assert self.utils.is_in(True, [None, False], True) is None
            assert self.utils.is_in(True, [None, False], False) is None

            assert self.utils.is_in(1, [2, 1], True)
            assert not self.utils.is_in(1, [2, 1], False)
            assert self.utils.is_in(1, [None, 1], True)
            assert not self.utils.is_in(1, [None, 1], False)

            assert self.utils.is_in(1, [None, 2], True) is None
            assert self.utils.is_in(1, [None, 2], False) is None

            assert self.utils.is_in(1.1, [2.2, 1.1], True)
            assert not self.utils.is_in(1.1, [2.2, 1.1], False)
            assert self.utils.is_in(1.1, [None, 1.1], True)
            assert not self.utils.is_in(1.1, [None, 1.1], False)

            assert self.utils.is_in(1.1, [None, 2.2], True) is None
            assert self.utils.is_in(1.1, [None, 2.2], False) is None

            assert self.utils.is_in("aa", ["bb", "aa"], True)
            assert not self.utils.is_in("aa", ["bb", "aa"], False)
            assert self.utils.is_in("aa", [None, "aa"], True)
            assert not self.utils.is_in("aa", [None, "aa"], False)

            assert self.utils.is_in("aa", [None, "bb"], True) is None
            assert self.utils.is_in("aa", [None, "b"], False) is None

            assert self.utils.is_in(
                date(2020, 1, 1), [date(2020, 1, 2), date(2020, 1, 1)], True
            )
            assert not self.utils.is_in(
                date(2020, 1, 1), [date(2020, 1, 2), date(2020, 1, 1)], False
            )
            assert self.utils.is_in(date(2020, 1, 1), [pd.NaT, date(2020, 1, 1)], True)
            assert not self.utils.is_in(
                date(2020, 1, 1), [None, date(2020, 1, 1)], False
            )

            assert (
                self.utils.is_in(date(2020, 1, 1), [pd.NaT, date(2020, 1, 2)], True)
                is None
            )
            assert (
                self.utils.is_in(date(2020, 1, 1), [None, date(2020, 1, 2)], False)
                is None
            )

        def test_is_in_sql(self):
            pdf = pd.DataFrame(
                dict(
                    a=[True, False, None],
                    b=[1, 2, None],
                    c=[1.1, 2.2, None],
                    d=["aa", "bb", None],
                    e=[date(2020, 1, 1), date(2020, 1, 2), None],
                )
            )

            df = self.to_df(pdf)
            df["h"] = self.utils.is_in(df["a"], [False, None], True)
            df["i"] = self.utils.is_in(df["a"], [False, None], False)
            df["j"] = self.utils.is_in(df["b"], [1, 3, None], True)
            df["k"] = self.utils.is_in(df["b"], [1, 3, None], False)
            df["l"] = self.utils.is_in(df["c"], [1.1, 3.3, None], True)
            df["m"] = self.utils.is_in(df["c"], [1.1, 3.3, None], False)
            df["n"] = self.utils.is_in(df["d"], ["aa", "cc", None], True)
            df["o"] = self.utils.is_in(df["d"], ["aa", "cc", None], False)
            df["p"] = self.utils.is_in(
                df["e"], [date(2020, 1, 1), date(2020, 1, 3), None], True
            )
            df["q"] = self.utils.is_in(
                df["e"], [date(2020, 1, 1), date(2020, 1, 3), None], False
            )

            assert_duck_eq(
                self.to_pd(df[list("jklmnopq")]),
                """
                SELECT
                    -- a IN (FALSE, NULL) AS h,
                    -- a NOT IN (FALSE, NULL) AS i,
                    b IN (3, 1, NULL) AS j,
                    b NOT IN (3, 1, NULL) AS k,
                    c IN (3.3, 1.1, NULL) AS l,
                    c NOT IN (3.3, 1.1, NULL) AS m,
                    d IN ('cc', 'aa', NULL) AS n,
                    d NOT IN ('cc', 'aa', NULL) AS o,
                    e IN ('2020-01-03', '2020-01-01', NULL) AS p,
                    e NOT IN ('2020-01-03', '2020-01-01', NULL) AS q
                FROM a
                """,
                a=pdf,
                check_order=False,
            )

            pdf = pd.DataFrame(
                dict(
                    a=[1.1, 2.2, None],
                    b=[1.1, None, None],
                    c=[None, 2.2, None],
                    d=[3.3, None, None],
                    e=[None, 4.4, None],
                )
            )

            df = self.to_df(pdf)
            df["h"] = self.utils.is_in(df["a"], [df["b"], df["c"]], True)
            df["i"] = self.utils.is_in(df["a"], [df["b"], df["c"]], False)
            df["j"] = self.utils.is_in(df["a"], [df["d"], df["e"]], True)
            df["k"] = self.utils.is_in(df["a"], [df["d"], df["e"]], False)
            df["l"] = self.utils.is_in(df["a"], [df["b"], df["d"], None], True)
            df["m"] = self.utils.is_in(df["a"], [df["b"], df["d"], None], False)

            assert_duck_eq(
                self.to_pd(df[list("hijklm")]),
                """
                SELECT
                    a IN (b, c) AS h,
                    a NOT IN (b, c) AS i,
                    a IN (d, e) AS j,
                    a NOT IN (d, e) AS k,
                    a IN (b, d, NULL) AS l,
                    a NOT IN (b, d, NULL) AS m
                FROM a
                """,
                a=pdf,
                check_order=False,
            )

        def test_is_between(self):
            # if col is null, then the result is null
            for a in [1, 2, None]:
                for b in [1, 2, None]:
                    for p in [True, False]:
                        assert self.utils.is_between(None, a, b, p) is None

            # one side is none and the result can't be determined, so null
            assert self.utils.is_between(2, None, 2, True) is None
            assert self.utils.is_between(2, None, 2, False) is None
            assert self.utils.is_between(3, 2, None, True) is None
            assert self.utils.is_between(3, 2, None, False) is None

            # one side is none but the result is still deterministic
            assert not self.utils.is_between(3, None, 2, True)
            assert self.utils.is_between(3, None, 2, False)
            assert not self.utils.is_between(1, 2, None, True)
            assert self.utils.is_between(1, 2, None, False)

            # if lower and upper are both nulls, the result is null
            assert self.utils.is_between(3, None, None, True) is None
            assert self.utils.is_between(3, None, None, False) is None

            # happy paths
            assert self.utils.is_between(1, 1, 2, True)
            assert not self.utils.is_between(2, 1, 2, False)
            assert not self.utils.is_between(0, 1, 2, True)
            assert self.utils.is_between(0, 1, 2, False)
            assert not self.utils.is_between(3, 1, 2, True)
            assert self.utils.is_between(3, 1, 2, False)

            assert self.utils.is_between("bb", "bb", "cc", True)
            assert not self.utils.is_between("cc", "bb", "cc", False)
            assert not self.utils.is_between("aa", "bb", "cc", True)
            assert self.utils.is_between("aa", "bb", "cc", False)

            assert self.utils.is_between(
                date(2020, 1, 2), date(2020, 1, 2), date(2020, 1, 3), True
            )
            assert not self.utils.is_between(
                date(2020, 1, 3), date(2020, 1, 2), date(2020, 1, 3), False
            )
            assert not self.utils.is_between(
                date(2020, 1, 1), date(2020, 1, 2), date(2020, 1, 3), True
            )
            assert self.utils.is_between(
                date(2020, 1, 1), date(2020, 1, 2), date(2020, 1, 3), False
            )

        def test_is_between_sql(self):
            pdf = make_rand_df(100, a=(float, 20), b=(float, 20), c=(float, 20))
            # pdf = make_rand_df(5, a=(float, 2), b=(float, 2), c=(float, 2))
            print(pdf)

            df = self.to_df(pdf)
            df["h"] = self.utils.is_between(df["a"], df["b"], df["c"], True)
            df["i"] = self.utils.is_between(df["a"], df["b"], df["c"], False)
            df["j"] = self.utils.is_between(None, df["b"], df["c"], True)
            df["k"] = self.utils.is_between(None, df["b"], df["c"], False)
            df["l"] = self.utils.is_between(df["a"], df["b"], None, True)
            df["m"] = self.utils.is_between(df["a"], df["b"], None, False)
            df["n"] = self.utils.is_between(df["a"], None, df["c"], True)
            df["o"] = self.utils.is_between(df["a"], None, df["c"], False)
            df["p"] = self.utils.is_between(df["a"], 0.5, df["c"], True)
            df["q"] = self.utils.is_between(df["a"], 0.5, df["c"], False)
            df["r"] = self.utils.is_between(df["a"], df["b"], 0.5, True)
            df["s"] = self.utils.is_between(df["a"], df["b"], 0.5, False)

            assert_duck_eq(
                self.to_pd(df[list("hijklmnopqrs")]),
                """
                SELECT
                    a BETWEEN b AND c AS h,
                    a NOT BETWEEN b AND c AS i,
                    NULL BETWEEN b AND c AS j,
                    NULL NOT BETWEEN b AND c AS k,
                    a BETWEEN b AND NULL AS l,
                    a NOT BETWEEN b AND NULL AS m,
                    a BETWEEN NULL AND c AS n,
                    a NOT BETWEEN NULL AND c AS o,
                    a BETWEEN 0.5 AND c AS p,
                    a NOT BETWEEN 0.5 AND c AS q,
                    a BETWEEN b AND 0.5 AS r,
                    a NOT BETWEEN b AND 0.5 AS s
                FROM a
                """,
                a=pdf,
                check_order=False,
            )

        def test_cast_coalesce_sql(self):
            pdf = make_rand_df(100, a=(float, 50), b=(float, 50), c=(float, 50))

            df = self.to_df(pdf)
            df["g"] = self.utils.coalesce([None])
            df["h"] = self.utils.coalesce([None, 10.1, None])
            df["i"] = self.utils.coalesce([df["a"], 10.1])
            df["j"] = self.utils.coalesce([10.1, df["a"]])
            df["k"] = self.utils.coalesce([df["a"], None])
            df["l"] = self.utils.coalesce([None, df["a"]])
            df["m"] = self.utils.coalesce([df["a"], df["b"], df["c"]])
            df["n"] = self.utils.coalesce([df["a"], df["b"], df["c"], 10.1])

            assert_duck_eq(
                self.to_pd(df[list("ghijklmn")]),
                """
                SELECT
                    COALESCE(NULL) AS g,
                    COALESCE(NULL, 10.1, NULL) AS h,
                    COALESCE(a, 10.1) AS i,
                    COALESCE(10.1, a) AS j,
                    COALESCE(a, NULL) AS k,
                    COALESCE(NULL, a) AS l,
                    COALESCE(a,b,c) AS m,
                    COALESCE(a,b,c,10.1) AS n
                FROM a
                """,
                a=pdf,
                check_order=False,
            )

            pdf = make_rand_df(100, a=(bool, 50), b=(bool, 50), c=(bool, 50))

            df = self.to_df(pdf)
            df["h"] = self.utils.coalesce([None, False, None])
            df["i"] = self.utils.coalesce([df["a"], False])
            df["j"] = self.utils.coalesce([False, df["a"]])
            df["k"] = self.utils.coalesce([df["a"], None])
            df["l"] = self.utils.coalesce([None, df["a"]])
            df["m"] = self.utils.coalesce([df["a"], df["b"], df["c"]])
            df["n"] = self.utils.coalesce([df["a"], df["b"], df["c"], False])

            assert_duck_eq(
                self.to_pd(df[list("hijklmn")]),
                """
                SELECT
                    COALESCE(NULL, FALSE) AS h,
                    COALESCE(a, FALSE) AS i,
                    COALESCE(FALSE, a) AS j,
                    COALESCE(a, NULL) AS k,
                    COALESCE(NULL, a) AS l,
                    COALESCE(a,b,c) AS m,
                    COALESCE(a,b,c,FALSE) AS n
                FROM (SELECT
                        CAST(a AS BOOLEAN) a,
                        CAST(b AS BOOLEAN) b,
                        CAST(c AS BOOLEAN) c FROM a)
                """,
                a=pdf,
                check_order=False,
            )

            pdf = make_rand_df(100, a=(int, 50), b=(int, 50), c=(int, 50))

            df = self.to_df(pdf)
            df["h"] = self.utils.coalesce([None, 10, None])
            df["i"] = self.utils.coalesce([df["a"], 10])
            df["j"] = self.utils.coalesce([10, df["a"]])
            df["k"] = self.utils.coalesce([df["a"], None])
            df["l"] = self.utils.coalesce([None, df["a"]])
            df["m"] = self.utils.coalesce([df["a"], df["b"], df["c"]])
            df["n"] = self.utils.coalesce([df["a"], df["b"], df["c"], 10])

            assert_duck_eq(
                self.to_pd(df[list("hijklmn")]),
                """
                SELECT
                    COALESCE(NULL, 10) AS h,
                    COALESCE(a, 10) AS i,
                    COALESCE(10, a) AS j,
                    COALESCE(a, NULL) AS k,
                    COALESCE(NULL, a) AS l,
                    COALESCE(a,b,c) AS m,
                    COALESCE(a,b,c,10) AS n
                FROM (SELECT
                        CAST(a AS INTEGER) a,
                        CAST(b AS INTEGER) b,
                        CAST(c AS INTEGER) c FROM a)
                """,
                a=pdf,
                check_order=False,
            )

            pdf = make_rand_df(100, a=(str, 50), b=(str, 50), c=(str, 50))

            df = self.to_df(pdf)
            df["h"] = self.utils.coalesce([None, "xx", None])
            df["i"] = self.utils.coalesce([df["a"], "xx"])
            df["j"] = self.utils.coalesce(["xx", df["a"]])
            df["k"] = self.utils.coalesce([df["a"], None])
            df["l"] = self.utils.coalesce([None, df["a"]])
            df["m"] = self.utils.coalesce([df["a"], df["b"], df["c"]])
            df["n"] = self.utils.coalesce([df["a"], df["b"], df["c"], "xx"])

            assert_duck_eq(
                self.to_pd(df[list("hijklmn")]),
                """
                SELECT
                    COALESCE(NULL, 'xx') AS h,
                    COALESCE(a, 'xx') AS i,
                    COALESCE('xx', a) AS j,
                    COALESCE(a, NULL) AS k,
                    COALESCE(NULL, a) AS l,
                    COALESCE(a,b,c) AS m,
                    COALESCE(a,b,c,'xx') AS n
                FROM a
                """,
                a=pdf,
                check_order=False,
            )

            pdf = make_rand_df(
                100, a=(datetime, 50), b=(datetime, 50), c=(datetime, 50)
            )

            ct = datetime(2020, 1, 1, 15)
            df = self.to_df(pdf)
            df["h"] = self.utils.coalesce([None, ct, None])
            df["i"] = self.utils.coalesce([df["a"], ct])
            df["j"] = self.utils.coalesce([ct, df["a"]])
            df["k"] = self.utils.coalesce([df["a"], None])
            df["l"] = self.utils.coalesce([None, df["a"]])
            df["m"] = self.utils.coalesce([df["a"], df["b"], df["c"]])
            df["n"] = self.utils.coalesce([df["a"], df["b"], df["c"], ct])

            assert_duck_eq(
                self.to_pd(df[list("hijklmn")]),
                """
                SELECT
                    COALESCE(NULL, TIMESTAMP '2020-01-01 15:00:00') AS h,
                    COALESCE(a, TIMESTAMP '2020-01-01 15:00:00') AS i,
                    COALESCE(TIMESTAMP '2020-01-01 15:00:00', a) AS j,
                    COALESCE(a, NULL) AS k,
                    COALESCE(NULL, a) AS l,
                    COALESCE(a,b,c) AS m,
                    COALESCE(a,b,c,TIMESTAMP '2020-01-01 15:00:00') AS n
                FROM a
                """,
                a=pdf,
                check_order=False,
            )

        def test_case_when(self):
            assert 4 == self.utils.case_when(default=4)
            assert 3 == self.utils.case_when((False, 1), (2, 3), default=4)
            assert 3 == self.utils.case_when((None, 1), (2, 3), default=4)
            assert 1 == self.utils.case_when((True, 1), (2, 3), default=4)
            assert 4 == self.utils.case_when((False, 1), (False, 3), default=4)

        def test_case_when_sql(self):
            pdf = make_rand_df(
                20, a=(bool, 10), b=(str, 10), c=(bool, 10), d=(str, 10), e=(str, 10)
            )

            df = self.to_df(pdf)
            df["h"] = self.utils.case_when((df["a"], df["b"]), (df["c"], df["d"]))
            df["i"] = self.utils.case_when(
                (df["a"], df["b"]), (df["c"], df["d"]), default=df["e"]
            )

            assert_duck_eq(
                self.to_pd(df[list("hi")]),
                """
                SELECT
                    CASE WHEN a THEN b WHEN c THEN d END AS h,
                    CASE WHEN a THEN b WHEN c THEN d ELSE e END AS i
                FROM a
                """,
                a=pdf,
                check_order=False,
            )

            pdf = make_rand_df(
                20,
                a=(float, 10),
                b=(float, 10),
                c=(float, 10),
                d=(float, 10),
                e=(float, 10),
            )

            df = self.to_df(pdf)
            df["h"] = self.utils.case_when(
                (df["a"] > 0.5, df["b"]), ((df["c"] > 0.5) | (df["a"] > 0.3), df["d"])
            )
            df["i"] = self.utils.case_when(
                (df["a"] > 0.5, df["b"]),
                ((df["c"] > 0.5) | (df["a"] > 0.3), df["d"]),
                default=df["e"],
            )
            df["j"] = self.utils.case_when(
                (df["a"] > 0.5, df["b"]),
                (df["a"] > 0.5, df["d"]),
                default=df["e"],
            )
            df["k"] = self.utils.case_when(
                (None, df["b"]),
                (df["a"] > 0.5, df["d"]),
                default=df["e"],
            )
            df["l"] = self.utils.case_when(
                (True, 2),
                (df["a"] > 0.5, df["d"]),
                default=df["e"],
            )
            df["m"] = self.utils.case_when(
                (True, None),
                (df["a"] > 0.5, df["d"]),
                default=df["e"],
            )

            assert_duck_eq(
                self.to_pd(df[list("hijklm")]),
                """
                SELECT
                    CASE
                        WHEN a>0.5 THEN b
                        WHEN c>0.5 OR a>0.3 THEN d END AS h,
                    CASE
                        WHEN a>0.5 THEN b
                        WHEN c>0.5 OR a>0.3 THEN d
                        ELSE e END AS i,
                    CASE
                        WHEN a>0.5 THEN b
                        WHEN a>0.5 THEN d
                        ELSE e END AS j,
                    CASE
                        WHEN NULL THEN b
                        WHEN a>0.5 THEN d
                        ELSE e END AS k,
                    CASE
                        WHEN TRUE THEN 2
                        WHEN a>0.5 THEN d
                        ELSE e END AS l,
                    CASE
                        WHEN TRUE THEN NULL
                        WHEN a>0.5 THEN d
                        ELSE e END AS m
                FROM a
                """,
                a=pdf,
                check_order=False,
            )

        def test_like(self):
            # nulls
            for p in [True, False]:
                for i in [True, False]:
                    assert (
                        self.utils.like(None, None, ignore_case=i, positive=p) is None
                    )
                    assert self.utils.like("x", None, ignore_case=i, positive=p) is None

            # empty
            assert self.utils.like("", "")
            assert not self.utils.like("abc", "")

            # simple
            assert not self.utils.like("abc", "aBc")
            assert self.utils.like("abc", "aBc", ignore_case=True)

            # start
            assert not self.utils.like("abc", "aB%")
            assert not self.utils.like("abc", "aB_")
            assert self.utils.like("abc", "aB%", ignore_case=True)
            assert self.utils.like("abc", "aB_", ignore_case=True)

            # end
            assert not self.utils.like("abc", "%Bc")
            assert not self.utils.like("abc", "_Bc")
            assert self.utils.like("abc", "%Bc", ignore_case=True)
            assert self.utils.like("abc", "_Bc", ignore_case=True)

            # start end
            assert not self.utils.like("abc", "A_c")
            assert not self.utils.like("abc", "A%c")
            assert self.utils.like("abc", "A_c", ignore_case=True)
            assert self.utils.like("abc", "A%c", ignore_case=True)

            # contain
            assert not self.utils.like("abc", "%B%")
            assert not self.utils.like("abc", "_B_")
            assert self.utils.like("abc", "%B%", ignore_case=True)
            assert self.utils.like("abc", "_B_", ignore_case=True)

            # not empty
            assert self.utils.like("abc", "_%")
            assert self.utils.like("abc", "%_")
            assert self.utils.like("abc", "%_%")

            # any
            assert self.utils.like("abc", "%")

        def test_like_sql(self):
            pdf = pd.DataFrame(
                dict(a=["abc", "ABC", "abd", "aBd", "", "ab\\%\\_c", None])
            )

            df = self.to_df(pdf)
            df["h"] = self.utils.like(df["a"], None)
            df["i"] = self.utils.like(df["a"], "")
            df["j"] = self.utils.like(df["a"], "abc", ignore_case=True)
            df["k"] = self.utils.like(df["a"], "aBc", ignore_case=False)
            df["l"] = self.utils.like(df["a"], "ab%", ignore_case=True)
            df["m"] = self.utils.like(df["a"], "aB%", ignore_case=False)
            df["n"] = self.utils.like(df["a"], "%bc", ignore_case=True)
            df["o"] = self.utils.like(df["a"], "%bc", ignore_case=False)
            df["p"] = self.utils.like(df["a"], "a%c", ignore_case=True)
            df["q"] = self.utils.like(df["a"], "a%c", ignore_case=False)
            df["r"] = self.utils.like(df["a"], "%bc%", ignore_case=True)
            df["s"] = self.utils.like(df["a"], "%bc%", ignore_case=False)
            df["t"] = self.utils.like(df["a"], "%_")
            df["u"] = self.utils.like(df["a"], "_%")
            df["v"] = self.utils.like(df["a"], "%_%")
            df["w"] = self.utils.like(df["a"], "_a%", ignore_case=True)
            df["x"] = self.utils.like(df["a"], "_a%", ignore_case=False)
            df["y"] = self.utils.like(df["a"], "%")

            assert_duck_eq(
                self.to_pd(df[list("hijklmnopqrstuvwxy")]),
                """
                SELECT
                    a LIKE NULL AS h,
                    a LIKE '' AS i,
                    a ILIKE 'abc' AS j,
                    a LIKE 'aBc' AS k,
                    a ILIKE 'ab%' AS l,
                    a LIKE 'aB%' AS m,
                    a ILIKE '%bc' AS n,
                    a LIKE '%bc' AS o,
                    a ILIKE 'a%c' AS p,
                    a LIKE 'a%c' AS q,
                    a ILIKE '%bc%' AS r,
                    a LIKE '%bc%' AS s,
                    a LIKE '%_' AS t,
                    a LIKE '_%' AS u,
                    a LIKE '%_%' AS v,
                    a ILIKE '_a%' AS w,
                    a LIKE '_a%' AS x,
                    a LIKE '%' AS y
                FROM a
                """,
                a=pdf,
                check_order=False,
            )

            df = self.to_df(pdf)
            df["h"] = self.utils.like(df["a"], None, positive=False)
            df["i"] = self.utils.like(df["a"], "", positive=False)
            df["j"] = self.utils.like(df["a"], "abc", ignore_case=True, positive=False)
            df["k"] = self.utils.like(df["a"], "aBc", ignore_case=False, positive=False)
            df["l"] = self.utils.like(df["a"], "ab%", ignore_case=True, positive=False)
            df["m"] = self.utils.like(df["a"], "aB%", ignore_case=False, positive=False)
            df["n"] = self.utils.like(df["a"], "%bc", ignore_case=True, positive=False)
            df["o"] = self.utils.like(df["a"], "%bc", ignore_case=False, positive=False)
            df["p"] = self.utils.like(df["a"], "a%c", ignore_case=True, positive=False)
            df["q"] = self.utils.like(df["a"], "a%c", ignore_case=False, positive=False)
            df["r"] = self.utils.like(df["a"], "%bc%", ignore_case=True, positive=False)
            df["s"] = self.utils.like(
                df["a"], "%bc%", ignore_case=False, positive=False
            )
            df["t"] = self.utils.like(df["a"], "%_", positive=False)
            df["u"] = self.utils.like(df["a"], "_%", positive=False)
            df["v"] = self.utils.like(df["a"], "%_%", positive=False)
            df["w"] = self.utils.like(df["a"], "_a%", ignore_case=True, positive=False)
            df["x"] = self.utils.like(df["a"], "_a%", ignore_case=False, positive=False)
            df["y"] = self.utils.like(df["a"], "%", positive=False)

            assert_duck_eq(
                self.to_pd(df[list("hijklmnopqrstuvwxy")]),
                """
                SELECT
                    a NOT LIKE NULL AS h,
                    a NOT LIKE '' AS i,
                    a NOT ILIKE 'abc' AS j,
                    a NOT LIKE 'aBc' AS k,
                    a NOT ILIKE 'ab%' AS l,
                    a NOT LIKE 'aB%' AS m,
                    a NOT ILIKE '%bc' AS n,
                    a NOT LIKE '%bc' AS o,
                    a NOT ILIKE 'a%c' AS p,
                    a NOT LIKE 'a%c' AS q,
                    a NOT ILIKE '%bc%' AS r,
                    a NOT LIKE '%bc%' AS s,
                    a NOT LIKE '%_' AS t,
                    a NOT LIKE '_%' AS u,
                    a NOT LIKE '%_%' AS v,
                    a NOT ILIKE '_a%' AS w,
                    a NOT LIKE '_a%' AS x,
                    a NOT LIKE '%' AS y
                FROM a
                """,
                a=pdf,
                check_order=False,
            )

        def test_cast_constant(self):
            assert self.utils.cast(None, bool) is None
            assert self.utils.cast(True, bool)
            assert not self.utils.cast(False, bool)
            assert self.utils.cast(float("nan"), bool) is None
            assert not self.utils.cast(0, bool)
            assert 1 == self.utils.cast(1, bool)
            assert 1 == self.utils.cast(-2, bool)
            assert 0 == self.utils.cast(0.0, bool)
            assert 1 == self.utils.cast(0.1, bool)
            assert 1 == self.utils.cast(-0.2, bool)
            assert 1 == self.utils.cast(float("inf"), bool)
            assert 1 == self.utils.cast(float("-inf"), bool)
            assert self.utils.cast("nan", bool) is None
            assert 1 == self.utils.cast("tRue", bool)
            assert 0 == self.utils.cast("fAlse", bool)

            assert self.utils.cast(None, int) is None
            assert 1 == self.utils.cast(True, int)
            assert 0 == self.utils.cast(False, int)
            assert self.utils.cast(float("nan"), int) is None
            assert 0 == self.utils.cast(0, int)
            assert 10 == self.utils.cast(10, int)
            assert 0 == self.utils.cast(0.0, int)
            assert 1 == self.utils.cast(1.1, int)
            assert -2 == self.utils.cast(-2.2, int)
            assert 0 == self.utils.cast("0", int)
            assert 10 == self.utils.cast("10", int)
            assert 0 == self.utils.cast("0.0", int)
            assert 1 == self.utils.cast("1.1", int)
            assert -2 == self.utils.cast("-2.2", int)
            assert self.utils.cast("nan", int) is None
            with raises(SlideCastError):
                assert self.utils.cast(float("inf"), int)
            with raises(SlideCastError):
                assert self.utils.cast(float("-inf"), int)

            assert self.utils.cast(None, float) is None
            assert 1.0 == self.utils.cast(True, float)
            assert 0.0 == self.utils.cast(False, float)
            assert self.utils.cast(float("nan"), float) is None
            assert 0.0 == self.utils.cast(0, float)
            assert 10.0 == self.utils.cast(10, float)
            assert 0.0 == self.utils.cast(0.0, float)
            assert 1.1 == self.utils.cast(1.1, float)
            assert -2.2 == self.utils.cast(-2.2, float)
            assert 0.0 == self.utils.cast("0", float)
            assert 10.0 == self.utils.cast("10", float)
            assert 0.0 == self.utils.cast("0.0", float)
            assert 1.1 == self.utils.cast("1.1", float)
            assert -2.2 == self.utils.cast("-2.2", float)
            assert self.utils.cast("nan", float) is None
            assert np.isinf(self.utils.cast(float("inf"), float))
            assert np.isinf(self.utils.cast(float("-inf"), float))

            assert self.utils.cast(None, str) is None
            assert "true" == self.utils.cast(True, str)
            assert "false" == self.utils.cast(False, str)
            assert "true" == self.utils.cast(-10, str, bool)
            assert "false" == self.utils.cast(0, str, bool)
            assert "10" == self.utils.cast(10, str)
            assert "0" == self.utils.cast(0, str)
            assert "10.0" == self.utils.cast(10.0, str)
            assert "-10.0" == self.utils.cast(-10.0, str)
            assert self.utils.cast(float("nan"), str) is None
            assert "inf" == self.utils.cast(float("inf"), str)
            assert "-inf" == self.utils.cast(float("-inf"), str)
            assert "xy" == self.utils.cast("xy", str)
            assert isinstance(self.utils.cast(date(2020, 1, 1), str), str)
            assert "2020-01-01" == self.utils.cast(date(2020, 1, 1), str)
            assert "2020-01-01 15:00:00" == self.utils.cast(
                datetime(2020, 1, 1, 15), str
            )
            assert self.utils.cast(pd.NaT, str) is None

            assert self.utils.cast(None, "date") is None
            assert self.utils.cast(None, "datetime") is None
            assert self.utils.cast("nat", "date") is None
            assert self.utils.cast("nat", "datetime") is None
            assert date(2020, 1, 1) == self.utils.cast("2020-01-01", "date")
            assert date(2020, 1, 1) == self.utils.cast("2020-01-01 15:00:00", "date")
            assert datetime(2020, 1, 1) == self.utils.cast("2020-01-01", "datetime")
            assert datetime(2020, 1, 1, 15) == self.utils.cast(
                "2020-01-01 15:00:00", "datetime"
            )

        def test_cast_bool(self):
            # happy path
            pdf = pd.DataFrame(
                dict(
                    a=[True, False, True],
                )
            )

            df = self.to_df(pdf)
            df["h"] = self.utils.cast(df.a, int)
            df["i"] = self.utils.cast(df.a, float)
            df["j"] = self.utils.cast(df.a, bool)
            df["k"] = self.utils.cast(df.a, str)

            assert_pdf_eq(
                self.to_pd(df[list("hijk")]),
                pd.DataFrame(
                    dict(
                        h=[1, 0, 1],
                        i=[1.0, 0.0, 1.0],
                        j=[True, False, True],
                        k=["true", "false", "true"],
                    ),
                ),
                check_order=False,
            )

            # from bool with None
            pdf = pd.DataFrame(
                dict(
                    a=[True, False, None],
                )
            )

            df = self.to_df(pdf)
            df["h"] = self.utils.cast(df.a, int, bool)
            df["i"] = self.utils.cast(df.a, float)
            df["j"] = self.utils.cast(df.a, bool, bool)
            df["k"] = self.utils.cast(df.a, str, bool)

            assert_pdf_eq(
                self.to_pd(df[list("hijk")]),
                pd.DataFrame(
                    dict(
                        h=[1, 0, None],
                        i=[1.0, 0.0, None],
                        j=[True, False, None],
                        k=["true", "false", None],
                    ),
                ),
                check_order=False,
            )

            # from float with None
            pdf = pd.DataFrame(
                dict(
                    a=[2.0, 0.0, -2.0, None, float("nan")],
                )
            )

            df = self.to_df(pdf)
            df["h"] = self.utils.cast(df.a, bool)

            assert_pdf_eq(
                self.to_pd(df[list("h")]),
                pd.DataFrame(
                    dict(
                        h=[True, False, True, None, None],
                    ),
                ),
                check_order=False,
            )

            # from int
            pdf = pd.DataFrame(
                dict(
                    a=[2, 0, -2],
                )
            )

            df = self.to_df(pdf)
            df["h"] = self.utils.cast(df.a, bool)

            assert_pdf_eq(
                self.to_pd(df[list("h")]),
                pd.DataFrame(
                    dict(
                        h=[True, False, True],
                    ),
                ),
                check_order=False,
            )

            # from bool with None to various
            pdf = pd.DataFrame(
                dict(
                    a=[1.0, 0.0, None],
                )
            )

            df = self.to_df(pdf)
            df["h"] = self.utils.cast(df.a, int, bool)
            df["i"] = self.utils.cast(df.a, float, bool)
            df["j"] = self.utils.cast(df.a, bool, bool)
            df["k"] = self.utils.cast(df.a, str, bool)

            assert_pdf_eq(
                self.to_pd(df[list("hijk")]),
                pd.DataFrame(
                    dict(
                        h=[1, 0, None],
                        i=[1.0, 0.0, None],
                        j=[True, False, None],
                        k=["true", "false", None],
                    ),
                ),
                check_order=False,
            )

            pdf = pd.DataFrame(
                dict(
                    a=[2.0, 0.0, float("nan")],
                )
            )

            df = self.to_df(pdf)
            df["h"] = self.utils.cast(df.a, int, bool)
            df["i"] = self.utils.cast(df.a, float, bool)
            df["j"] = self.utils.cast(df.a, bool, bool)
            df["k"] = self.utils.cast(df.a, str, bool)

            assert_pdf_eq(
                self.to_pd(df[list("hijk")]),
                pd.DataFrame(
                    dict(
                        h=[1, 0, None],
                        i=[1.0, 0.0, None],
                        j=[2.0, 0.0, float("nan")],
                        k=["true", "false", None],
                    ),
                ),
                check_order=False,
            )

            # from strings
            pdf = pd.DataFrame(
                dict(
                    a=["tRue", "fAlse", "true"],
                    b=["tRue", "fAlse", None],
                    c=["1", "0", "abc"],
                    d=["1.0", "0.0", "abc"],
                )
            )

            df = self.to_df(pdf)
            df["h"] = self.utils.cast(df.a, bool, str)
            df["i"] = self.utils.cast(df.b, bool, str)
            df["j"] = self.utils.cast(df.c, bool, str)
            df["k"] = self.utils.cast(df.d, bool, str)

            assert_pdf_eq(
                self.to_pd(df[list("hijk")]),
                pd.DataFrame(
                    dict(
                        h=[True, False, True],
                        i=[True, False, None],
                        j=[True, False, None],
                        k=[True, False, None],
                    ),
                ),
                check_order=False,
            )

        def test_cast_int(self):
            # happy path
            pdf = pd.DataFrame(
                dict(
                    a=[True, False, True],
                    b=[2, 3, 4],
                    c=[1.1, 2.2, 3.3],
                    d=["1", "2", "3"],
                    e=["5.5", "6.6", "7.7"],
                )
            )

            df = self.to_df(pdf)
            df["h"] = self.utils.cast(df.a, int)
            df["i"] = self.utils.cast(df.b, int)
            df["j"] = self.utils.cast(df.c, int)
            df["k"] = self.utils.cast(df.d, int)
            df["l"] = self.utils.cast(df.e, int)

            assert_pdf_eq(
                self.to_pd(df[list("hijkl")]),
                pd.DataFrame(
                    dict(
                        h=[1, 0, 1],
                        i=[2, 3, 4],
                        j=[1, 2, 3],
                        k=[1, 2, 3],
                        l=[5, 6, 7],
                    ),
                ),
                check_order=False,
            )

            # from int with None
            pdf = pd.DataFrame(
                dict(
                    a=[2, 3, None],
                )
            )

            df = self.to_df(pdf)
            df["h"] = self.utils.cast(df.a, int)

            assert_pdf_eq(
                self.to_pd(df[list("h")]),
                pd.DataFrame(
                    dict(
                        h=[2, 3, None],
                    ),
                ),
                check_order=False,
            )

            # from float with None
            pdf = pd.DataFrame(
                dict(
                    a=[2.1, float("nan"), None],
                )
            )

            df = self.to_df(pdf)
            df["h"] = self.utils.cast(df.a, int)

            assert_pdf_eq(
                self.to_pd(df[list("h")]),
                pd.DataFrame(
                    dict(
                        h=[2, None, None],
                    ),
                ),
                check_order=False,
            )

            pdf = pd.DataFrame(
                dict(
                    a=[2.1, float("inf"), None],
                )
            )

            df = self.to_df(pdf)
            with raises(SlideCastError):
                df["h"] = self.utils.cast(df.a, int)

            # from string with None
            pdf = pd.DataFrame(
                dict(
                    a=["2.1", "naN", None],
                )
            )

            df = self.to_df(pdf)
            df["h"] = self.utils.cast(df.a, int)

            assert_pdf_eq(
                self.to_pd(df[list("h")]),
                pd.DataFrame(
                    dict(
                        h=[2, None, None],
                    ),
                ),
                check_order=False,
            )

            # overflow, TODO: pandas can't raise exception
            pdf = pd.DataFrame(
                dict(
                    a=[10000, -10000],
                )
            )

            # df = self.to_df(pdf)
            # with raises(SlideCastError):
            #    df["h"] = self.utils.cast(df.a, "int8")

        def test_cast_float(self):
            # happy path
            pdf = pd.DataFrame(
                dict(
                    a=[True, False, True],
                    b=[2, 3, 4],
                    c=[1.1, 2.2, 3.3],
                    d=[2.0, 0.0, -1.0],
                    e=["5.5", "6.6", "7.7"],
                )
            )

            df = self.to_df(pdf)
            df["h"] = self.utils.cast(df.a, float)
            df["i"] = self.utils.cast(df.b, float)
            df["j"] = self.utils.cast(df.c, float)
            df["k"] = self.utils.cast(df.d, float, bool)
            df["l"] = self.utils.cast(df.e, float)

            assert_pdf_eq(
                self.to_pd(df[list("hijkl")]),
                pd.DataFrame(
                    dict(
                        h=[1, 0, 1],
                        i=[2, 3, 4],
                        j=[1.1, 2.2, 3.3],
                        k=[1.0, 0.0, 1.0],
                        l=[5.5, 6.6, 7.7],
                    ),
                ),
                check_order=False,
            )

            # from float with None
            pdf = pd.DataFrame(
                dict(
                    a=[2.1, float("nan"), float("inf"), None],
                )
            )

            df = self.to_df(pdf)
            df["h"] = self.utils.cast(df.a, float)

            assert_pdf_eq(
                self.to_pd(df[list("h")]),
                pd.DataFrame(
                    dict(
                        h=[2.1, float("nan"), float("inf"), None],
                    ),
                ),
                check_order=False,
            )

            # from string with None
            pdf = pd.DataFrame(
                dict(
                    a=["2.1", "naN", "inf", "-inf", None],
                )
            )

            df = self.to_df(pdf)
            df["h"] = self.utils.cast(df.a, float)

            assert_pdf_eq(
                self.to_pd(df[list("h")]),
                pd.DataFrame(
                    dict(
                        h=[2.1, None, float("inf"), float("-inf"), None],
                    ),
                ),
                check_order=False,
            )

        def test_cast_str(self):
            # happy path
            pdf = pd.DataFrame(
                dict(
                    a=[False, True, True],
                    b=[2, 3, 4],
                    c=[1.1, 2.2, 3.3],
                    d=[
                        datetime(2020, 1, 2),
                        datetime(2020, 1, 3),
                        datetime(2020, 1, 4),
                    ],
                    e=["aa", "ab", "ac"],
                )
            )

            df = self.to_df(pdf)
            df["h"] = self.utils.cast(df.a, str)
            df["i"] = self.utils.cast(df.b, str)
            df["j"] = self.utils.cast(df.c, str)
            df["k"] = self.utils.cast(df.d, str)
            df["l"] = self.utils.cast(df.e, str)

            assert_pdf_eq(
                self.to_pd(df[list("hijkl")]),
                pd.DataFrame(
                    dict(
                        h=["false", "true", "true"],
                        i=["2", "3", "4"],
                        j=["1.1", "2.2", "3.3"],
                        k=["2020-01-02", "2020-01-03", "2020-01-04"],
                        l=["aa", "ab", "ac"],
                    ),
                ),
                check_order=False,
            )

            # from bool with None
            pdf = pd.DataFrame(
                dict(
                    a=[True, False, None],
                )
            )

            df = self.to_df(pdf)
            df["h"] = self.utils.cast(df.a, str, bool)

            assert_pdf_eq(
                self.to_pd(df[list("h")]),
                pd.DataFrame(
                    dict(
                        h=["true", "false", None],
                    ),
                ),
                check_order=False,
            )

            # from float with None
            pdf = pd.DataFrame(
                dict(
                    a=[2.1, float("nan"), float("inf"), None],
                )
            )

            df = self.to_df(pdf)
            df["h"] = self.utils.cast(df.a, str)

            assert_pdf_eq(
                self.to_pd(df[list("h")]),
                pd.DataFrame(
                    dict(
                        h=["2.1", None, "inf", None],
                    ),
                ),
                check_order=False,
            )

            # from int with None
            pdf = pd.DataFrame(
                dict(
                    a=[1, None],
                )
            )

            df = self.to_df(pdf)
            df["h"] = self.utils.cast(df.a, str, int)

            assert_pdf_eq(
                self.to_pd(df[list("h")]),
                pd.DataFrame(
                    dict(
                        h=["1", None],
                    ),
                ),
                check_order=False,
            )

            # from timestamp with None
            pdf = pd.DataFrame(
                dict(
                    a=[
                        datetime(2020, 1, 1),
                        datetime(2020, 1, 1, 15, 2, 3),
                        pd.NaT,
                        None,
                    ],
                )
            )

            df = self.to_df(pdf)
            df["h"] = self.utils.cast(df.a, str)

            assert_pdf_eq(
                self.to_pd(df[list("h")]),
                pd.DataFrame(
                    dict(
                        h=["2020-01-01 00:00:00", "2020-01-01 15:02:03", None, None],
                    ),
                ),
                check_order=False,
            )

            # from date with None
            pdf = pd.DataFrame(
                dict(
                    a=[
                        date(2020, 1, 1),
                        date(2020, 1, 2),
                        pd.NaT,
                        None,
                    ],
                )
            )

            df = self.to_df(pdf)
            df["h"] = self.utils.cast(df.a, str, "date")

            assert_pdf_eq(
                self.to_pd(df[list("h")]),
                pd.DataFrame(
                    dict(
                        h=["2020-01-01", "2020-01-02", None, None],
                    ),
                ),
                check_order=False,
            )

        def test_cast_time(self):
            # happy path
            pdf = pd.DataFrame(
                dict(
                    a=["2020-01-01", "2020-01-02", "2020-01-03"],
                    b=[
                        "2020-01-01 01:00:00",
                        "2020-01-02 14:00:00",
                        "2020-01-03 15:00:00",
                    ],
                )
            )

            df = self.to_df(pdf)
            df["h"] = self.utils.cast(df.a, "date")
            df["i"] = self.utils.cast(df.a, "datetime")
            df["j"] = self.utils.cast(df.b, "date")
            df["k"] = self.utils.cast(df.b, "datetime")

            assert_pdf_eq(
                self.to_pd(df[list("hijk")]),
                pd.DataFrame(
                    dict(
                        h=[date(2020, 1, 1), date(2020, 1, 2), date(2020, 1, 3)],
                        i=[
                            datetime(2020, 1, 1),
                            datetime(2020, 1, 2),
                            datetime(2020, 1, 3),
                        ],
                        j=[date(2020, 1, 1), date(2020, 1, 2), date(2020, 1, 3)],
                        k=[
                            datetime(2020, 1, 1, 1),
                            datetime(2020, 1, 2, 14),
                            datetime(2020, 1, 3, 15),
                        ],
                    ),
                ),
                check_order=False,
            )

            # str -> date with None
            pdf = pd.DataFrame(
                dict(
                    a=["2020-01-01", "2020-01-02", None],
                )
            )

            df = self.to_df(pdf)
            df["h"] = self.utils.cast(df.a, "date")

            assert_pdf_eq(
                self.to_pd(df[list("h")]),
                pd.DataFrame(
                    dict(
                        h=[date(2020, 1, 1), date(2020, 1, 2), None],
                    ),
                ),
                check_order=False,
            )

            # str -> datetime with None
            pdf = pd.DataFrame(
                dict(
                    a=["2020-01-01 11:00:00", "2020-01-02 12:00:00", None],
                )
            )

            df = self.to_df(pdf)
            df["h"] = self.utils.cast(df.a, "datetime")

            assert_pdf_eq(
                self.to_pd(df[list("h")]),
                pd.DataFrame(
                    dict(
                        h=[datetime(2020, 1, 1, 11), datetime(2020, 1, 2, 12), None],
                    ),
                ),
                check_order=False,
            )

        def test_cols_to_df(self):
            df = self.to_df([["a", 1]], "a:str,b:long")
            res = self.utils.cols_to_df([df["b"], df["a"]])
            assert_pdf_eq(res, self.to_pd(self.to_df([[1, "a"]], "b:long,a:str")))
            res = self.utils.cols_to_df([df["b"], df["a"]], ["x", "y"])
            assert_pdf_eq(res, self.to_pd(self.to_df([[1, "a"]], "x:long,y:str")))

            res = self.utils.cols_to_df([123, df["a"]], names=["x", "y"])
            assert_pdf_eq(res, self.to_pd(self.to_df([[123, "a"]], "x:long,y:str")))

            with raises(SlideInvalidOperation):
                res = self.utils.cols_to_df([123, 456], names=["x", "y"])

        def test_to_schema(self):
            df = self.to_df([[1.0, 2], [2.0, 3]])
            raises(ValueError, lambda: self.utils.to_schema(df))
            df = self.to_df([[1.0, 2], [2.0, 3]], columns=["x", "y"])
            assert list(pa.Schema.from_pandas(df)) == list(self.utils.to_schema(df))
            df = self.to_df([["a", 2], ["b", 3]], columns=["x", "y"])
            assert list(pa.Schema.from_pandas(df)) == list(self.utils.to_schema(df))
            df = self.to_df([], columns=["x", "y"])
            df = df.astype(dtype={"x": np.int32, "y": np.dtype("object")})
            assert [pa.field("x", pa.int32()), pa.field("y", pa.string())] == list(
                self.utils.to_schema(df)
            )
            df = self.to_df([[1, "x"], [2, "y"]], columns=["x", "y"])
            df = df.astype(dtype={"x": np.int32, "y": np.dtype("object")})
            assert list(pa.Schema.from_pandas(df)) == list(self.utils.to_schema(df))
            df = self.to_df([[1, "x"], [2, "y"]], columns=["x", "y"])
            df = df.astype(dtype={"x": np.int32, "y": np.dtype(str)})
            assert list(pa.Schema.from_pandas(df)) == list(self.utils.to_schema(df))
            df = self.to_df([[1, "x"], [2, "y"]], columns=["x", "y"])
            df = df.astype(dtype={"x": np.int32, "y": np.dtype("str")})
            assert list(pa.Schema.from_pandas(df)) == list(self.utils.to_schema(df))

            # timestamp test
            df = self.to_df(
                [[datetime(2020, 1, 1, 2, 3, 4, 5), date(2020, 2, 2)]],
                columns=["a", "b"],
            )
            assert list(expression_to_schema("a:datetime,b:date")) == list(
                self.utils.to_schema(df)
            )

            # test index
            df = self.to_df([[3.0, 2], [2.0, 3]], columns=["x", "y"])
            df = df.sort_values(["x"])
            assert list(pa.Schema.from_pandas(df, preserve_index=False)) == list(
                self.utils.to_schema(df)
            )
            df.index.name = "x"
            raises(ValueError, lambda: self.utils.to_schema(df))
            df = df.reset_index(drop=True)
            assert list(pa.Schema.from_pandas(df)) == list(self.utils.to_schema(df))
            df["p"] = "p"
            df = df.set_index(["p"])
            df.index.name = None
            raises(ValueError, lambda: self.utils.to_schema(df))

        def test_as_array_iterable(self):
            df = self.to_df([], "a:str,b:int")
            assert [] == self.utils.as_array(df)
            assert [] == self.utils.as_array(df, type_safe=True)

            df = self.to_df([["a", 1]], "a:str,b:int")
            assert [["a", 1]] == self.utils.as_array(df)
            assert [["a", 1]] == self.utils.as_array(df, columns=["a", "b"])
            assert [[1, "a"]] == self.utils.as_array(df, columns=["b", "a"])
            assert [[1, "a"]] == self.utils.as_array(
                df, columns=["b", "a"], schema=Schema("a:str,b:int").pa_schema
            )

            # prevent pandas auto type casting
            df = self.to_df([[1.0, 1.1]], "a:double,b:int", null_safe=True)
            data = self.utils.as_array(df, schema=Schema("a:double,b:int").pa_schema)
            assert [[1.0, 1]] == data
            assert isinstance(data[0][0], float)
            assert isinstance(data[0][1], int)
            assert [[1.0, 1]] == self.utils.as_array(df, columns=["a", "b"])
            assert [[1, 1.0]] == self.utils.as_array(df, columns=["b", "a"])

            df = self.to_df([[np.float64(1.0), 1.1]], "a:double,b:int", null_safe=True)
            assert [[1.0, 1]] == self.utils.as_array(df)
            assert isinstance(self.utils.as_array(df)[0][0], float)
            assert isinstance(self.utils.as_array(df)[0][1], int)

            df = self.to_df(
                [[pd.Timestamp("2020-01-01"), 1.1]],
                "a:datetime,b:int",
                null_safe=True,
            )
            assert [[datetime(2020, 1, 1), 1]] == self.utils.as_array(df)
            assert isinstance(self.utils.as_array(df, type_safe=True)[0][0], datetime)
            assert isinstance(self.utils.as_array(df, type_safe=True)[0][1], int)

            df = self.to_df([[pd.NaT, 1.1]], "a:datetime,b:int", null_safe=True)
            assert self.utils.as_array(df, type_safe=True)[0][0] is None
            assert isinstance(self.utils.as_array(df, type_safe=True)[0][1], int)

            df = self.to_df([[1.0, 1.1]], "a:double,b:int", null_safe=True)
            assert [[1.0, 1]] == self.utils.as_array(df, type_safe=True)
            assert isinstance(self.utils.as_array(df)[0][0], float)
            assert isinstance(self.utils.as_array(df)[0][1], int)

        def test_as_array_iterable_datetime(self):
            df = self.to_df(
                [[datetime(2020, 1, 1, 2, 3, 4, 5), date(2020, 2, 2)]],
                columns=["a", "b"],
            )
            v1 = list(self.utils.as_array_iterable(df, type_safe=True))[0]
            v2 = list(
                self.utils.as_array_iterable(
                    df, schema=expression_to_schema("a:datetime,b:date"), type_safe=True
                )
            )[0]
            assert v1[0] == v2[0]
            assert not isinstance(v1[0], pd.Timestamp)
            assert type(v1[0]) == datetime
            assert type(v1[0]) == type(v2[0])

            assert v1[1] == v2[1]
            assert not isinstance(v1[1], pd.Timestamp)
            assert type(v1[1]) == date
            assert type(v1[1]) == type(v2[1])

        def test_nested(self):
            # data = [[dict(b=[30, "40"])]]
            # s = expression_to_schema("a:{a:str,b:[int]}")
            # df = self.to_df(data, "a:{a:str,b:[int]}")
            # a = df.as_array(type_safe=True)
            # assert [[dict(a=None, b=[30, 40])]] == a

            data = [[[json.dumps(dict(b=[30, "40"]))]]]
            df = self.to_df(data, "a:[{a:str,b:[int]}]", enforce_type=False)
            a = self.utils.as_array(
                df, schema=Schema("a:[{a:str,b:[int]}]").pa_schema, type_safe=True
            )
            assert [[[dict(a=None, b=[30, 40])]]] == a

            data = [[json.dumps(["1", 2])]]
            df = self.to_df(data, "a:[int]", enforce_type=False)
            a = self.utils.as_array(
                df, schema=Schema("a:[int]").pa_schema, type_safe=True
            )
            assert [[[1, 2]]] == a

        def test_binary(self):
            b = pickle.dumps("xyz")
            data = [[b, b"xy"]]
            df = self.to_df(data, "a:bytes,b:bytes")
            a = self.utils.as_array(df, type_safe=True)
            assert [[b, b"xy"]] == a

        def test_nan_none(self):
            df = self.to_df([[None, None]], "b:str,c:double", null_safe=True)
            assert df.iloc[0, 0] is None
            arr = self.utils.as_array(df, type_safe=True)[0]
            assert arr[0] is None
            assert arr[1] is None

            df = self.to_df([[None, None]], "b:int,c:bool", null_safe=True)
            arr = self.utils.as_array(df, type_safe=True)[0]
            assert arr[0] is None
            assert arr[1] is None

            df = self.to_df([], "b:str,c:double", null_safe=True)
            assert len(self.utils.as_array(df)) == 0

        def test_boolean_enforce(self):
            df = self.to_df(
                [[1, True], [2, False], [3, None]], "b:int,c:bool", null_safe=True
            )
            arr = self.utils.as_array(df, type_safe=True)
            assert [[1, True], [2, False], [3, None]] == arr

            df = self.to_df([[1, 1], [2, 0]], "b:int,c:bool", null_safe=True)
            arr = self.utils.as_array(df, type_safe=True)
            assert [[1, True], [2, False]] == arr

            df = self.to_df(
                [[1, "trUe"], [2, "False"], [3, None]],
                "b:int,c:bool",
                null_safe=True,
            )

            arr = self.utils.as_array(df, type_safe=True)
            assert [[1, True], [2, False], [3, None]] == arr

        def test_sql_group_by_apply(self):
            df = self.to_df(
                [["a", 1], ["a", 2], [None, 3]], "b:str,c:long", null_safe=True
            )

            def _m1(df):
                self.utils.ensure_compatible(df)
                df["ct"] = df.shape[0]
                return df

            res = self.utils.sql_groupby_apply(df, ["b"], _m1)
            self.utils.ensure_compatible(res)
            assert 3 == res.shape[0]
            assert 3 == res.shape[1]
            assert [["a", 1, 2], ["a", 2, 2], [None, 3, 1]] == res.values.tolist()

            res = self.utils.sql_groupby_apply(df, [], _m1)
            self.utils.ensure_compatible(res)
            assert 3 == res.shape[0]
            assert 3 == res.shape[1]
            assert [["a", 1, 3], ["a", 2, 3], [None, 3, 3]] == res.values.tolist()

            df = self.to_df(
                [[1.0, "a"], [1.0, "b"], [None, "c"], [None, "d"]],
                "b:double,c:str",
                null_safe=True,
            )
            res = self.utils.sql_groupby_apply(df, ["b"], _m1)
            assert [
                [1.0, "a", 2],
                [1.0, "b", 2],
                [float("nan"), "c", 2],
                [float("nan"), "d", 2],
            ].__repr__() == res.values.tolist().__repr__()

        def test_sql_group_by_apply_special_types(self):
            def _m1(df):
                self.utils.ensure_compatible(df)
                # df["ct"] = df.shape[0]
                return df.assign(ct=df.shape[0])

            df = self.to_df(
                [["a", 1.0], [None, 3.0], [None, 3.0], [None, None]],
                "a:str,b:double",
                null_safe=True,
            )
            res = self.utils.sql_groupby_apply(df, ["a", "b"], _m1)
            self.utils.ensure_compatible(res)
            assert 4 == res.shape[0]
            assert 3 == res.shape[1]
            assert_pdf_eq(
                self.to_pd(
                    self.to_df(
                        [
                            ["a", 1.0, 1],
                            [None, 3.0, 2],
                            [None, 3.0, 2],
                            [None, None, 1],
                        ],
                        "a:str,b:double,ct:int",
                        null_safe=True,
                    )
                ),
                self.to_pd(res),
            )

            dt = datetime.now()
            df = self.to_df(
                [["a", dt], [None, dt], [None, dt], [None, None]],
                "a:str,b:datetime",
                null_safe=True,
            )
            res = self.utils.sql_groupby_apply(df, ["a", "b"], _m1)
            self.utils.ensure_compatible(res)
            assert 4 == res.shape[0]
            assert 3 == res.shape[1]
            assert_pdf_eq(
                self.to_pd(
                    self.to_df(
                        [["a", dt, 1], [None, dt, 2], [None, dt, 2], [None, None, 1]],
                        "a:str,b:datetime,ct:int",
                        null_safe=True,
                    )
                ),
                self.to_pd(res),
            )

            dt = date(2020, 1, 1)
            df = self.to_df(
                [["a", dt], [None, dt], [None, dt], [None, None]],
                "a:str,b:date",
                null_safe=True,
            )
            res = self.utils.sql_groupby_apply(df, ["a", "b"], _m1)
            self.utils.ensure_compatible(res)
            assert 4 == res.shape[0]
            assert 3 == res.shape[1]
            assert_pdf_eq(
                self.to_pd(
                    self.to_df(
                        [["a", dt, 1], [None, dt, 2], [None, dt, 2], [None, None, 1]],
                        "a:str,b:date,ct:int",
                        null_safe=True,
                    )
                ),
                self.to_pd(res),
            )

            dt = date(2020, 1, 1)
            df = self.to_df(
                [["a", dt], ["b", dt], ["b", dt], ["b", None]],
                "a:str,b:date",
                null_safe=True,
            )
            res = self.utils.sql_groupby_apply(df, ["a", "b"], _m1)
            self.utils.ensure_compatible(res)
            assert 4 == res.shape[0]
            assert 3 == res.shape[1]
            assert_pdf_eq(
                self.to_pd(
                    self.to_df(
                        [["a", dt, 1], ["b", dt, 2], ["b", dt, 2], ["b", None, 1]],
                        "a:str,b:date,ct:int",
                        null_safe=True,
                    )
                ),
                self.to_pd(res),
            )

        def test_drop_duplicates(self):
            def assert_eq(df1, df2):
                d1 = self.to_pd(self.utils.drop_duplicates(df1))
                assert_pdf_eq(d1, self.to_pd(df2), check_order=False)

            a = self.to_df([["x", "a"], ["x", "a"], [None, None]], ["a", "b"])
            assert_eq(a, pd.DataFrame([["x", "a"], [None, None]], columns=["a", "b"]))

        def test_drop_duplicates_sql(self):
            df = make_rand_df(100, a=int, b=int)
            assert_duck_eq(
                self.to_pd(self.utils.drop_duplicates(df)),
                "SELECT DISTINCT * FROM a",
                a=df,
                check_order=False,
            )

            df = make_rand_df(100, a=(int, 50), b=(int, 50))
            assert_duck_eq(
                self.to_pd(self.utils.drop_duplicates(df)),
                "SELECT DISTINCT * FROM a",
                a=df,
                check_order=False,
            )

            df = make_rand_df(100, a=(int, 50), b=(str, 50), c=float)
            assert_duck_eq(
                self.to_pd(self.utils.drop_duplicates(df)),
                "SELECT DISTINCT * FROM a",
                a=df,
                check_order=False,
            )

            df = make_rand_df(100, a=(int, 50), b=(datetime, 50), c=float)
            assert_duck_eq(
                self.to_pd(self.utils.drop_duplicates(df)),
                "SELECT DISTINCT * FROM a",
                a=df,
                check_order=False,
            )

        def test_union(self):
            def assert_eq(df1, df2, unique, expected, expected_cols):
                res = self.to_pd(self.utils.union(df1, df2, unique=unique))
                assert_pdf_eq(
                    res,
                    pd.DataFrame(expected, columns=expected_cols),
                    check_order=False,
                )

            a = self.to_df([["x", "a"], ["x", "a"], [None, None]], ["a", "b"])
            b = self.to_df([["xx", "aa"], [None, None], ["a", "x"]], ["b", "a"])
            assert_eq(
                a,
                b,
                False,
                [
                    ["x", "a"],
                    ["x", "a"],
                    [None, None],
                    ["xx", "aa"],
                    [None, None],
                    ["a", "x"],
                ],
                ["a", "b"],
            )
            assert_eq(
                a,
                b,
                True,
                [["x", "a"], ["xx", "aa"], [None, None], ["a", "x"]],
                ["a", "b"],
            )

        def test_union_sql(self):
            a = make_rand_df(30, b=(int, 10), c=(str, 10), d=(datetime, 10))
            b = make_rand_df(80, b=(int, 50), c=(str, 50), d=(datetime, 50))
            c = make_rand_df(100, b=(int, 50), c=(str, 50), d=(datetime, 50))
            d = self.to_pd(
                self.utils.union(self.utils.union(a, b, unique=True), c, unique=True)
            )
            assert_duck_eq(
                d,
                """
                SELECT * FROM a
                    UNION SELECT * FROM b
                    UNION SELECT * FROM c
                """,
                a=a,
                b=b,
                c=c,
            )
            e = self.to_pd(
                self.utils.union(self.utils.union(a, b, unique=False), c, unique=False)
            )
            assert_duck_eq(
                e,
                """
                SELECT * FROM a
                    UNION ALL SELECT * FROM b
                    UNION ALL SELECT * FROM c
                """,
                a=a,
                b=b,
                c=c,
            )

        def test_intersect(self):
            def assert_eq(df1, df2, unique, expected, expected_cols):
                res = self.to_pd(self.utils.intersect(df1, df2, unique=unique))
                assert_pdf_eq(res, expected, expected_cols)

            a = self.to_df([["x", "a"], ["x", "a"], [None, None]], ["a", "b"])
            b = self.to_df(
                [["xx", "aa"], [None, None], [None, None], ["a", "x"]], ["b", "a"]
            )
            assert_eq(a, b, False, [[None, None]], ["a", "b"])
            assert_eq(a, b, True, [[None, None]], ["a", "b"])
            b = self.to_df([["xx", "aa"], [None, None], ["x", "a"]], ["b", "a"])
            assert_eq(a, b, False, [["x", "a"], ["x", "a"], [None, None]], ["a", "b"])
            assert_eq(a, b, True, [["x", "a"], [None, None]], ["a", "b"])

        def test_intersect_sql(self):
            a = make_rand_df(30, b=(int, 10), c=(str, 10))
            b = make_rand_df(80, b=(int, 50), c=(str, 50))
            c = make_rand_df(100, b=(int, 50), c=(str, 50))
            d = self.to_pd(
                self.utils.intersect(
                    self.utils.intersect(c, b, unique=True), a, unique=True
                )
            )
            assert_duck_eq(
                d,
                """
                SELECT * FROM c
                    INTERSECT SELECT * FROM b
                    INTERSECT SELECT * FROM a
                """,
                a=a,
                b=b,
                c=c,
            )

            a = make_rand_df(30, b=(int, 10), c=(datetime, 10))
            b = make_rand_df(80, b=(int, 50), c=(datetime, 50))
            c = make_rand_df(100, b=(int, 50), c=(datetime, 50))
            d = self.to_pd(
                self.utils.intersect(
                    self.utils.intersect(c, b, unique=True), a, unique=True
                )
            )
            assert_duck_eq(
                d,
                """
                SELECT * FROM c
                    INTERSECT SELECT * FROM b
                    INTERSECT SELECT * FROM a
                """,
                a=a,
                b=b,
                c=c,
            )

        def test_except(self):
            def assert_eq(df1, df2, unique, expected, expected_cols):
                res = self.to_pd(self.utils.except_df(df1, df2, unique=unique))
                assert_pdf_eq(res, expected, expected_cols)

            a = self.to_df([["x", "a"], ["x", "a"], [None, None]], ["a", "b"])
            b = self.to_df([["xx", "aa"], [None, None], ["a", "x"]], ["b", "a"])
            assert_eq(a, b, False, [["x", "a"], ["x", "a"]], ["a", "b"])
            assert_eq(a, b, True, [["x", "a"]], ["a", "b"])
            b = self.to_df([["xx", "aa"], [None, None], ["x", "a"]], ["b", "a"])
            assert_eq(a, b, False, [], ["a", "b"])
            assert_eq(a, b, True, [], ["a", "b"])

        def test_except_sql(self):
            a = make_rand_df(30, b=(int, 10), c=(str, 10))
            b = make_rand_df(80, b=(int, 50), c=(str, 50))
            c = make_rand_df(100, b=(int, 50), c=(str, 50))
            d = self.to_pd(
                self.utils.except_df(
                    self.utils.except_df(c, b, unique=True), a, unique=True
                )
            )
            assert_duck_eq(
                d,
                """
                SELECT * FROM c
                    EXCEPT SELECT * FROM b
                    EXCEPT SELECT * FROM a
                """,
                a=a,
                b=b,
                c=c,
            )

            a = make_rand_df(30, b=(int, 10), c=(datetime, 10))
            b = make_rand_df(80, b=(int, 50), c=(datetime, 50))
            c = make_rand_df(100, b=(int, 50), c=(datetime, 50))
            d = self.to_pd(
                self.utils.except_df(
                    self.utils.except_df(c, b, unique=True), a, unique=True
                )
            )
            assert_duck_eq(
                d,
                """
                SELECT * FROM c
                    EXCEPT SELECT * FROM b
                    EXCEPT SELECT * FROM a
                """,
                a=a,
                b=b,
                c=c,
            )

        def test_joins(self):
            def assert_eq(df1, df2, join_type, on, expected, expected_cols):
                res = self.to_pd(self.utils.join(df1, df2, join_type=join_type, on=on))
                assert_pdf_eq(res, expected, expected_cols, check_order=False)

            df1 = self.to_df([[0, 1], [2, 3]], ["a", "b"])
            df2 = self.to_df([[0, 10], [20, 30]], ["a", "c"])
            df3 = self.to_df([[0, 1], [None, 3]], ["a", "b"])
            df4 = self.to_df([[0, 10], [None, 30]], ["a", "c"])
            assert_eq(df1, df2, "inner", ["a"], [[0, 1, 10]], ["a", "b", "c"])
            assert_eq(df3, df4, "inner", ["a"], [[0, 1, 10]], ["a", "b", "c"])
            assert_eq(df1, df2, "left_semi", ["a"], [[0, 1]], ["a", "b"])
            assert_eq(df3, df4, "left_semi", ["a"], [[0, 1]], ["a", "b"])
            assert_eq(df1, df2, "left_anti", ["a"], [[2, 3]], ["a", "b"])
            assert_eq(df3, df4, "left_anti", ["a"], [[None, 3]], ["a", "b"])
            assert_eq(
                df1,
                df2,
                "left_outer",
                ["a"],
                [[0, 1, 10], [2, 3, None]],
                ["a", "b", "c"],
            )
            assert_eq(
                df3,
                df4,
                "left_outer",
                ["a"],
                [[0, 1, 10], [None, 3, None]],
                ["a", "b", "c"],
            )
            assert_eq(
                df1,
                df2,
                "right_outer",
                ["a"],
                [[0, 1, 10], [20, None, 30]],
                ["a", "b", "c"],
            )
            assert_eq(
                df3,
                df4,
                "right_outer",
                ["a"],
                [[0, 1, 10], [None, None, 30]],
                ["a", "b", "c"],
            )
            assert_eq(
                df1,
                df2,
                "full_outer",
                ["a"],
                [[0, 1, 10], [2, 3, None], [20, None, 30]],
                ["a", "b", "c"],
            )
            assert_eq(
                df3,
                df4,
                "full_outer",
                ["a"],
                [[0, 1, 10], [None, 3, None], [None, None, 30]],
                ["a", "b", "c"],
            )

            df1 = self.to_df([[0, 1], [None, 3]], ["a", "b"])
            df2 = self.to_df([[0, 10], [None, 30]], ["c", "d"])
            assert_eq(
                df1,
                df2,
                "cross",
                [],
                [
                    [0, 1, 0, 10],
                    [None, 3, 0, 10],
                    [0, 1, None, 30],
                    [None, 3, None, 30],
                ],
                ["a", "b", "c", "d"],
            )

        def test_join_inner_sql(self):
            a = make_rand_df(100, a=(int, 40), b=(datetime, 40), c=(float, 40))
            b = make_rand_df(80, d=(float, 10), a=(int, 10), b=(datetime, 10))
            assert_duck_eq(
                self.to_df(self.utils.join(a, b, "inner", on=["a", "b"])),
                "SELECT a.*, d FROM a INNER JOIN b ON a.a=b.a AND a.b=b.b",
                a=a,
                b=b,
                check_order=False,
            )

        def test_join_left_sql(self):
            a = make_rand_df(100, a=(int, 40), b=(datetime, 40), c=(float, 40))
            b = make_rand_df(80, d=(float, 10), a=(int, 10), b=(datetime, 10))
            assert_duck_eq(
                self.to_df(self.utils.join(a, b, "left", on=["a", "b"])),
                "SELECT a.*, d FROM a LEFT JOIN b ON a.a=b.a AND a.b=b.b",
                a=a,
                b=b,
                check_order=False,
            )

        def test_join_right_sql(self):
            a = make_rand_df(100, a=(int, 40), b=(datetime, 40), c=(float, 40))
            b = make_rand_df(80, d=(float, 10), a=(int, 10), b=(datetime, 10))
            assert_duck_eq(
                self.to_df(self.utils.join(a, b, "right", on=["a", "b"])),
                "SELECT c, b.* FROM a RIGHT JOIN b ON a.a=b.a AND a.b=b.b",
                a=a,
                b=b,
                check_order=False,
            )

        def test_join_full_sql(self):
            a = make_rand_df(100, a=(int, 40), b=(datetime, 40), c=(float, 40))
            b = make_rand_df(80, d=(float, 10), a=(int, 10), b=(datetime, 10))
            assert_duck_eq(
                self.to_df(self.utils.join(a, b, "full", on=["a", "b"])),
                """SELECT COALESCE(a.a, b.a) AS a, COALESCE(a.b, b.b) AS b, c, d
                FROM a FULL JOIN b ON a.a=b.a AND a.b=b.b""",
                a=a,
                b=b,
                check_order=False,
            )

        def test_join_cross_sql(self):
            a = make_rand_df(10, a=(int, 4), b=(datetime, 4), c=(float, 4))
            b = make_rand_df(20, dd=(float, 1), aa=(int, 1), bb=(datetime, 1))
            assert_duck_eq(
                self.to_df(self.utils.join(a, b, "cross", on=[])),
                "SELECT * FROM a CROSS JOIN b",
                a=a,
                b=b,
                check_order=False,
            )

        def test_join_semi(self):
            a = make_rand_df(100, a=(int, 40), b=(datetime, 40), c=(float, 40))
            b = make_rand_df(80, d=(float, 10), a=(int, 10), b=(datetime, 10))
            assert_duck_eq(
                self.to_df(self.utils.join(a, b, "semi", on=["a", "b"])),
                """SELECT a.* FROM a INNER JOIN (SELECT DISTINCT a,b FROM b) x
                ON a.a=x.a AND a.b=x.b
                """,
                a=a,
                b=b,
                check_order=False,
            )

        def test_join_anti(self):
            a = make_rand_df(100, a=(int, 40), b=(datetime, 40), c=(float, 40))
            b = make_rand_df(80, d=(float, 10), a=(int, 10), b=(datetime, 10))
            assert_duck_eq(
                self.to_df(self.utils.join(a, b, "anti", on=["a", "b"])),
                """SELECT a.* FROM a LEFT JOIN (SELECT a,b, 1 AS z FROM b) x
                ON a.a=x.a AND a.b=x.b WHERE z IS NULL
                """,
                a=a,
                b=b,
                check_order=False,
            )

        def test_join_multi_sql(self):
            a = make_rand_df(100, a=(int, 40), b=(datetime, 40), c=(float, 40))
            b = make_rand_df(80, d=(float, 10), a=(int, 10), b=(datetime, 10))
            c = make_rand_df(80, dd=(float, 10), a=(int, 10), b=(datetime, 10))
            assert_duck_eq(
                self.to_df(
                    self.utils.join(
                        self.utils.join(a, b, "inner", on=["a", "b"]),
                        c,
                        "inner",
                        on=["a", "b"],
                    )
                ),
                """
                SELECT a.*,d,dd FROM a
                    INNER JOIN b ON a.a=b.a AND a.b=b.b
                    INNER JOIN c ON a.a=c.a AND c.b=b.b
                """,
                a=a,
                b=b,
                c=c,
            )
