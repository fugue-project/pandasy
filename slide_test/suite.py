import json
import pickle
from datetime import date, datetime
from typing import Any
from unittest import TestCase

import numpy as np
import pandas as pd
import pyarrow as pa
from pytest import raises
from slide.utils import SlideUtils
from triad import Schema
from triad.utils.pyarrow import expression_to_schema

from slide_test.utils import assert_duck_eq, assert_pdf_eq, make_rand_df


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
            # if isinstance(columns , str):
            # s = expression_to_schema(columns)
            # df = pd.DataFrame(data, columns=s.names)
            # self.native = PD_UTILS.null_safe(df, s, enforce)

            raise NotImplementedError

        def test_is_series(self):
            df = self.to_df([["a", 1]], "a:str,b:long")
            assert self.utils.is_series(df["a"])
            assert not self.utils.is_series(None)
            assert not self.utils.is_series(1)
            assert not self.utils.is_series("abc")

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

        def test_comparison_op(self):
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

        def test_cols_to_df(self):
            df = self.to_df([["a", 1]], "a:str,b:long")
            res = self.utils.cols_to_df([df["b"], df["a"]])
            assert_pdf_eq(res, self.to_pd(self.to_df([[1, "a"]], "b:long,a:str")))
            res = self.utils.cols_to_df([df["b"], df["a"]], ["x", "y"])
            assert_pdf_eq(res, self.to_pd(self.to_df([[1, "a"]], "x:long,y:str")))

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
