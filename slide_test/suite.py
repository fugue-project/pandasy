import json
import pickle
from datetime import date, datetime
from typing import Any
from unittest import TestCase

import numpy as np
import pandas as pd
import pyarrow as pa
from slide.utils import SlideUtils
from pytest import raises
from triad import Schema
from triad.utils.pandas_like import _DEFAULT_DATETIME
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

        def assert_eq(self, df1, df2) -> None:
            _1 = self.to_pd(df1)
            _2 = self.to_pd(df2)
            pd.testing.assert_frame_equal(_1, _2, check_dtype=False)

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

        def test_fillna_default(self):
            df = self.to_df([["a"], [None]], columns=["x"])
            s = self.utils.fillna_default(df["x"])
            assert ["a", 0] == s.tolist()

            df = self.to_df([["a"], ["b"]], columns=["x"])
            s = self.utils.fillna_default(df["x"].astype(np.str_))
            assert ["a", "b"] == s.tolist()

            dt = datetime.now()
            df = self.to_df([[dt], [None]], columns=["x"])
            s = self.utils.fillna_default(df["x"])
            assert [dt, _DEFAULT_DATETIME] == s.tolist()

            df = self.to_df([[True], [None]], columns=["x"])
            s = self.utils.fillna_default(df["x"])
            assert [True, 0] == s.tolist()

            df = self.to_df([[True], [False]], columns=["x"])
            s = self.utils.fillna_default(df["x"].astype(bool))
            assert [True, False] == s.tolist()

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
                df["ct"] = df.shape[0]
                return df

            df = self.to_df(
                [["a", 1.0], [None, 3.0], [None, 3.0], [None, None]],
                "a:str,b:double",
                null_safe=True,
            )
            res = self.utils.sql_groupby_apply(df, ["a", "b"], _m1)
            self.utils.ensure_compatible(res)
            assert 4 == res.shape[0]
            assert 3 == res.shape[1]
            self.assert_eq(
                self.to_df(
                    [["a", 1.0, 1], [None, 3.0, 2], [None, 3.0, 2], [None, None, 1]],
                    "a:str,b:double,ct:int",
                    null_safe=True,
                ),
                res,
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
            self.assert_eq(
                self.to_df(
                    [["a", dt, 1], [None, dt, 2], [None, dt, 2], [None, None, 1]],
                    "a:str,b:datetime,ct:int",
                    null_safe=True,
                ),
                res,
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
            self.assert_eq(
                self.to_df(
                    [["a", dt, 1], [None, dt, 2], [None, dt, 2], [None, None, 1]],
                    "a:str,b:date,ct:int",
                    null_safe=True,
                ),
                res,
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
            self.assert_eq(
                self.to_df(
                    [["a", dt, 1], ["b", dt, 2], ["b", dt, 2], ["b", None, 1]],
                    "a:str,b:date,ct:int",
                    null_safe=True,
                ),
                res,
            )
