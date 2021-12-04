from datetime import date, datetime
from typing import Any

import pandas as pd
from pytest import raises
from slide.exceptions import SlideIndexIncompatibleError
from slide.utils import SlideUtils
from slide_test.suite import SlideTestSuite
from triad import Schema
from triad.utils.pyarrow import expression_to_schema

from slide_pandas.utils import PandasUtils


class PandasTests(SlideTestSuite.Tests):
    def make_utils(self) -> SlideUtils:
        return PandasUtils()

    def to_pd(self, data: Any) -> pd.DataFrame:
        assert isinstance(data, pd.DataFrame)
        return data

    def to_df(
        self,
        data: Any,
        columns: Any = None,
        coerce: bool = True,
    ):
        if isinstance(columns, str):
            s = expression_to_schema(columns)
            df = pd.DataFrame(data, columns=s.names)
            if coerce:
                df = self.utils.cast_df(df.convert_dtypes(), s)
        else:
            df = pd.DataFrame(data, columns=columns).copy()
        return df

    def test_to_schema_2(self):
        # timestamp test
        df = self.to_df(
            [[datetime(2020, 1, 1, 2, 3, 4, 5), date(2020, 2, 2)]],
            columns=["a", "b"],
        )
        assert Schema("a:datetime,b:date") == Schema(self.utils.to_schema(df))

    def test_index_compatible(self):
        df = self.to_df([[3.0, 2], [2.1, 3]], columns=["x", "y"])
        df = df.sort_values(["x"])
        self.utils.ensure_compatible(df)
        df.index.name = "x"
        raises(SlideIndexIncompatibleError, lambda: self.utils.ensure_compatible(df))
        df = df.reset_index(drop=True)
        self.utils.ensure_compatible(df)
        df["p"] = "p"
        df = df.set_index(["p"])
        df.index.name = None
        raises(SlideIndexIncompatibleError, lambda: self.utils.ensure_compatible(df))
