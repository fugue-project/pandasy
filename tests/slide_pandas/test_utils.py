from triad.utils.pyarrow import expression_to_schema
from slide.utils import SlideUtils
from slide_test.suite import SlideTestSuite

from slide_pandas.utils import PandasUtils
import pandas as pd
from typing import Any


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
