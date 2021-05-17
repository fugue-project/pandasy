from triad.utils.pyarrow import expression_to_schema
from pandasy.utils import PandasyUtils
from pandasy_test.suite import PandasyTestSuite

from pandasy_pandas.utils import PandasUtils
import pandas as pd
from typing import Any


class PandasTests(PandasyTestSuite.Tests):
    def make_utils(self) -> PandasyUtils:
        return PandasUtils()

    def to_df(self, data: Any, columns: Any = None, enforce_type: bool = False):
        if isinstance(columns, str):
            s = expression_to_schema(columns)
            df = pd.DataFrame(data, columns=s.names)
            if enforce_type:
                df = self.utils.enforce_type(df, s, True)
        else:
            df = pd.DataFrame(data, columns=columns)
        return df
