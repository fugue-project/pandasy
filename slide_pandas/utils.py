from typing import List, Optional
import pandas as pd
from slide.utils import SlideUtils


class PandasUtils(SlideUtils[pd.DataFrame, pd.Series]):
    """A collection of pandas utils"""

    def cols_to_df(
        self, cols: List[pd.Series], names: Optional[List[str]] = None
    ) -> pd.DataFrame:
        if names is None:
            return pd.DataFrame({c.name: c for c in cols})
        else:
            return pd.DataFrame({name: c for name, c in zip(names, cols)})
