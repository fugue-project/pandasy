from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd
from slide.exceptions import SlideInvalidOperation
from slide.utils import SlideUtils
from triad.utils.assertion import assert_or_throw

_KEY_COL_NAME = "__safe_groupby_key__"
_DEFAULT_DATETIME = datetime(2000, 1, 1)


class PandasUtils(SlideUtils[pd.DataFrame, pd.Series]):
    """A collection of pandas utils"""

    def is_series(self, obj: Any) -> bool:
        return isinstance(obj, pd.Series)

    def to_series(self, obj: Any, name: Optional[str] = None) -> pd.Series:
        if self.is_series(obj):
            if name is not None and obj.name != name:
                return obj.rename(name)
            return obj
        if isinstance(obj, (np.ndarray, list)):
            return pd.Series(obj, name=name)
        raise NotImplementedError  # pragma: no cover

    def to_constant_series(
        self,
        constant: Any,
        from_series: pd.Series,
        dtype: Any = None,
        name: Optional[str] = None,
    ) -> pd.Series:
        return pd.Series(constant, index=from_series.index, dtype=dtype, name=name)

    def cols_to_df(
        self, cols: List[pd.Series], names: Optional[List[str]] = None
    ) -> pd.DataFrame:
        assert_or_throw(
            any(self.is_series(s) for s in cols),
            SlideInvalidOperation("at least one value in cols should be series"),
        )
        if names is None:
            return pd.DataFrame({c.name: c for c in cols})
        return pd.DataFrame(dict(zip(names, cols)))

    def sql_groupby_apply(
        self,
        df: pd.DataFrame,
        cols: List[str],
        func: Callable[[pd.DataFrame], pd.DataFrame],
        **kwargs: Any,
    ) -> pd.DataFrame:
        if pd.__version__ < "1.2":  # pragma: no cover
            # https://github.com/pandas-dev/pandas/issues/35889
            return self._sql_groupby_apply_older_version(df, cols, func, **kwargs)
        self.ensure_compatible(df)
        if len(cols) == 0:
            return func(df)
        return (
            df.groupby(cols, dropna=False)
            .apply(lambda df: func(df.reset_index(drop=True)), **kwargs)
            .reset_index(drop=True)
        )

    def _sql_groupby_apply_older_version(
        self,
        df: pd.DataFrame,
        cols: List[str],
        func: Callable[[pd.DataFrame], pd.DataFrame],
        **kwargs: Any,
    ) -> pd.DataFrame:  # pragma: no cover
        def _wrapper(keys: List[str], df: pd.DataFrame) -> pd.DataFrame:
            return func(df.drop(keys, axis=1).reset_index(drop=True))

        def _fillna_default(col: Any) -> Any:
            dtype = col.dtype
            if np.issubdtype(dtype, np.datetime64):
                return col.fillna(_DEFAULT_DATETIME)
            if np.issubdtype(dtype, np.str_) or np.issubdtype(
                dtype, np.string_
            ):  # pragma: no cover
                return col.fillna("")
            if np.issubdtype(dtype, np.bool_):
                return col.fillna(False)
            return col.fillna(0)

        self.ensure_compatible(df)
        if len(cols) == 0:
            return func(df)
        params: Dict[str, Any] = {}
        for c in cols:
            params[_KEY_COL_NAME + "null_" + c] = df[c].isnull()
            params[_KEY_COL_NAME + "fill_" + c] = _fillna_default(df[c])
        keys = list(params.keys())
        gdf = df.assign(**params)
        return (
            gdf.groupby(keys)
            .apply(lambda df: _wrapper(keys, df), **kwargs)
            .reset_index(drop=True)
        )
