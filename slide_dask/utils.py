from typing import Any, Callable, List, Optional, Union

import dask.dataframe as dd
import numpy as np
import pandas as pd
import pyarrow as pa
from slide.exceptions import SlideInvalidOperation
from slide.utils import SlideUtils
from triad.utils.assertion import assert_or_throw
from triad.utils.pyarrow import to_pandas_dtype
from pandas.api.types import is_object_dtype


class DaskUtils(SlideUtils[dd.DataFrame, dd.Series]):
    """A collection of Dask utils"""

    def is_series(self, obj: Any) -> bool:
        return isinstance(obj, dd.Series)

    def to_series(self, obj: Any, name: Optional[str] = None) -> dd.Series:
        if self.is_series(obj):
            if name is not None and obj.name != name:
                return obj.rename(name)
            return obj
        if isinstance(obj, (np.ndarray, list)):
            return dd.from_array(np.array(obj), columns=name)
        if isinstance(obj, pd.Series):
            res = dd.from_pandas(obj, chunksize=50000)
            if name is not None and res.name != name:
                return res.rename(name)
            return res
        raise NotImplementedError  # pragma: no cover

    def series_to_array(self, col: dd.Series) -> List[Any]:
        return col.compute().tolist()

    def scalar_to_series(
        self,
        scalar: Any,
        reference: Union[dd.Series, dd.DataFrame],
        dtype: Any = None,
        name: Optional[str] = None,
    ) -> dd.Series:
        if pa.types.is_nested(pa.scalar(scalar).type):
            assert_or_throw(
                dtype is None or is_object_dtype(dtype),
                ValueError(
                    "for nested scalar type, dtype must be None or dtype(object)"
                ),
            )
            if self.is_series(reference):
                return reference.map(lambda _: scalar, meta=(name, dtype))
            else:
                return reference[reference.columns[0]].map(
                    lambda _: scalar, meta=(name, dtype)
                )
        if dtype is not None:
            if self.is_series(reference):
                return reference.map(lambda _: scalar, meta=(name, dtype))
            else:
                return reference[reference.columns[0]].map(
                    lambda _: scalar, meta=(name, dtype)
                )
        tdf = reference.to_frame() if isinstance(reference, dd.Series) else reference
        tn = name or "_tmp_"
        tdf[tn] = scalar
        return tdf[tn]

    def cols_to_df(  # noqa: C901
        self,
        cols: List[Any],
        names: Optional[List[str]] = None,
        reference: Union[dd.Series, dd.DataFrame, None] = None,
    ) -> dd.DataFrame:
        _cols = list(cols)
        _ref: Any = None
        _nested: List[int] = []
        _ref_idx = -1
        for i in range(len(cols)):
            if self.is_series(_cols[i]):
                if _ref is None:
                    _ref = _cols[i]
                    _ref_idx = i
            elif pa.types.is_nested(pa.scalar(_cols[i]).type):
                _nested.append(i)
        if _ref is None:
            assert_or_throw(
                reference is not None,
                SlideInvalidOperation(
                    "reference can't be null when all cols are scalars"
                ),
            )
            _cols[0] = self.scalar_to_series(_cols[0], reference=reference)
            _ref = _cols[0]
            _ref_idx = 0
        for n in _nested:
            if not self.is_series(_cols[n]):
                _cols[n] = self.scalar_to_series(_cols[n], reference=_ref)
        if names is None:
            col_names: List[str] = [c.name for c in cols]
        else:
            col_names = names
        tdf = _ref.to_frame(col_names[_ref_idx])
        for j in range(len(_cols)):
            if _ref_idx != j:
                tdf[col_names[j]] = _cols[j]
        return tdf[col_names]

    def is_compatile_index(self, df: dd.DataFrame) -> bool:
        """Check whether the datafame is compatible with the operations inside
        this utils collection
        :param df: dask dataframe
        :return: if it is compatible
        """
        return (
            isinstance(
                df.index,
                (pd.RangeIndex, pd.Int64Index, pd.UInt64Index, dd.Index),
            )
            or self.empty(df)
        )

    def sql_groupby_apply(
        self,
        df: dd.DataFrame,
        cols: List[str],
        func: Callable[[dd.DataFrame], dd.DataFrame],
        output_schema: Optional[pa.Schema] = None,
        **kwargs: Any,
    ) -> dd.DataFrame:
        assert_or_throw(
            output_schema is not None, ValueError("for Dask, output_schema is required")
        )
        meta = to_pandas_dtype(output_schema, use_extension_types=True)
        self.ensure_compatible(df)
        if len(cols) == 0:
            return df.map_partitions(func, meta=meta).reset_index(drop=True)
        return (
            df.groupby(cols, dropna=False, group_keys=False)
            .apply(lambda tdf: func(tdf.reset_index(drop=True)), meta=meta, **kwargs)
            .reset_index(drop=True)
        )

    def as_pandas(self, df: dd.DataFrame) -> pd.DataFrame:
        return df.compute()

    def filter_df(self, df: dd.DataFrame, cond: Any) -> dd.DataFrame:
        c = self._safe_bool(cond)
        if self.is_series(c):
            return df[c]
        elif c:
            return df
        else:
            return dd.from_pandas(df.head(0), npartitions=2)

    def _cast_to_datetime(
        self,
        col: dd.Series,
        from_type: pa.DataType,
        inf_type: pa.DataType,
        safe_dtype: np.dtype,
    ) -> dd.Series:
        return dd.to_datetime(col)

    def _cast_to_float(
        self,
        col: dd.Series,
        from_type: pa.DataType,
        inf_type: pa.DataType,
        safe_dtype: np.dtype,
    ) -> dd.Series:
        if pd.__version__ < "1.2":  # pragma: no cover
            if pa.types.is_string(inf_type):
                return col.fillna("nan").astype(safe_dtype)
        return super()._cast_to_float(
            col=col, from_type=from_type, inf_type=inf_type, safe_dtype=safe_dtype
        )
