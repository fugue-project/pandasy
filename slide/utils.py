from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    Iterable,
    List,
    Optional,
    Tuple,
    TypeVar,
)

import numpy as np
import pandas as pd
import pyarrow as pa
from slide.exceptions import SlideCastError
from triad.utils.assertion import assert_or_throw
from triad.utils.pyarrow import (
    TRIAD_DEFAULT_TIMESTAMP,
    apply_schema,
    to_pa_datatype,
    to_pandas_dtype,
)

TDf = TypeVar("TDf", bound=Any)
TCol = TypeVar("TCol", bound=Any)
_ANTI_INDICATOR = "__anti_indicator__"
_CROSS_INDICATOR = "__corss_indicator__"


def parse_join_type(join_type: str) -> str:
    """Parse and normalize join type string. The normalization
    will lower the string, remove all space and ``_``, and then
    map to the limited options.

    Here are the options after normalization: ``inner``, ``cross``,
    ``left_semi``, ``left_anti``, ``left_outer``, ``right_outer``,
    ``full_outer``.

    :param join_type: the raw join type string
    :raises NotImplementedError: if not supported
    :return: the normalized join type string
    """
    join_type = join_type.replace(" ", "").replace("_", "").lower()
    if join_type in ["inner", "cross"]:
        return join_type
    if join_type in ["inner", "join"]:
        return "inner"
    if join_type in ["semi", "leftsemi"]:
        return "left_semi"
    if join_type in ["anti", "leftanti"]:
        return "left_anti"
    if join_type in ["left", "leftouter"]:
        return "left_outer"
    if join_type in ["right", "rightouter"]:
        return "right_outer"
    if join_type in ["outer", "full", "fullouter"]:
        return "full_outer"
    raise NotImplementedError(join_type)


class SlideUtils(Generic[TDf, TCol]):
    """A collection of utils for general pandas like dataframes"""

    def is_series(self, obj: Any) -> bool:  # pragma: no cover
        """Check whether is a series type

        :param obj: the object
        :return: whether it is a series
        """
        raise NotImplementedError

    def to_series(
        self, obj: Any, name: Optional[str] = None
    ) -> TCol:  # pragma: no cover
        """Convert an object to series

        :param obj: the object
        :param name: name of the series, defaults to None
        :return: the series
        """
        raise NotImplementedError

    def to_constant_series(
        self,
        constant: Any,
        from_series: TCol,
        dtype: Any = None,
        name: Optional[str] = None,
    ) -> TCol:  # pragma: no cover
        """Convert a constant to a series with the same index of ``from_series``

        :param constant: the constant
        :param from_series: the reference series for index
        :param dtype: default data type, defaults to None
        :param name: name of the series, defaults to None
        :return: the series
        """
        raise NotImplementedError

    def get_col_pa_type(self, col: Any) -> pa.DataType:
        """Get column or constant pyarrow data type

        :param col: the column or the constant
        :return: pyarrow data type
        """
        if self.is_series(col):
            tp = col.dtype
            if tp == np.dtype("object") or tp == np.dtype(str):
                return pa.string()
            return pa.from_numpy_dtype(tp)
        return to_pa_datatype(type(col))

    def unary_arithmetic_op(self, col: Any, op: str) -> Any:
        """Unary arithmetic operator on series/constants

        :param col: a series or a constant
        :param op: can be ``+`` or ``-``
        :return: the transformed series or constant
        :raises NotImplementedError: if ``op`` is not supported

        .. note:

        All behaviors should be consistent with SQL correspondent operations.
        """
        if op == "+":
            return col
        if op == "-":
            return 0 - col
        raise NotImplementedError(f"{op} is not supported")  # pragma: no cover

    def binary_arithmetic_op(self, col1: Any, col2: Any, op: str) -> Any:
        """Binary arithmetic operations ``+``, ``-``, ``*``, ``/``

        :param col1: the first column (series or constant)
        :param col2: the second column (series or constant)
        :param op: ``+``, ``-``, ``*``, ``/``
        :return: the result after the operation (series or constant)
        :raises NotImplementedError: if ``op`` is not supported

        .. note:

        All behaviors should be consistent with SQL correspondent operations.
        """
        if op == "+":
            return col1 + col2
        if op == "-":
            return col1 - col2
        if op == "*":
            return col1 * col2
        if op == "/":
            return col1 / col2
        raise NotImplementedError(f"{op} is not supported")  # pragma: no cover

    def comparison_op(self, col1: Any, col2: Any, op: str) -> Any:
        """Binary comparison ``<``, ``<=``, ``==``, ``>``, ``>=``

        :param col1: the first column (series or constant)
        :param col2: the second column (series or constant)
        :param op: ``<``, ``<=``, ``==``, ``>``, ``>=``
        :return: the result after the operation (series or constant)
        :raises NotImplementedError: if ``op`` is not supported

        .. note:

        All behaviors should be consistent with SQL correspondent operations.
        """

        if col1 is None and col2 is None:
            return None
        if op == "==":
            s: Any = col1 == col2
        elif op == "!=":
            s = col1 != col2
        elif op == "<":
            s = col1 < col2
        elif op == "<=":
            s = col1 <= col2
        elif op == ">":
            s = col1 > col2
        elif op == ">=":
            s = col1 >= col2
        else:  # pragma: no cover
            raise NotImplementedError(f"{op} is not supported")
        return self._set_op_result_to_none(s, col1, col2)

    def binary_logical_op(self, col1: Any, col2: Any, op: str) -> Any:
        """Binary logical operations ``and``, ``or``

        :param col1: the first column (series or constant)
        :param col2: the second column (series or constant)
        :param op: ``and``, ``or``
        :return: the result after the operation (series or constant)
        :raises NotImplementedError: if ``op`` is not supported

        .. note:

        All behaviors should be consistent with SQL correspondent operations.
        """
        c1 = self._safe_bool(col1)
        c2 = self._safe_bool(col2)
        if op == "and":
            s: Any = c1 * c2
            # in sql, FALSE AND anything is False
            if self.is_series(s):
                s = s.mask((c1 == 0) | (c2 == 0), 0)
            elif (c1 == 0) | (c2 == 0):
                s = 0.0
        elif op == "or":
            s = c1 + c2
            # in sql, True OR anything is True
            if self.is_series(s):
                s = s.mask((c1 > 0) | (c2 > 0), 1)
            elif (c1 > 0) | (c2 > 0):
                s = 1.0
        else:  # pragma: no cover
            raise NotImplementedError(f"{op} is not supported")
        return s

    def logical_not(self, col: Any) -> Any:
        """Logical ``NOT``

        .. note:

        All behaviors should be consistent with SQL correspondent operations.
        """
        s = self._safe_bool(col)
        if self.is_series(s):
            nulls = s.isnull()
            s = s == 0
            s = s.mask(nulls, None)
            return s
        return 1.0 - s

    def cast(  # noqa: C901
        self, col: Any, type_obj: Any, input_type: Any = None
    ) -> Any:
        """Cast ``col`` to a new type. ``type_obj`` must be
        able to be converted by :func:`~triad.utils.pyarrow.to_pa_datatype`.

        :param col: a series or a constant
        :param type_obj: an objected that can be accepted by
            :func:`~triad.utils.pyarrow.to_pa_datatype`
        :param input_type: an objected that is either None or to be accepted by
            :func:`~triad.utils.pyarrow.to_pa_datatype`, defaults to None.
        :return: the new column or constant

        .. note:

        If ``input_type`` is not None, then it can be used to determine
        the casting behavior. This can be useful when the input is boolean with
        nulls or strings, where the pandas dtype may not provide the accurate
        type information.
        """
        to_type = to_pa_datatype(type_obj)
        p_type = str if pa.types.is_string(to_type) else to_type.to_pandas_dtype()
        if self.is_series(col):
            try:
                try:
                    from_type = (
                        self.get_col_pa_type(col)
                        if input_type is None
                        else to_pa_datatype(input_type)
                    )
                    if from_type == to_type:
                        return col
                except Exception:  # pragma: no cover
                    return col.astype(p_type)
                if pa.types.is_boolean(from_type):
                    if pa.types.is_string(to_type):
                        nulls = col.isnull()
                        neg = col == 0
                        pos = col != 0
                        res = col.astype(str)
                        return (
                            res.mask(neg, "false").mask(pos, "true").mask(nulls, None)
                        )
                    if pa.types.is_integer(to_type) or pa.types.is_floating(to_type):
                        nulls = col.isnull()
                        return (col.fillna(0) != 0).astype(p_type).mask(nulls, None)
                if pa.types.is_integer(from_type):
                    if pa.types.is_boolean(to_type):
                        return col != 0
                    if pa.types.is_string(to_type):
                        temp = col.astype(np.float64)
                        nulls = temp.isnull()
                        return (
                            temp.fillna(0)
                            .astype(np.int64)
                            .astype(p_type)
                            .mask(nulls, None)
                        )
                elif pa.types.is_floating(from_type):
                    if pa.types.is_boolean(to_type):
                        nulls = col.isnull()
                        return (col != 0).mask(nulls, None)
                    if pa.types.is_integer(to_type):
                        nulls = col.isnull()
                        return col.fillna(0).astype(p_type).mask(nulls, None)
                    if pa.types.is_string(to_type):
                        nulls = col.isnull()
                        return col.fillna(0).astype(p_type).mask(nulls, None)
                elif pa.types.is_string(from_type):
                    if pa.types.is_boolean(to_type):
                        lower = col.str.lower()
                        res = lower.isin(["true", "1", "1.0"])
                        nulls = (~res) & (~lower.isin(["false", "0", "0.0"]))
                        return res.mask(nulls, None)
                    if pa.types.is_integer(to_type):
                        temp = col.astype(np.float64)
                        nulls = temp.isnull()
                        return temp.fillna(0).astype(p_type).mask(nulls, None)
                elif pa.types.is_timestamp(from_type) or pa.types.is_date(from_type):
                    if pa.types.is_string(to_type):
                        nulls = col.isnull()
                        return col.astype(p_type).mask(nulls, None)
                return col.astype(p_type)
            except (TypeError, ValueError) as te:
                raise SlideCastError(f"failed to cast to {p_type}") from te
        else:
            if col is None:
                return None
            res = self.cast(
                self.to_series([col]),
                type_obj=type_obj,
                input_type=self.get_col_pa_type(col)
                if input_type is None
                else input_type,
            ).iloc[0]
            if pd.isna(res):
                return None
            return res

    def filter_df(self, df: TDf, cond: Any) -> TDf:
        """Filter dataframe by a boolean series or a constant

        :param df: the dataframe
        :param cond: a boolean seris or a constant
        :return: the filtered dataframe

        .. note:

        Filtering behavior should be consistent with SQL.
        """
        c = self._safe_bool(cond)
        if self.is_series(c):
            return df[c > 0]
        elif c > 0:
            return df
        else:
            return df.head(0)

    def is_value(self, col: Any, value: Any, positive: bool = True) -> Any:
        """Check if the series or constant is ``value``

        :param col: the series or constant
        :param value: ``None``, ``True`` or ``False``
        :param positive: check ``is value`` or ``is not value``,
            defaults to True (``is value``)
        :raises NotImplementedError: if value is not supported
        :return: a bool value or a series
        """
        if self.is_series(col):
            if value is None:
                if positive:
                    return col.isnull()
                else:
                    return ~col.isnull()
            elif isinstance(value, bool) and value:
                if positive:
                    return (col != 0) & (~col.isnull())
                else:
                    return (col == 0) | col.isnull()
            elif isinstance(value, bool) and not value:
                if positive:
                    return (col == 0) & (~col.isnull())
                else:
                    return (col != 0) | col.isnull()
            raise NotImplementedError(value)
        else:
            return self.is_value(
                self.to_series([col]), value=value, positive=positive
            ).iloc[0]

    def is_in(self, col: Any, values: List[Any], positive: bool) -> Any:  # noqa: C901
        """Check if a series or a constant is in ``values``

        :param col: the series or the constant
        :param values: a list of constants and series (can mix)
        :param positive: ``is in`` or ``is not in``
        :return: the correspondent boolean series or constant

        .. note:

        This behavior should be consistent with SQL ``IN`` and ``NOT IN``.
        The return values can be ``True``, ``False`` and ``None``
        """
        if self.is_series(col):
            cols = [x for x in values if self.is_series(x)]
            others = [x for x in values if not self.is_series(x)]
            has_null_constant = any(pd.isna(x) for x in others)
            innulls: Any = None
            if positive:
                o: Any = col.isin(others)
                for c in cols:
                    o = o | (col == c)
                    if not has_null_constant:
                        if innulls is None:
                            innulls = c.isnull()
                        else:
                            innulls = innulls | c.isnull()
            else:
                o = ~col.isin(others)
                for c in cols:
                    o = o & (col != c)
                    if not has_null_constant:
                        if innulls is None:
                            innulls = c.isnull()
                        else:
                            innulls = innulls | c.isnull()
            if has_null_constant:
                o = o.mask(o == (0 if positive else 1), None)
            elif innulls is not None:
                o = o.mask(innulls & (o == (0 if positive else 1)), None)
            return o.mask(col.isnull(), None)
        else:
            res = self.is_in(
                self.to_series([col]), values=values, positive=positive
            ).iloc[0]
            return None if pd.isna(res) else bool(res)

    def is_between(self, col: Any, lower: Any, upper: Any, positive: bool) -> Any:
        """Check if a series or a constant is ``>=lower`` and ``<=upper``

        :param col: the series or the constant
        :param lower: the lower bound, which can be series or a constant
        :param upper: the upper bound, which can be series or a constant
        :param positive: ``is between`` or ``is not between``
        :return: the correspondent boolean series or constant

        .. note:

        This behavior should be consistent with SQL ``BETWEEN`` and ``NOT BETWEEN``.
        The return values can be ``True``, ``False`` and ``None``
        """
        if col is None:
            return None
        if self.is_series(col):
            left = (lower <= col).fillna(False)
            right = (col <= upper).fillna(False)
            ln = lower.isnull() if self.is_series(lower) else lower is None
            un = upper.isnull() if self.is_series(upper) else upper is None
            s: Any = left & right
            s = s.mask(col.isnull() | (ln & un), None)
            if self.is_series(lower) or lower is None:
                s = s.mask(right & ln, None)
            if self.is_series(upper) or upper is None:
                s = s.mask(left & un, None)

            if positive:
                return s
            nulls = s.isnull()
            return (~(s.fillna(False))).mask(nulls, None)
        else:
            res = self.is_between(
                self.to_series([col]), lower=lower, upper=upper, positive=positive
            ).iloc[0]
            return None if pd.isna(res) else bool(res)

    def coalesce(self, cols: List[Any]) -> Any:
        """Coalesce multiple series and constants

        :param cols: the collection of series and constants in order
        :return: the coalesced series or constant

        .. note:

        This behavior should be consistent with SQL ``COALESCE``
        """
        if any(self.is_series(s) for s in cols):
            tmp = self.cols_to_df(cols, [f"_{n}" for n in range(len(cols))])
            return tmp.fillna(method="bfill", axis=1)["_0"]
        for x in cols:
            if x is not None:
                return x
        return None

    def cols_to_df(self, cols: List[Any], names: Optional[List[str]] = None) -> TDf:
        """Construct the dataframe from a list of columns (series)

        :param cols: the collection of series or constants, at least one value must
            be a series
        :param names: the correspondent column names, defaults to None

        :return: the dataframe

        .. note::

        If ``names`` is not provided, then every series in ``cols`` must be
        named. Otherise, ``names`` must align with ``cols``. But whether names
        have duplications or invalid chars will not be verified by this method
        """
        raise NotImplementedError  # pragma: no cover

    def empty(self, df: TDf) -> bool:
        """Check if the dataframe is empty

        :param df: pandas like dataframe
        :return: if it is empty
        """
        return len(df.index) == 0

    def as_arrow(self, df: TDf, schema: Optional[pa.Schema] = None) -> pa.Table:
        """Convert pandas like dataframe to pyarrow table

        :param df: pandas like dataframe
        :param schema: if specified, it will be used to construct pyarrow table,
          defaults to None
        :return: pyarrow table
        """
        return pa.Table.from_pandas(df, schema=schema, preserve_index=False, safe=False)

    def as_array_iterable(
        self,
        df: TDf,
        schema: Optional[pa.Schema] = None,
        columns: Optional[List[str]] = None,
        type_safe: bool = False,
    ) -> Iterable[List[Any]]:
        """Convert pandas like dataframe to iterable of rows in the format of list.

        :param df: pandas like dataframe
        :param schema: schema of the input. With None, it will infer the schema,
          it can't infer wrong schema for nested types, so try to be explicit
        :param columns: columns to output, None for all columns
        :param type_safe: whether to enforce the types in schema, if False, it will
            return the original values from the dataframe
        :return: iterable of rows, each row is a list

        .. note::

        If there are nested types in schema, the conversion can be slower
        """
        if self.empty(df):
            return
        if schema is None:
            schema = self.to_schema(df)
        if columns is not None:
            df = df[columns]
            schema = pa.schema([schema.field(n) for n in columns])
        if not type_safe:
            for arr in df.itertuples(index=False, name=None):
                yield list(arr)
        elif all(not pa.types.is_nested(x) for x in schema.types):
            p = self.as_arrow(df, schema)
            d = p.to_pydict()
            cols = [d[n] for n in schema.names]
            for arr in zip(*cols):
                yield list(arr)
        else:
            # If schema has nested types, the conversion will be much slower
            for arr in apply_schema(
                schema,
                df.itertuples(index=False, name=None),
                copy=True,
                deep=True,
                str_as_json=True,
            ):
                yield arr

    def as_array(
        self,
        df: TDf,
        schema: Optional[pa.Schema] = None,
        columns: Optional[List[str]] = None,
        type_safe: bool = False,
    ) -> List[List[Any]]:
        return list(
            self.as_array_iterable(
                df, schema=schema, columns=columns, type_safe=type_safe
            )
        )

    def to_schema(self, df: TDf) -> pa.Schema:
        """Extract pandas dataframe schema as pyarrow schema. This is a replacement
        of pyarrow.Schema.from_pandas, and it can correctly handle string type and
        empty dataframes

        :param df: pandas dataframe
        :raises ValueError: if pandas dataframe does not have named schema
        :return: pyarrow.Schema

        .. note::

        The dataframe must be either empty, or with type pd.RangeIndex, pd.Int64Index
        or pd.UInt64Index and without a name, otherwise, `ValueError` will raise.
        """
        self.ensure_compatible(df)
        assert_or_throw(
            df.columns.dtype == "object",
            ValueError("Pandas dataframe must have named schema"),
        )

        def get_fields() -> Iterable[pa.Field]:
            if isinstance(df, pd.DataFrame) and len(df.index) > 0:
                yield from pa.Schema.from_pandas(df, preserve_index=False)
            else:
                for i in range(df.shape[1]):
                    tp = df.dtypes[i]
                    if tp == np.dtype("object") or tp == np.dtype(str):
                        t = pa.string()
                    else:
                        t = pa.from_numpy_dtype(tp)
                    yield pa.field(df.columns[i], t)

        fields: List[pa.Field] = []
        for field in get_fields():
            if pa.types.is_timestamp(field.type):
                fields.append(pa.field(field.name, TRIAD_DEFAULT_TIMESTAMP))
            else:
                fields.append(field)
        return pa.schema(fields)

    def enforce_type(  # noqa: C901
        self, df: TDf, schema: pa.Schema, null_safe: bool = False
    ) -> TDf:
        """Enforce the pandas like dataframe to comply with `schema`.

        :param df: pandas like dataframe
        :param schema: pyarrow schema
        :param null_safe: whether to enforce None value for int, string and bool values
        :return: converted dataframe

        .. note::

        When `null_safe` is true, the native column types in the dataframe may change,
        for example, if a column of `int64` has None values, the output will make sure
        each value in the column is either None or an integer, however, due to the
        behavior of pandas like dataframes, the type of the columns may
        no longer be `int64`. This method does not enforce struct and list types
        """
        if self.empty(df):
            return df
        if not null_safe:
            return df.astype(dtype=to_pandas_dtype(schema))
        cols: List[TCol] = []
        for v in schema:
            s = df[v.name]
            if pa.types.is_string(v.type):
                ns = s.isnull()
                s = s.astype(str).mask(ns, None)
            elif pa.types.is_boolean(v.type):
                ns = s.isnull()
                if pd.api.types.is_string_dtype(s.dtype):
                    try:
                        s = s.str.lower() == "true"
                    except AttributeError:
                        s = s.fillna(0).astype(bool)
                else:
                    s = s.fillna(0).astype(bool)
                s = s.mask(ns, None)
            elif pa.types.is_integer(v.type):
                ns = s.isnull()
                s = s.fillna(0).astype(v.type.to_pandas_dtype()).mask(ns, None)
            elif not pa.types.is_struct(v.type) and not pa.types.is_list(v.type):
                s = s.astype(v.type.to_pandas_dtype())
            cols.append(s)
        return self.cols_to_df(cols)

    def sql_groupby_apply(
        self,
        df: TDf,
        cols: List[str],
        func: Callable[[TDf], TDf],
        **kwargs: Any,
    ) -> TDf:
        """Safe groupby apply operation on pandas like dataframes.
        In pandas like groupby apply, if any key is null, the whole group is dropped.
        This method makes sure those groups are included.

        :param df: pandas like dataframe
        :param cols: columns to group on, can be empty
        :param func: apply function, df in, df out
        :return: output dataframe

        .. note::

        The dataframe must be either empty, or with type pd.RangeIndex, pd.Int64Index
        or pd.UInt64Index and without a name, otherwise, `ValueError` will raise.
        """
        raise NotImplementedError  # pragma: no cover

    def is_compatile_index(self, df: TDf) -> bool:
        """Check whether the datafame is compatible with the operations inside
        this utils collection

        :param df: pandas like dataframe
        :return: if it is compatible
        """
        return isinstance(df.index, (pd.RangeIndex, pd.Int64Index, pd.UInt64Index))

    def ensure_compatible(self, df: TDf) -> None:
        """Check whether the datafame is compatible with the operations inside
        this utils collection, if not, it will raise ValueError

        :param df: pandas like dataframe
        :raises ValueError: if not compatible
        """
        if df.index.name is not None:
            raise ValueError("pandas like datafame index can't have name")
        if self.is_compatile_index(df):
            return
        if self.empty(df):
            return
        raise ValueError(
            f"pandas like datafame must have default index, but got {type(df.index)}"
        )

    def drop_duplicates(self, df: TDf) -> TDf:
        """Select distinct rows from dataframe

        :param df: the dataframe
        :return: the result with only distinct rows
        """
        return df.drop_duplicates(ignore_index=True)

    def union(self, df1: TDf, df2: TDf, unique: bool) -> TDf:
        """Union two dataframes

        :param df1: the first dataframe
        :param df2: the second dataframe
        :param unique: whether return only unique rows
        :return: unioned dataframe
        """
        ndf1, ndf2 = self._preprocess_set_op(df1, df2)
        ndf = ndf1.append(ndf2, ignore_index=True)
        if unique:
            ndf = self.drop_duplicates(ndf)
        return ndf

    def intersect(self, df1: TDf, df2: TDf, unique: bool) -> TDf:
        """Intersect two dataframes

        :param ndf1: the first dataframe
        :param ndf2: the second dataframe
        :param unique: whether return only unique rows
        :return: intersected dataframe
        """
        ndf1, ndf2 = self._preprocess_set_op(df1, df2)
        ndf = ndf1.merge(self.drop_duplicates(ndf2))
        if unique:
            ndf = self.drop_duplicates(ndf)
        return ndf

    def except_df(
        self,
        df1: TDf,
        df2: TDf,
        unique: bool,
        anti_indicator_col: str = _ANTI_INDICATOR,
    ) -> TDf:
        """Exclude df2 from df1

        :param df1: the first dataframe
        :param df2: the second dataframe
        :param unique: whether return only unique rows
        :return: df1 - df2

        .. note:

        The behavior is not well defined when unique is False
        """
        ndf1, ndf2 = self._preprocess_set_op(df1, df2)
        ndf2 = self._with_indicator(self.drop_duplicates(ndf2), anti_indicator_col)
        ndf = ndf1.merge(ndf2, how="left", on=list(ndf1.columns))
        ndf = ndf[ndf[anti_indicator_col].isnull()].drop([anti_indicator_col], axis=1)
        if unique:
            ndf = self.drop_duplicates(ndf)
        return ndf

    def join(
        self,
        ndf1: TDf,
        ndf2: TDf,
        join_type: str,
        on: List[str],
        anti_indicator_col: str = _ANTI_INDICATOR,
        cross_indicator_col: str = _CROSS_INDICATOR,
    ) -> TDf:
        """Join two dataframes.

        :param ndf1: the first dataframe
        :param ndf2: the second dataframe
        :param join_type: see :func:`~.parse_join_type`
        :param on: join keys for pandas like ``merge`` to use
        :param anti_indicator_col: temporary column name for anti join,
            defaults to _ANTI_INDICATOR
        :param cross_indicator_col: temporary column name for cross join,
            defaults to _CROSS_INDICATOR
        :raises NotImplementedError: if join type is not supported
        :return: the joined dataframe

        .. note:

        All join behaviors should be consistent with SQL correspondent joins.
        """
        join_type = parse_join_type(join_type)
        if join_type == "inner":
            ndf1 = ndf1.dropna(subset=on)
            ndf2 = ndf2.dropna(subset=on)
            joined = ndf1.merge(ndf2, how=join_type, on=on)
        elif join_type == "left_semi":
            ndf1 = ndf1.dropna(subset=on)
            ndf2 = self.drop_duplicates(ndf2[on].dropna())
            joined = ndf1.merge(ndf2, how="inner", on=on)
        elif join_type == "left_anti":
            # TODO: lack of test to make sure original ndf2 is not changed
            ndf2 = self.drop_duplicates(ndf2[on].dropna())
            ndf2 = self._with_indicator(ndf2, anti_indicator_col)
            joined = ndf1.merge(ndf2, how="left", on=on)
            joined = joined[joined[anti_indicator_col].isnull()].drop(
                [anti_indicator_col], axis=1
            )
        elif join_type == "left_outer":
            ndf2 = ndf2.dropna(subset=on)
            joined = ndf1.merge(ndf2, how="left", on=on)
        elif join_type == "right_outer":
            ndf1 = ndf1.dropna(subset=on)
            joined = ndf1.merge(ndf2, how="right", on=on)
        elif join_type == "full_outer":
            add: List[str] = []
            add_df1: Dict[str, TCol] = {}
            add_df2: Dict[str, TCol] = {}
            for f in on:
                name = f + "_null"
                s1 = ndf1[f].isnull().astype(int)
                add_df1[name] = s1
                s2 = ndf2[f].isnull().astype(int) * 2
                add_df2[name] = s2
                add.append(name)
            joined = (
                ndf1.assign(**add_df1)
                .merge(ndf2.assign(**add_df2), how="outer", on=add + on)
                .drop(add, axis=1)
            )
        elif join_type == "cross":
            assert_or_throw(
                len(on) == 0, ValueError(f"cross join can't have join keys {on}")
            )
            ndf1 = self._with_indicator(ndf1, cross_indicator_col)
            ndf2 = self._with_indicator(ndf2, cross_indicator_col)
            joined = ndf1.merge(ndf2, how="inner", on=[cross_indicator_col]).drop(
                [cross_indicator_col], axis=1
            )
        else:  # pragma: no cover
            raise NotImplementedError(join_type)
        return joined

    def _set_op_result_to_none(self, series: Any, s1: Any, s2: Any) -> Any:
        if not self.is_series(series):
            return series
        if self.is_series(s1):
            series = series.mask(s1.isnull(), None)
        if self.is_series(s2):
            series = series.mask(s2.isnull(), None)
        return series

    def _safe_bool(self, col: Any) -> Any:
        if self.is_series(col):
            return col.astype("f8")
        if col is None:
            return float("nan")
        return float(col > 0)

    def _preprocess_set_op(self, ndf1: TDf, ndf2: TDf) -> Tuple[TDf, TDf]:
        assert_or_throw(
            len(list(ndf1.columns)) == len(list(ndf2.columns)),
            ValueError("two dataframes have different number of columns"),
        )
        cols: List[TCol] = []
        same = True
        for c1, c2 in zip(ndf1.columns, ndf2.columns):
            same &= c1 == c2
            cols.append(ndf2[c2])
        if same:
            return ndf1, ndf2
        return ndf1, self.cols_to_df(cols, list(ndf1.columns))

    def _with_indicator(self, df: TDf, name: str) -> TDf:
        return df.assign(**{name: 1})
