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
from triad.utils.assertion import assert_or_throw
from triad.utils.pyarrow import (
    TRIAD_DEFAULT_TIMESTAMP,
    to_pa_datatype,
    to_single_pandas_dtype,
    _TypeConverter,
)

from slide._string_utils import LikeExpr, LikeExprShortcut
from slide.exceptions import SlideCastError, SlideIndexIncompatibleError

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

    def to_safe_pa_type(self, tp: Any) -> pa.DataType:
        if isinstance(tp, (np.dtype, pd.api.extensions.ExtensionDtype)):
            if pd.api.types.is_datetime64_any_dtype(tp):
                return TRIAD_DEFAULT_TIMESTAMP
            if pd.api.types.is_object_dtype(tp):
                return pa.string()
            if pd.__version__ >= "1.2":
                if pd.Float64Dtype() == tp:
                    return pa.float64()
                if pd.Float32Dtype() == tp:
                    return pa.float32()
        return to_pa_datatype(tp)

    def is_series(self, obj: Any) -> bool:  # pragma: no cover
        """Check whether is a series type

        :param obj: the object
        :return: whether it is a series
        """
        raise NotImplementedError

    def to_series(self, obj: Any, name: Optional[str] = None) -> TCol:
        """Convert an object to series

        :param obj: the object
        :param name: name of the series, defaults to None
        :return: the series
        """
        raise NotImplementedError  # pragma: no cover

    def series_to_array(self, col: TCol) -> List[Any]:
        """Convert a series to numpy array

        :param col: the series
        :return: the numpy array
        """
        raise NotImplementedError  # pragma: no cover

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
            return self.to_safe_pa_type(tp)
        return self.to_safe_pa_type(type(col))

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
            if not self.is_series(c1) and not self.is_series(c2):
                return c1 and c2
            return c1 & c2
        elif op == "or":
            if not self.is_series(c1) and not self.is_series(c2):
                return c1 or c2
            return c1 | c2
        raise NotImplementedError(f"{op} is not supported")  # pragma: no cover

    def logical_not(self, col: Any) -> Any:
        """Logical ``NOT``

        .. note:

        All behaviors should be consistent with SQL correspondent operations.
        """
        b = self._safe_bool(col)
        if self.is_series(b):
            return ~b
        return None if b is None else not b

    def cast(  # noqa: C901
        self, col: Any, type_obj: Any, input_type: Any = None
    ) -> Any:
        """Cast ``col`` to a new type. ``type_obj`` must be
        able to be converted by :func:`~triad.utils.pyarrow.self.to_safe_pa_type`.

        :param col: a series or a constant
        :param type_obj: an objected that can be accepted by
            :func:`~triad.utils.pyarrow.self.to_safe_pa_type`
        :param input_type: an objected that is either None or to be accepted by
            :func:`~triad.utils.pyarrow.self.to_safe_pa_type`, defaults to None.
        :return: the new column or constant

        .. note:

        If ``input_type`` is not None, then it can be used to determine
        the casting behavior. This can be useful when the input is boolean with
        nulls or strings, where the pandas dtype may not provide the accurate
        type information.
        """
        try:
            if self.is_series(col):
                to_type = self.to_safe_pa_type(type_obj)
                input_pa_type = (
                    None if input_type is None else self.to_safe_pa_type(input_type)
                )
                if (  # nested/binary as input/output
                    pa.types.is_nested(to_type)
                    or pa.types.is_binary(to_type)
                    or (
                        input_pa_type is not None
                        and (
                            pa.types.is_nested(input_pa_type)
                            or pa.types.is_binary(input_pa_type)
                        )
                    )
                ):
                    assert_or_throw(
                        pd.api.types.is_object_dtype(col.dtype),
                        SlideCastError(f"unexpected column type {col.dtype}"),
                    )
                    assert_or_throw(
                        input_type is None
                        or self.to_safe_pa_type(input_type) == to_type,
                        SlideCastError(f"unexpected column type hint {input_type}"),
                    )
                    return col

                t_type = to_single_pandas_dtype(to_type, use_extension_types=True)
                inf_type = self.get_col_pa_type(col)
                has_hint = input_type is not None
                from_type = input_pa_type or inf_type

                if pa.types.is_string(to_type):
                    if (
                        has_hint
                        and pa.types.is_string(from_type)
                        and pa.types.is_string(inf_type)
                    ):
                        return col
                elif from_type == inf_type == to_type:
                    return col

                if pa.types.is_boolean(to_type):
                    return self._cast_to_bool(col, from_type, inf_type, t_type)
                if pa.types.is_integer(to_type):
                    return self._cast_to_int(col, from_type, inf_type, t_type)
                elif pa.types.is_floating(to_type):
                    return self._cast_to_float(col, from_type, inf_type, t_type)
                elif pa.types.is_timestamp(to_type):
                    return self._cast_to_datetime(col, from_type, inf_type, t_type)
                elif pa.types.is_date(to_type):
                    return self._cast_to_date(col, from_type, inf_type, t_type)
                elif pa.types.is_string(to_type):
                    return self._cast_to_str(col, from_type, inf_type, t_type)
                return col.astype(t_type)  # pragma: no cover
            else:
                if col is None:
                    return None
                from_type = (
                    self.get_col_pa_type(col) if input_type is None else input_type
                )
                res = self.series_to_array(
                    self.cast(
                        self.to_series([col]),
                        type_obj=type_obj,
                        input_type=from_type,
                    )
                )[0]
                if pd.isna(res):
                    return None
                return res
        except (TypeError, ValueError) as te:
            raise SlideCastError(str(te)) from te

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
            return df[c]
        elif c:
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
            return self.series_to_array(
                self.is_value(self.to_series([col]), value=value, positive=positive)
            )[0]

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
            res = self.series_to_array(
                self.is_in(self.to_series([col]), values=values, positive=positive)
            )[0]
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
            left = (
                self.to_constant_series(False, col)
                if lower is None
                else (lower <= col).fillna(False)
            )
            right = (
                self.to_constant_series(False, col)
                if upper is None
                else (col <= upper).fillna(False)
            )
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
            return (s == 0).mask(s.isnull(), None)
        else:
            res = self.series_to_array(
                self.is_between(
                    self.to_series([col]),
                    lower=lower
                    if lower is None or self.is_series(lower)
                    else self.to_series([lower]),
                    upper=upper
                    if upper is None or self.is_series(upper)
                    else self.to_series([upper]),
                    positive=positive,
                )
            )[0]
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

    def case_when(self, *pairs: Tuple[Any, Any], default: Any = None) -> Any:
        """SQL ``CASE WHEN``

        :param pairs: condition and value pairs, both can be either a
            series or a constant
        :param default: default value if none of the conditions satisfies,
            defaults to None
        :return: the final series or constant

        .. note:

        This behavior should be consistent with SQL ``CASE WHEN``
        """

        def _safe_pos(s: Any) -> Any:
            if self.is_series(s):
                return (~(s.isnull())) & (s != 0)
            return not pd.isna(s) and s != 0

        def get_series() -> Iterable[Tuple[str, Any]]:
            for n in range(len(pairs)):
                yield f"w_{n}", _safe_pos(pairs[n][0])
                yield f"t_{n}", pairs[n][1]
            yield "d", default

        all_series = list(get_series())
        if any(self.is_series(x[1]) for x in all_series):
            tmp = self.cols_to_df(
                [x[1] for x in all_series], names=[x[0] for x in all_series]
            )
            res = tmp["d"]
            for n in reversed(range(len(pairs))):
                if pairs[n][1] is None:
                    res = res.mask(tmp[f"w_{n}"], pd.NA)
                else:
                    res = res.mask(tmp[f"w_{n}"], tmp[f"t_{n}"])
            return res
        sd = {x[0]: x[1] for x in all_series}
        for n in range(len(pairs)):
            if sd[f"w_{n}"] == 1.0:
                return sd[f"t_{n}"]
        return sd["d"]

    def like(  # noqa: C901
        self, col: Any, expr: Any, ignore_case: bool = False, positive: bool = True
    ) -> Any:
        """SQL ``LIKE``

        :param col: a series or a constant
        :param expr: a pattern expression
        :param ignore_case: whether to ignore case, defaults to False
        :param positive: ``LIKE`` or ``NOT LIKE``, defaults to True
        :return: the correspondent boolean series or constant

        .. note:

        This behavior should be consistent with SQL ``LIKE``
        """
        assert_or_throw(
            expr is None or isinstance(expr, str),
            NotImplementedError("expr can only be a string"),
        )

        def like_series(col: TCol) -> TCol:
            le = LikeExpr(expr)
            if le.shortcut == LikeExprShortcut.EMPTY:
                return col == ""
            if le.shortcut == LikeExprShortcut.NOT_EMPTY:
                return col != ""
            if le.shortcut == LikeExprShortcut.SIMPLE:
                if not ignore_case:
                    return col == le.tokens[0][1]
                else:
                    return col.str.lower() == le.tokens[0][1].lower()
            if le.shortcut == LikeExprShortcut.ANY:
                return ~(col.isnull())
            if le.shortcut == LikeExprShortcut.START:
                if not ignore_case:
                    return col.str.startswith(le.tokens[0][1])
                return col.str.lower().str.startswith(le.tokens[0][1].lower())
            if le.shortcut == LikeExprShortcut.END:
                if not ignore_case:
                    return col.str.endswith(le.tokens[1][1]).mask(nulls, None)
                return col.str.lower().str.endswith(le.tokens[1][1].lower())
            if le.shortcut == LikeExprShortcut.START_END:
                if not ignore_case:
                    return col.str.startswith(le.tokens[0][1]) & col.str.endswith(
                        le.tokens[2][1]
                    )
                return col.str.lower().str.startswith(
                    le.tokens[0][1].lower()
                ) & col.str.lower().str.endswith(le.tokens[2][1].lower())
            if le.shortcut == LikeExprShortcut.CONTAIN:
                if not ignore_case:
                    return col.str.contains(le.tokens[1][1])
                return col.str.lower().str.contains(le.tokens[1][1].lower())
            if le.shortcut == LikeExprShortcut.NA:
                return col.str.match(le.re(), case=not ignore_case)
            raise NotImplementedError(le.shortcut)  # pragma: no cover

        if self.is_series(col):
            if expr is None:
                return self.to_constant_series(float("nan"), col)
            nulls = col.isnull()
            res = like_series(col)
            if positive:
                return res.mask(nulls, None)
            return (res == 0).mask(nulls, None)
        else:
            res = self.series_to_array(
                self.like(self.to_series([col]), expr=expr, ignore_case=ignore_case)
            )[0]
            return None if pd.isna(res) else bool(res)

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

    def as_arrow(self, df: TDf, schema: pa.Schema, type_safe: bool = True) -> pa.Table:
        """Convert the dataframe to pyarrow table

        :param df: pandas like dataframe
        :param schema: if specified, it will be used to construct pyarrow table,
          defaults to None
        :param type_safe: check for overflows or other unsafe conversions
        :return: pyarrow table
        """
        pdf = self.as_pandas(df)
        return pa.Table.from_pandas(
            pdf, schema=schema, preserve_index=False, safe=type_safe
        )

    def as_pandas(self, df: TDf) -> pd.DataFrame:
        """Convert the dataframe to pandas dataframe

        :return: the pandas dataframe
        """
        raise NotImplementedError  # pragma: no cover

    def create_native_converter(
        self,
        input_schema: pa.Schema,
        type_safe: bool,
    ) -> "SlideDataFrameNativeConverter":
        """Create a converter that convert the dataframes into python native iterables

        :param input_schema: schema of the input dataframe
        :param type_safe: whether to enforce the types in schema, if False, it will
            return the original values from the dataframes
        :return: the converter

        .. tip::

        This converter can be reused on multiple dataframes with the same structure
        """
        return SlideDataFrameNativeConverter(self, input_schema, type_safe)

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
            ValueError("Dataframe must have named schema"),
        )

        def get_fields() -> Iterable[pa.Field]:
            for c in df.columns:
                tp = df[c].dtype
                if tp == np.dtype("object") or tp == np.dtype(str):
                    t = pa.string()
                else:
                    t = self.to_safe_pa_type(tp)
                    if pa.types.is_timestamp(t):
                        t = TRIAD_DEFAULT_TIMESTAMP
                yield pa.field(c, t)

        return pa.schema(list(get_fields()))

    def cast_df(  # noqa: C901
        self, df: TDf, schema: pa.Schema, input_schema: Optional[pa.Schema] = None
    ) -> TDf:
        """Cast a dataframe to comply with `schema`.

        :param df: pandas like dataframe
        :param schema: pyarrow schema to convert to
        :param input_schema: the known input pyarrow schema, defaults to None
        :return: converted dataframe

        .. note::

        ``input_schema`` is important because sometimes the column types can be
        different from expected. For example if a boolean series contains Nones,
        the dtype will be object, without a input type hint, the function can't
        do the conversion correctly.
        """
        if input_schema is None:
            cols = [self.cast(df[v.name], v.type) for v in schema]
        else:
            cols = [
                self.cast(df[v.name], v.type, input_type=i.type)
                for v, i in zip(schema, input_schema)
            ]
        return self.cols_to_df(cols)

    def sql_groupby_apply(
        self,
        df: TDf,
        cols: List[str],
        func: Callable[[TDf], TDf],
        output_schema: Optional[pa.Schema] = None,
        **kwargs: Any,
    ) -> TDf:
        """Safe groupby apply operation on pandas like dataframes.
        In pandas like groupby apply, if any key is null, the whole group is dropped.
        This method makes sure those groups are included.

        :param df: pandas like dataframe
        :param cols: columns to group on, can be empty
        :param func: apply function, df in, df out
        :param output_schema: output schema hint for the apply
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
            raise SlideIndexIncompatibleError(
                "pandas like datafame index can't have name"
            )
        if self.is_compatile_index(df):
            return
        if self.empty(df):
            return
        raise SlideIndexIncompatibleError(
            f"pandas like datafame must have default index, but got {type(df.index)}"
        )

    def drop_duplicates(self, df: TDf) -> TDf:
        """Select distinct rows from dataframe

            raise SlideIndexIncompatibleError(
                "pandas like datafame index can't have name"
            )
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
        ndf = ndf1.append(ndf2)
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
            return col.astype("boolean")
        if col is None:
            return None
        return col != 0

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

    def _cast_to_bool(
        self,
        col: TCol,
        from_type: pa.DataType,
        inf_type: pa.DataType,
        safe_dtype: np.dtype,
    ) -> TCol:
        if pa.types.is_boolean(from_type):
            if (
                pa.types.is_integer(inf_type)
                or pa.types.is_floating(inf_type)
                or pa.types.is_string(inf_type)  # bool/int with nulls
            ):
                nulls = col.isnull()
                return (col != 0).mask(nulls, pd.NA).astype(safe_dtype)
        elif pa.types.is_integer(from_type) or pa.types.is_floating(from_type):
            nulls = col.isnull()
            return (col != 0).mask(nulls, pd.NA).astype(safe_dtype)
        elif pa.types.is_string(from_type):
            lower = col.str.lower()
            res = lower.isin(["true", "1", "1.0"])
            nulls = (~res) & (~lower.isin(["false", "0", "0.0"]))
            return res.mask(nulls, pd.NA).astype(safe_dtype)
        raise SlideCastError(f"unable to cast from {from_type} to {safe_dtype}")

    def _cast_to_int(
        self,
        col: TCol,
        from_type: pa.DataType,
        inf_type: pa.DataType,
        safe_dtype: np.dtype,
    ) -> TCol:
        def _convert_int_like() -> TCol:
            nulls = col.isnull()
            tp = to_single_pandas_dtype(
                self.to_safe_pa_type(safe_dtype), use_extension_types=False
            )
            return col.fillna(0).astype(tp).astype(safe_dtype).mask(nulls, pd.NA)

        if pa.types.is_boolean(from_type):
            if pa.types.is_string(inf_type):  # bool with nulls
                return _convert_int_like()
            return col.astype(safe_dtype)
        elif pa.types.is_integer(from_type):
            if pa.types.is_string(inf_type):  # pragma: no cover
                # int with nulls
                return _convert_int_like()
            return col.astype(safe_dtype)
        elif pa.types.is_floating(from_type):
            nulls = col.isnull()
            tp = to_single_pandas_dtype(
                self.to_safe_pa_type(safe_dtype), use_extension_types=False
            )
            return col.fillna(0).astype(tp).astype(safe_dtype).mask(nulls, pd.NA)
        elif pa.types.is_string(from_type):  # integer string representations
            # SQL can convert '1.1' to 1 as an integer
            temp = self._cast_to_float(
                col, from_type=from_type, inf_type=inf_type, safe_dtype=np.float64
            )
            nulls = temp.isnull()
            tp = to_single_pandas_dtype(
                self.to_safe_pa_type(safe_dtype), use_extension_types=False
            )
            return temp.fillna(0).astype(tp).astype(safe_dtype).mask(nulls, pd.NA)
        raise SlideCastError(f"unable to cast from {from_type} to {safe_dtype}")

    def _cast_to_float(
        self,
        col: TCol,
        from_type: pa.DataType,
        inf_type: pa.DataType,
        safe_dtype: np.dtype,
    ) -> TCol:
        return col.astype(safe_dtype)

    def _cast_to_str(
        self,
        col: TCol,
        from_type: pa.DataType,
        inf_type: pa.DataType,
        safe_dtype: np.dtype,
    ) -> TCol:
        nulls = col.isnull()
        if pa.types.is_boolean(from_type):
            if pa.types.is_boolean(inf_type):
                return col.astype(safe_dtype).str.lower().mask(nulls, pd.NA)
            if (
                pa.types.is_integer(inf_type)
                or pa.types.is_floating(inf_type)
                or pa.types.is_string(inf_type)  # bool with nulls
            ):
                return (
                    (col != 0)
                    .astype("boolean")
                    .astype(safe_dtype)
                    .str.lower()
                    .mask(nulls, pd.NA)
                )
            else:  # pragma: no cover
                raise SlideCastError(
                    f"underlying data type {inf_type} is impossible to be boolean"
                )
        if pa.types.is_integer(from_type) and inf_type != from_type:
            return (
                col.fillna(0)
                .astype(to_single_pandas_dtype(from_type, use_extension_types=False))
                .astype(safe_dtype)
                .mask(nulls, pd.NA)
            )
        return col.astype(safe_dtype).mask(nulls, pd.NA)

    def _cast_to_datetime(
        self,
        col: TCol,
        from_type: pa.DataType,
        inf_type: pa.DataType,
        safe_dtype: np.dtype,
    ) -> TCol:
        return col.astype(safe_dtype)

    def _cast_to_date(
        self,
        col: TCol,
        from_type: pa.DataType,
        inf_type: pa.DataType,
        safe_dtype: np.dtype,
    ) -> TCol:
        if pd.__version__ < "1.2":  # pragma: no cover
            return col.astype(safe_dtype).dt.floor("D")
        return col.astype(safe_dtype).dt.date


class SlideDataFrameNativeConverter:
    def __init__(
        self,
        utils: SlideUtils,
        schema: pa.Schema,
        type_safe: bool,
    ):
        """Convert pandas like dataframe to iterable of rows in the format of list.

        :param utils: the associated SlideUtils
        :param schema: schema of the input dataframe
        :param type_safe: whether to enforce the types in schema, if False, it will
            return the original values from the dataframes

        .. note::

        If there are nested types in schema, the conversion can be slower
        """
        self._utils = utils
        self._schema = schema
        self._has_time = any(
            pa.types.is_timestamp(x) or pa.types.is_date(x) for x in schema.types
        )
        if not type_safe:
            self._as_array_iterable = self._as_array_iterable_not_type_safe
            self._as_arrays = self._as_arrays_not_type_safe
            self._as_dict_iterable = self._as_dict_iterable_not_type_safe
            self._as_dicts = self._as_dicts_not_type_safe
        else:
            self._split_nested(self._schema)
            if self._converter is None:
                self._as_array_iterable = self._as_array_iterable_simple
                self._as_arrays = self._as_arrays_simple
                self._as_dict_iterable = self._as_dict_iterable_simple
                self._as_dicts = self._as_dicts_simple
            elif self._simple_part is None:
                self._as_array_iterable = self._as_array_iterable_nested
                self._as_arrays = self._as_arrays_nested
                self._as_dict_iterable = self._as_dict_iterable_nested
                self._as_dicts = self._as_dicts_nested
            else:
                self._as_array_iterable = self._as_array_iterable_hybrid
                self._as_arrays = self._as_arrays_hybrid
                self._as_dict_iterable = self._as_dict_iterable_hybrid
                self._as_dicts = self._as_dicts_hybrid
        pass

    def as_array_iterable(self, df: Any) -> Iterable[List[Any]]:
        """Convert the dataframe to an iterable of rows in the format of list.

        :param df: the dataframe
        :return: an iterable of rows, each row is a list
        """
        return self._as_array_iterable(df)

    def as_arrays(self, df: Any) -> List[List[Any]]:
        """Convert the dataframe to a list of rows in the format of list.

        :param df: the dataframe
        :return: a list of rows, each row is a list
        """
        return self._as_arrays(df)

    def as_dict_iterable(self, df: Any) -> Iterable[Dict[str, Any]]:
        """Convert the dataframe to an iterable of rows in the format of dict.

        :param df: the dataframe
        :return: an iterable of rows, each row is a dict
        """
        return self._as_dict_iterable(df)

    def as_dicts(self, df: Any) -> List[Dict[str, Any]]:
        """Convert the dataframe to a list of rows in the format of dict.

        :param df: the dataframe
        :return: a list of rows, each row is a dict
        """
        return self._as_dicts(df)

    def _time_safe(self, df: Any) -> Any:
        return df.astype(object) if self._has_time else df

    def _as_array_iterable_not_type_safe(self, df: Any) -> Iterable[List[Any]]:
        for arr in self._time_safe(df).itertuples(index=False, name=None):
            yield list(arr)

    def _as_arrays_not_type_safe(self, df: Any) -> List[List[Any]]:
        return self._time_safe(self._utils.as_pandas(df)).values.tolist()

    def _as_dict_iterable_not_type_safe(self, df: Any) -> Iterable[Dict[str, Any]]:
        names = list(self._schema.names)
        for arr in self._time_safe(df).itertuples(index=False, name=None):
            yield dict(zip(names, arr))

    def _as_dicts_not_type_safe(self, df: Any) -> List[Dict[str, Any]]:
        return self._time_safe(self._utils.as_pandas(df)).to_dict("records")

    def _as_array_iterable_simple(self, df: Any) -> Iterable[List[Any]]:
        return self._get_arrow_arrays_simple(df, self._schema)

    def _as_arrays_simple(self, df: Any) -> List[List[Any]]:
        return list(self._get_arrow_arrays_simple(df, self._schema))

    def _as_dict_iterable_simple(self, df: Any) -> Iterable[Dict[str, Any]]:
        for arr in self._get_arrow_arrays_simple(df, self._schema):
            yield dict(zip(self._schema.names, arr))

    def _as_dicts_simple(self, df: Any) -> List[Dict[str, Any]]:
        return list(self._as_dict_iterable_simple(df))

    def _as_array_iterable_hybrid(self, df: Any) -> Iterable[List[Any]]:
        for arr1, arr2 in zip(self._simple_part(df), self._nested_part(df)):
            yield self._remap_arrs(arr1, arr2)

    def _as_arrays_hybrid(self, df: Any) -> List[List[Any]]:
        return list(self._as_array_iterable_hybrid(df))

    def _as_dict_iterable_hybrid(self, df: Any) -> Iterable[Dict[str, Any]]:
        names = list(self._schema.names)
        for arr in self._as_array_iterable_hybrid(df):
            yield dict(zip(names, arr))

    def _as_dicts_hybrid(self, df: Any) -> List[Dict[str, Any]]:
        return list(self._as_dict_iterable_hybrid(df))

    def _as_array_iterable_nested(self, df: Any) -> Iterable[List[Any]]:
        return self._nested_part(df)

    def _as_arrays_nested(self, df: Any) -> List[List[Any]]:
        return list(self._nested_part(df))

    def _as_dict_iterable_nested(self, df: Any) -> Iterable[Dict[str, Any]]:
        names = list(self._schema.names)
        for arr in self._nested_part(df):
            yield dict(zip(names, arr))

    def _as_dicts_nested(self, df: Any) -> List[Dict[str, Any]]:
        return list(self._as_dict_iterable_nested(df))

    def _split_nested(self, schema: pa.Schema) -> None:
        cols1: List[pa.Field] = []
        cols2: List[pa.Field] = []
        self._remap: List[Tuple[int, int]] = []
        for field in schema:
            if pa.types.is_nested(field.type):
                self._remap.append((1, len(cols2)))
                cols2.append(field)
            else:
                self._remap.append((0, len(cols1)))
                cols1.append(field)
        self._simple_schema = pa.schema(cols1)
        self._simple_part: Any = (
            None
            if len(cols1) == 0
            else lambda df: self._get_arrow_arrays_simple(
                df[self._simple_schema.names], self._simple_schema
            )
        )
        self._nested_schema = pa.schema(cols2)
        self._converter: Any = (
            None
            if len(cols2) == 0
            else _TypeConverter(
                pa.schema(cols2), copy=True, deep=True, str_as_json=True
            )
        )
        self._nested_part = lambda df: self._get_arrays_nested(
            df[self._nested_schema.names], self._nested_schema
        )

    def _remap_arrs(self, *arrs: List[List[Any]]) -> List[Any]:
        return [arrs[x[0]][x[1]] for x in self._remap]

    def _get_arrow_arrays_simple(
        self, df: Any, schema: pa.Schema
    ) -> Iterable[List[Any]]:
        p = self._utils.as_arrow(df, schema, True)
        d = p.to_pydict()
        cols = [d[n] for n in schema.names]
        for arr in zip(*cols):
            yield list(arr)

    def _get_arrays_nested(self, df: Any, schema: pa.Schema) -> Iterable[List[Any]]:
        for item in df.itertuples(index=False, name=None):
            yield self._converter.row_to_py(item)
