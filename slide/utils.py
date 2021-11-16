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
from triad.utils.pyarrow import TRIAD_DEFAULT_TIMESTAMP, apply_schema, to_pandas_dtype

TDF = TypeVar("TDF", bound=Any)
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


class SlideUtils(Generic[TDF, TCol]):
    """A collection of utils for general pandas like dataframes"""

    def cols_to_df(self, cols: List[TCol], names: Optional[List[str]] = None) -> TDF:
        """Construct the dataframe from a list of columns (serieses)

        :param cols: the collection of columns
        :param names: the correspondent column names, defaults to None

        :return: the dataframe

        .. note::

        If ``names`` is not provided, then every series in ``cols`` must be
        named. Otherise, ``names`` must align with ``cols``. But whether names
        have duplications or invalid chars will not be verified by this method
        """
        raise NotImplementedError  # pragma: no cover

    def empty(self, df: TDF) -> bool:
        """Check if the dataframe is empty

        :param df: pandas like dataframe
        :return: if it is empty
        """
        return len(df.index) == 0

    def as_arrow(self, df: TDF, schema: Optional[pa.Schema] = None) -> pa.Table:
        """Convert pandas like dataframe to pyarrow table

        :param df: pandas like dataframe
        :param schema: if specified, it will be used to construct pyarrow table,
          defaults to None
        :return: pyarrow table
        """
        return pa.Table.from_pandas(df, schema=schema, preserve_index=False, safe=False)

    def as_array_iterable(
        self,
        df: TDF,
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
        df: TDF,
        schema: Optional[pa.Schema] = None,
        columns: Optional[List[str]] = None,
        type_safe: bool = False,
    ) -> List[List[Any]]:
        return list(
            self.as_array_iterable(
                df, schema=schema, columns=columns, type_safe=type_safe
            )
        )

    def to_schema(self, df: TDF) -> pa.Schema:
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
        self, df: TDF, schema: pa.Schema, null_safe: bool = False
    ) -> TDF:
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
        df: TDF,
        cols: List[str],
        func: Callable[[TDF], TDF],
        **kwargs: Any,
    ) -> TDF:
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

    def is_compatile_index(self, df: TDF) -> bool:
        """Check whether the datafame is compatible with the operations inside
        this utils collection

        :param df: pandas like dataframe
        :return: if it is compatible
        """
        return isinstance(df.index, (pd.RangeIndex, pd.Int64Index, pd.UInt64Index))

    def ensure_compatible(self, df: TDF) -> None:
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

    def drop_duplicates(self, df: TDF) -> TDF:
        """Select distinct rows from dataframe

        :param df: the dataframe
        :return: the result with only distinct rows
        """
        return df.drop_duplicates(ignore_index=True)

    def union(self, df1: TDF, df2: TDF, unique: bool) -> TDF:
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

    def intersect(self, df1: TDF, df2: TDF, unique: bool) -> TDF:
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
        df1: TDF,
        df2: TDF,
        unique: bool,
        anti_indicator_col: str = _ANTI_INDICATOR,
    ) -> TDF:
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
        ndf1: TDF,
        ndf2: TDF,
        join_type: str,
        on: List[str],
        anti_indicator_col: str = _ANTI_INDICATOR,
        cross_indicator_col: str = _CROSS_INDICATOR,
    ) -> TDF:
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

    def _preprocess_set_op(self, ndf1: TDF, ndf2: TDF) -> Tuple[TDF, TDF]:
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

    def _with_indicator(self, df: TDF, name: str) -> TDF:
        return df.assign(**{name: 1})
