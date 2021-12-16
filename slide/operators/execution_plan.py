from builtins import isinstance
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
import pyarrow as pa
from slide._type_utils import infer_union_type
from slide.operators.graph import Context, Graph, Node, Operator


class DataFrameOperator(Operator):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self._nodes: Dict[str, Node] = {}

    @property
    def output_schema(self) -> pa.Schema:
        raise NotImplementedError  # pragma: no cover

    @property
    def nodes(self) -> Dict[str, Node]:
        return self._nodes


class MapOperator(Operator):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)

        nodes = {x.node for x in args if isinstance(x, MapOperator)}
        nodes = nodes.union(
            x.node for x in kwargs.values() if isinstance(x, MapOperator)
        )
        self._node = Node(nodes)

    @property
    def output_type(self) -> pa.DataType:
        raise NotImplementedError  # pragma: no cover

    @property
    def output_name(self) -> Optional[str]:
        raise NotImplementedError  # pragma: no cover

    @property
    def node(self) -> Node:
        return self._node


class GetDataFrameOperator(DataFrameOperator):
    def __init__(self, name: str, input_schema: pa.Schema):
        super().__init__(name, str(input_schema))
        self._name = name
        self._schema = input_schema

        for f in input_schema:
            self.nodes[f.name] = Node(set())

    @property
    def output_schema(self) -> pa.Schema:
        return self._schema

    def execute(self, context: Context) -> None:
        context[self] = context[self._name]


class UnionOperator(DataFrameOperator):
    def __init__(
        self, df1: DataFrameOperator, df2: DataFrameOperator, distinct: bool = True
    ):
        super().__init__(df1, df2, distinct)
        fields1: List[pa.Field] = []
        fields2: List[pa.Field] = []
        for f1, f2 in zip(df1.output_schema, df2.output_schema):
            inf_type = infer_union_type(f1.type, f2.type)
            fields1.append(pa.field(f1.name, inf_type))
            fields2.append(pa.field(f2.name, inf_type))
            self.nodes[f1.name] = Node({df1.nodes[f1.name], df2.nodes[f2.name]})
        self._schema1 = pa.schema(fields1)
        self._schema2 = pa.schema(fields2)
        self._output_schema = self._schema1
        self._df1 = df1
        self._df2 = df2
        self._distinct = distinct

    @property
    def output_schema(self) -> pa.Schema:
        return self._output_schema

    def execute(self, context: Context) -> None:
        df1 = context[self._df1]
        df2 = context[self._df2]
        df1 = context.utils.cast_df(df1, self._schema1, self._df1.output_schema)
        df2 = context.utils.cast_df(df2, self._schema2, self._df2.output_schema)
        context[self] = context.utils.union(df1, df2, unique=self._distinct)


class ExceptOperator(DataFrameOperator):
    def __init__(
        self, df1: DataFrameOperator, df2: DataFrameOperator, distinct: bool = True
    ):
        super().__init__(df1, df2, distinct)
        self._df1 = df1
        self._df2 = df2
        self._distinct = distinct
        for f1, f2 in zip(df1.output_schema, df2.output_schema):
            self.nodes[f1.name] = Node({df1.nodes[f1.name], df2.nodes[f2.name]})

    @property
    def output_schema(self) -> pa.Schema:
        return self._df1.output_schema

    def execute(self, context: Context) -> None:
        df1 = context[self._df1]
        df2 = context[self._df2]
        context[self] = context.utils.except_df(df1, df2, unique=self._distinct)


class IntersectOperator(DataFrameOperator):
    def __init__(
        self, df1: DataFrameOperator, df2: DataFrameOperator, distinct: bool = True
    ):
        super().__init__(df1, df2, distinct)
        self._df1 = df1
        self._df2 = df2
        self._distinct = distinct
        for f1, f2 in zip(df1.output_schema, df2.output_schema):
            self.nodes[f1.name] = Node({df1.nodes[f1.name], df2.nodes[f2.name]})

    @property
    def output_schema(self) -> pa.Schema:
        return self._df1.output_schema

    def execute(self, context: Context) -> None:
        df1 = context[self._df1]
        df2 = context[self._df2]
        context[self] = context.utils.intersect(df1, df2, unique=self._distinct)


class FilterOperator(DataFrameOperator):
    def __init__(self, df: DataFrameOperator, filter_col: str, drop: bool = True):
        super().__init__(df, filter_col, drop)
        self._df = df
        self._filter_col = filter_col
        self._drop = drop
        if not drop:
            self._output_schema = df.output_schema
        else:
            self._output_schema = [x for x in df.output_schema if x.name != filter_col]
        for f in self._output_schema:
            self.nodes[f.name] = Node({df.nodes[f.name], df.nodes[filter_col]})

    @property
    def output_schema(self) -> pa.Schema:
        return self._output_schema

    def execute(self, context: Context) -> None:
        df = context[self._df]
        res = context.utils.filter_df(df, df[self._filter_col])
        if self._drop:
            res = context.utils.drop_columns(df, [self._filter_col])
        context[self] = res


class JoinOperator(DataFrameOperator):
    def __init__(self, df1: DataFrameOperator, df2: DataFrameOperator, how: str):
        super().__init__(df1, df2, how)
        self._df1 = df1
        self._df2 = df2
        self._how = how
        self._on = list(
            set(f.name for f in df1.output_schema).intersection(  # noqa: C401
                f.name for f in df2.output_schema
            )
        )
        f1 = [f for f in df1.output_schema if f.name not in self._on]
        f2 = [f for f in df2.output_schema if f.name not in self._on]
        self._output_schema = pa.schema(f1 + f2)

        on_nodes = [df1.nodes[n] for n in self._on] + [df2.nodes[n] for n in self._on]
        for f in f1:
            self.nodes[f.name] = Node({df1.nodes[f.name], *on_nodes})
        for f in f2:
            self.nodes[f.name] = Node({df2.nodes[f.name], *on_nodes})

    @property
    def output_schema(self) -> pa.Schema:
        return self._output_schema

    def execute(self, context: Context) -> None:
        df1 = context[self._df1]
        df2 = context[self._df2]
        res = context.utils.join(df1, df2, join_type=self._how, on=self._on)
        context[self] = context.utils.select_columns(
            res, [f.name for f in self.output_schema]
        )


class OutputDataFrameOperator(DataFrameOperator):
    def __init__(self, df: DataFrameOperator):
        super().__init__(df)
        self._df = df

        for k, v in df.nodes.items():
            self.nodes[k] = v

    def execute(self, context: Context) -> None:
        context.set_output(context[self._df])


class GetColumn(MapOperator):
    def __init__(self, df: DataFrameOperator, name: str):
        super().__init__(df, name)
        self._name = name
        self._df = df
        self._node = df.nodes[name]

    @property
    def output_type(self) -> pa.DataType:
        return self._df.output_schema.field_by_name(self._name).type

    @property
    def output_name(self) -> Optional[str]:
        return self._name

    def execute(self, context: Context) -> None:  # type: ignore
        context[self] = context[self._df][self._name]


class LitColumn(MapOperator):
    def __init__(self, value: Any, input_type: Optional[pa.DataType] = None):
        super().__init__(value, str(input_type))
        self._value = value
        self._output_type = pa.scalar(value, input_type).type

    @property
    def output_type(self) -> pa.DataType:
        return self._output_type

    def execute(self, context: Context) -> None:
        context[self] = self._value


class UnaryOperator(MapOperator):
    def __init__(self, op: str, col: MapOperator):
        super().__init__(op, col)
        self._op = op
        self._col = col
        self._output_type = self._get_output_type(op, col.output_type)

    @property
    def output_type(self) -> pa.DataType:
        return self._output_type

    @property
    def output_name(self) -> Optional[str]:
        return self._col.output_name

    def execute(self, context: Context) -> None:
        if self._op in ["+", "-"]:
            context[self] = context.utils.unary_arithmetic_op(
                context[self._col], op=self._op
            )
        elif self._op == "~":
            context[self] = context.utils.logical_not(context[self._col])
        else:
            raise NotImplementedError(self._op)  # pragma: no cover

    def _get_output_type(self, op: str, input_type: pa.DataType) -> pa.DataType:
        if op == "+":
            if pa.types.is_integer(input_type) or pa.types.is_floating(input_type):
                return input_type
        elif op == "-":
            if pa.types.is_integer(input_type):
                return pa.int64()
            if pa.types.is_floating(input_type):
                return input_type
        elif op == "~":
            if pa.types.is_boolean(input_type):
                return input_type
        raise ValueError(f"'{op}' can't be applied to {input_type}")


class BinaryOperator(MapOperator):
    def __init__(self, op: str, col1: MapOperator, col2: MapOperator):
        super().__init__(op, col1, col2)
        self._op = op
        self._col1 = col1
        self._col2 = col2
        self._output_type = self._get_output_type(
            op, col1.output_type, col2.output_type
        )

    @property
    def output_type(self) -> pa.DataType:
        return self._output_type

    def execute(self, context: Context) -> None:
        if self._op in ["+", "-", "*", "/"]:
            res = context.utils.binary_arithmetic_op(
                context[self._col1], context[self._col2], op=self._op
            )
            if (  # int/int -> int
                pa.types.is_integer(self._col1.output_type)
                and pa.types.is_integer(self._col2.output_type)
                and not pd.api.types.is_integer_dtype(res.dtype)
            ):
                res = context.utils.cast(res, "int64")
            context[self] = res
        elif self._op in ["&", "|"]:
            context[self] = context.utils.binary_logical_op(
                context[self._col1],
                context[self._col2],
                op="and" if self._op == "&" else "or",
            )
        else:
            raise NotImplementedError(self._op)  # pragma: no cover

    def _get_output_type(  # noqa: C901
        self, op: str, t1: pa.DataType, t2: pa.DataType
    ) -> pa.DataType:
        if op == "+":
            if pa.types.is_integer(t1):
                if pa.types.is_integer(t2):
                    return pa.int64()
                if pa.types.is_floating(t2):
                    return pa.float64()
            elif pa.types.is_floating(t1):
                if pa.types.is_integer(t2) or pa.types.is_floating(t2):
                    return pa.float64()
            # TODO: time + interval
        if op == "-":
            if pa.types.is_integer(t1):
                if pa.types.is_integer(t2):
                    return pa.int64()
                if pa.types.is_floating(t2):
                    return pa.float64()
            elif pa.types.is_floating(t1):
                if pa.types.is_integer(t2) or pa.types.is_floating(t2):
                    return pa.float64()
            # TODO: time - interval
            # TODO: time - time
        elif op in ["*", "/"]:
            if pa.types.is_integer(t1):
                if pa.types.is_integer(t2):
                    return pa.int64()
                if pa.types.is_floating(t2):
                    return pa.float64()
            elif pa.types.is_floating(t1):
                if pa.types.is_integer(t2) or pa.types.is_floating(t2):
                    return pa.float64()
        elif op in ["&", "|"]:
            if (pa.types.is_boolean(t1) or pa.types.is_null(t1)) and (
                pa.types.is_boolean(t2) or pa.types.is_null(t2)
            ):
                return pa.bool_()
        raise ValueError(  # pragma: no cover
            f"'{op}' can't be applied to {t1} and {t2}"
        )


class ColsToDataFrameOperator(DataFrameOperator):
    def __init__(
        self,
        *args: Union[MapOperator, Tuple[MapOperator, str]],
        reference: DataFrameOperator,
    ):
        self._data: List[Any] = [
            (x, x.output_name) if isinstance(x, MapOperator) else x for x in args
        ]
        super().__init__(*self._data, reference)
        self._output_schema = pa.schema(
            [pa.field(x[1], x[0].output_type) for x in self._data]
        )

        self._ref = reference
        self._nodes = {v: k.node for k, v in self._data}

    @property
    def output_schema(self) -> pa.Schema:
        return self._output_schema

    @property
    def nodes(self) -> Dict[str, Node]:
        return self._nodes

    def execute(self, context: Context) -> None:
        cols = [context[x] for x, _ in self._data]
        names = [y for _, y in self._data]
        context[self] = context.utils.cols_to_df(
            cols, names=names, reference=context[self._ref]
        )


class ExecutionPlan(Graph):
    def __init__(self):
        super().__init__()
        self._output_schema: Optional[pa.Schema] = None

    def df(self, df: Any, input_schema: pa.Schema) -> Operator:
        return self.add(GetDataFrameOperator(df, input_schema))

    def union(
        self, df1: DataFrameOperator, df2: DataFrameOperator, distinct: bool = True
    ) -> Operator:
        return self.add(UnionOperator(df1, df2, distinct=distinct))

    def except_df(
        self, df1: DataFrameOperator, df2: DataFrameOperator, distinct: bool = True
    ) -> Operator:
        return self.add(ExceptOperator(df1, df2, distinct=distinct))

    def intersect(
        self, df1: DataFrameOperator, df2: DataFrameOperator, distinct: bool = True
    ) -> Operator:
        return self.add(IntersectOperator(df1, df2, distinct=distinct))

    def filter_df(
        self, df: DataFrameOperator, filter_col: str, drop: bool = True
    ) -> Operator:
        return self.add(FilterOperator(df, filter_col, drop))

    def join(
        self, df1: DataFrameOperator, df2: DataFrameOperator, how: str
    ) -> Operator:
        return self.add(JoinOperator(df1, df2, how))

    def output(self, df: DataFrameOperator) -> None:
        self.add(OutputDataFrameOperator(df))
        self.set_output_schema(df.output_schema)

    def col(self, df: DataFrameOperator, name: str) -> Operator:
        return self.add(GetColumn(df, name))

    def lit(self, value: Any, input_type: Optional[pa.DataType] = None) -> Operator:
        return self.add(LitColumn(value, input_type))

    def unary(self, op: str, col: MapOperator) -> Operator:
        return self.add(UnaryOperator(op, col))

    def binary(self, op: str, col1: MapOperator, col2: MapOperator) -> Operator:
        return self.add(BinaryOperator(op, col1, col2))

    def cols_to_df(
        self,
        *args: Union[MapOperator, Tuple[MapOperator, str]],
        reference: DataFrameOperator,
    ) -> Operator:
        return self.add(ColsToDataFrameOperator(*args, reference=reference))
