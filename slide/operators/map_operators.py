from typing import Any, Dict, List, Optional, Tuple, Union

import pyarrow as pa
import pandas as pd
from slide.utils import SlideUtils
from triad import Schema, to_uuid


class MapOperator:
    def __init__(self, *args: Any, **kwargs: Any):
        self._args = args
        self._kwargs = kwargs
        self._uuid = to_uuid(self.identifier, self._args, self._kwargs)
        pass

    @property
    def identifier(self) -> str:
        return str(type(self))

    @property
    def key(self) -> str:
        return "_" + to_uuid(self)[:8]

    def execute(self, context: "MapOperationsContext") -> None:
        raise NotImplementedError  # pragma: no cover

    @property
    def output_type(self) -> pa.DataType:
        raise NotImplementedError  # pragma: no cover

    @property
    def output_name(self) -> Optional[str]:
        raise NotImplementedError  # pragma: no cover

    def __uuid__(self) -> str:
        return self._uuid


class GetColumn(MapOperator):
    def __init__(self, name: str, input_type: pa.DataType):
        super().__init__(name)
        self._name = name
        self._output_type = input_type

    @property
    def output_type(self) -> pa.DataType:
        return self._output_type

    @property
    def output_name(self) -> Optional[str]:
        return self._name

    def execute(self, context: "MapOperationsContext") -> None:
        context[self] = context.df[self._name]


class LitColumn(MapOperator):
    def __init__(self, value: Any, input_type: Optional[pa.DataType] = None):
        super().__init__(value)
        self._value = value
        self._output_type = pa.scalar(value, input_type).type

    @property
    def output_type(self) -> pa.DataType:
        return self._output_type

    def execute(self, context: "MapOperationsContext") -> None:
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

    def execute(self, context: "MapOperationsContext") -> None:
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

    def execute(self, context: "MapOperationsContext") -> None:
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


class MapOutputOperator(MapOperator):
    def __init__(self, *args: Union[MapOperator, Tuple[MapOperator, str]]):
        self._data: List[Any] = [
            (x, x.output_name) if isinstance(x, MapOperator) else x for x in args
        ]
        super().__init__(*self._data)

    def execute(self, context: "MapOperationsContext") -> None:
        cols = [context[x] for x, _ in self._data]
        names = [y for _, y in self._data]
        context.set_output(
            context.utils.cols_to_df(cols, names=names, reference=context.df)
        )


class MapOperationsContext:
    def __init__(self, utils: SlideUtils, df: Any):
        self._utils = utils
        self._df = df
        self._output: Any = None
        self._results: Dict[str, Any] = {}

    @property
    def utils(self) -> SlideUtils:
        return self._utils

    @property
    def df(self) -> Any:
        return self._df

    @property
    def output(self) -> Any:
        return self._output

    def set_output(self, df: Any) -> None:
        self._output = df

    def __setitem__(self, op: MapOperator, value: Any) -> None:
        self._results[op.key] = value

    def __getitem__(self, op: MapOperator) -> None:
        return self._results[op.key]


class MapExecutionPlan:
    def __init__(self, input_schema: pa.Schema):
        self._input_schema = input_schema
        self._steps: List[MapOperator] = []
        self._steps_dict: Dict[str, MapOperator] = {}

    def add(self, op: MapOperator) -> MapOperator:
        key = op.key
        if key in self._steps_dict:
            return self._steps_dict[key]
        self._steps_dict[key] = op
        self._steps.append(op)
        return op

    def col(self, name: str) -> MapOperator:
        return self.add(GetColumn(name, self._input_schema.field_by_name(name).type))

    def lit(self, value: Any, input_type: Optional[pa.DataType] = None) -> MapOperator:
        return self.add(LitColumn(value, input_type))

    def unary(self, op: str, col: MapOperator) -> MapOperator:
        return self.add(UnaryOperator(op, col))

    def binary(self, op: str, col1: MapOperator, col2: MapOperator) -> MapOperator:
        return self.add(BinaryOperator(op, col1, col2))

    def output(self, *args: Union[MapOperator, Tuple[MapOperator, str]]) -> None:
        self.add(MapOutputOperator(*args))

    def __len__(self) -> int:
        return len(self._steps)

    def __uuid__(self) -> str:
        return to_uuid(str(Schema(self._input_schema)), self._steps)

    def execute(self, context: MapOperationsContext) -> None:
        for step in self._steps:
            step.execute(context)
