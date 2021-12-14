from typing import Any, Dict, List, Optional

import pyarrow as pa
from slide.exceptions import SlideInvalidOperation
from slide.utils import SlideUtils
from triad import assert_or_throw, to_uuid


class Operator:
    def __init__(self, *args: Any, **kwargs: Any):
        self._uuid = to_uuid(self.identifier, args, kwargs)
        pass

    @property
    def identifier(self) -> str:
        return str(type(self))

    @property
    def key(self) -> str:
        return "_" + to_uuid(self)[:8]

    def __uuid__(self) -> str:
        return self._uuid

    def execute(self, context: "Context") -> None:
        raise NotImplementedError  # pragma: no cover


class Context:
    def __init__(self, utils: SlideUtils):
        self._utils = utils
        self._output: Any = None
        self._results: Dict[str, Any] = {}

    @property
    def utils(self) -> SlideUtils:
        return self._utils

    @property
    def output(self) -> Any:
        return self._output

    def set_output(self, df: Any) -> None:
        self._output = df

    def __setitem__(self, op: Operator, value: Any) -> None:
        self._results[op.key] = value

    def __getitem__(self, op: Operator) -> None:
        return self._results[op.key]


class Graph:
    def __init__(self):
        self._steps: List[Operator] = []
        self._steps_dict: Dict[str, Operator] = {}
        self._output_schema: Optional[pa.Schema] = None

    @property
    def output_schema(self) -> pa.Schema:
        assert_or_throw(
            self._output_schema is not None, SlideInvalidOperation("output is not set")
        )
        return self._output_schema

    def set_output_schema(self, schema: pa.Schema) -> None:
        assert_or_throw(
            self._output_schema is None, SlideInvalidOperation("output is already set")
        )
        self._output_schema = schema

    def add(self, op: Operator) -> Operator:
        key = op.key
        if key in self._steps_dict:
            return self._steps_dict[key]
        self._steps_dict[key] = op
        self._steps.append(op)
        return op

    @property
    def steps(self) -> List[Operator]:
        return self._steps

    def __len__(self) -> int:
        return len(self._steps)

    def execute(self, context: Context) -> None:
        for step in self.steps:
            step.execute(context)
