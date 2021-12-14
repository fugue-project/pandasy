from typing import Any, Callable, Optional

import pyarrow as pa
from slide.operators.graph import Context, Graph, Operator
from slide.operators.map_operators import MapExecutionPlan, MapOperationsContext


class DataFrameOperator(Operator):
    @property
    def output_schema(self) -> pa.Schema:
        raise NotImplementedError  # pragma: no cover


class GetDataFrameOperator(DataFrameOperator):
    def __init__(self, df: Any, input_schema: pa.Schema):
        super().__init__(id(df), str(input_schema))
        self._df = df
        self._schema = input_schema

    @property
    def output_schema(self) -> pa.Schema:
        return self._schema

    def execute(self, context: Context) -> None:
        context[self] = self._df


class SelectOperator(DataFrameOperator):
    def __init__(
        self, df: DataFrameOperator, builder: Callable[[MapExecutionPlan], None]
    ):
        self._plan = MapExecutionPlan(df.output_schema)
        builder(self._plan)
        self._output_schema = self._plan.output_schema
        self._df = df
        super().__init__(df, self._plan)

    @property
    def output_schema(self) -> pa.Schema:
        return self._output_schema

    def execute(self, context: Context) -> None:
        indf = context[self._df]
        ctx = MapOperationsContext(context.utils, indf)
        self._plan.execute(ctx)
        context[self] = ctx.output


class OutputDataFrameOperator(DataFrameOperator):
    def __init__(self, df: DataFrameOperator):
        super().__init__(df)
        self._df = df

    def execute(self, context: Context) -> None:
        context.set_output(context[self._df])


class ExecutionPlan(Graph):
    def __init__(self):
        super().__init__()
        self._output_schema: Optional[pa.Schema] = None

    def df(self, df: Any, input_schema: pa.Schema) -> Operator:
        return self.add(GetDataFrameOperator(df, input_schema))

    def select(
        self, df: DataFrameOperator, builder: Callable[[MapExecutionPlan], None]
    ) -> Operator:
        return self.add(SelectOperator(df, builder))

    def output(self, df: DataFrameOperator) -> None:
        self.add(OutputDataFrameOperator(df))
        self.set_output_schema(df.output_schema)
