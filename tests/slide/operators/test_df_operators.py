import pandas as pd
from slide.operators.df_operators import ExecutionPlan
from slide.operators.graph import Context
from slide.operators.map_operators import MapExecutionPlan
from slide_pandas import PandasUtils
from slide_test.utils import assert_duck_eq
from triad import Schema


def test_select():
    def build(map_plan: MapExecutionPlan) -> None:
        a = map_plan.col("a")
        b = map_plan.col("b")
        c = map_plan.binary("+", a, b)
        map_plan.output(a, b, (c, "c"))

    plan = ExecutionPlan()
    df = plan.df("a", Schema("a:long,b:double").pa_schema)
    assert Schema(df.output_schema) == "a:long,b:double"
    df = plan.select(df, build)
    assert Schema(df.output_schema) == "a:long,b:double,c:double"
    plan.output(df)
    assert Schema(plan.output_schema) == "a:long,b:double,c:double"

    ctx = Context(PandasUtils())
    pdf = pd.DataFrame([[0, 1.2], [2, 3.1]], columns=["a", "b"])
    ctx["a"] = pdf
    plan.execute(ctx)

    assert_duck_eq(
        ctx.output,
        "SELECT a, b, a+b AS c FROM a",
        a=pdf,
        check_order=False,
    )
