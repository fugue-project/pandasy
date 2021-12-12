import pandas as pd
from slide.operators.select_operators import SelectExecutionContext, SelectExecutionPlan
from slide_pandas import PandasUtils
from slide_test.utils import assert_duck_eq, assert_pdf_eq
from triad import Schema
import pyarrow as pa
from pytest import raises


def test_col_op():
    pdf = pd.DataFrame([[0, 1.1, True], [3, 4.1, False]], columns=["a", "b", "c"])
    ctx = SelectExecutionContext(PandasUtils(), pdf)
    plan = SelectExecutionPlan(Schema("a:uint,b:float32,c:bool").pa_schema)
    col0 = plan.col("c")
    assert pa.bool_() == col0.output_type
    assert "c" == col0.output_name
    col1 = plan.col("a")
    assert pa.uint32() == col1.output_type
    assert "a" == col1.output_name
    col2 = plan.col("b")
    assert pa.float32() == col2.output_type
    assert "b" == col2.output_name

    plan.output(col0, col1, col2)
    plan.execute(ctx)

    assert_duck_eq(
        ctx.output,
        "SELECT c, a, b FROM a",
        a=pdf,
        check_order=False,
    )


def test_lit_op():
    pdf = pd.DataFrame([[0, 1.1, True], [3, 4.1, False]], columns=["a", "b", "c"])
    ctx = SelectExecutionContext(PandasUtils(), pdf)
    plan = SelectExecutionPlan(Schema("a:uint,b:float32,c:bool").pa_schema)
    col0 = plan.lit(None)
    assert pa.null() == col0.output_type
    col1 = plan.lit("abc")
    assert pa.string() == col1.output_type
    col2 = plan.lit(1, pa.uint8())
    assert pa.uint8() == col2.output_type
    col3 = plan.col("a")

    plan.output((col1, "x"), (col2, "y"), col3)
    plan.execute(ctx)

    assert_duck_eq(
        ctx.output,
        "SELECT 'abc' AS x, 1 AS y, a FROM a",
        a=pdf,
        check_order=False,
    )


def test_pure_lit_op():
    pdf = pd.DataFrame([[0, 1.1, True], [3, 4.1, False]], columns=["a", "b", "c"])
    ctx = SelectExecutionContext(PandasUtils(), pdf)
    plan = SelectExecutionPlan(Schema("a:uint,b:float32,c:bool").pa_schema)
    col0 = plan.lit(None)
    assert pa.null() == col0.output_type
    col1 = plan.lit("abc")
    assert pa.string() == col1.output_type
    col2 = plan.lit(1, pa.uint8())
    assert pa.uint8() == col2.output_type
    col3 = plan.lit(b"\0abc")
    assert pa.binary() == col3.output_type
    col4 = plan.lit([1, 2])
    assert pa.types.is_nested(col4.output_type)

    plan.output((col1, "a"), (col2, "b"), (col3, "c"), (col4, "d"))
    plan.execute(ctx)

    expected = [["abc", 1, b"\0abc", [1, 2]], ["abc", 1, b"\0abc", [1, 2]]]

    assert expected == ctx.output.astype(object).values.tolist()


def test_unary_op():
    pdf = pd.DataFrame([[0, 1.1, True], [3, 4.1, False]], columns=["a", "b", "c"])
    ctx = SelectExecutionContext(PandasUtils(), pdf)
    plan = SelectExecutionPlan(Schema("a:uint,b:float32,c:bool").pa_schema)
    col0 = plan.col("c")
    col1 = plan.col("a")
    col2 = plan.col("b")
    col3 = plan.unary("+", col1)
    assert pa.uint32() == col3.output_type
    assert "a" == col3.output_name
    col4 = plan.unary("-", col1)
    assert pa.int64() == col4.output_type
    col5 = plan.unary("+", col2)
    assert pa.float32() == col5.output_type
    col6 = plan.unary("-", col2)
    assert pa.float32() == col6.output_type

    raises(ValueError, lambda: plan.unary("-", col0))
    raises(ValueError, lambda: plan.unary("+", col0))
    raises(ValueError, lambda: plan.unary("~", col1))
    raises(ValueError, lambda: plan.unary("~", col2))

    col10 = plan.unary("~", col0)
    plan.output((col3, "c3"), (col4, "c4"), (col5, "c5"), (col6, "c6"), (col10, "c10"))
    plan.execute(ctx)

    assert_duck_eq(
        ctx.output,
        """
        SELECT
            a AS c3, -a AS c4,
            b AS c5, -b AS c6,
            NOT c AS c10
        FROM a
        """,
        a=pdf,
        check_order=False,
    )


def test_plan():
    pdf = pd.DataFrame([[0, 1.1], [3, 4.1]], columns=["a", "b"])
    ctx = SelectExecutionContext(PandasUtils(), pdf)
    plan = SelectExecutionPlan(Schema("a:int,b:float").pa_schema)
    col1 = plan.col("b")
    col2 = plan.col("a")
    col3 = plan.binary("+", col1, col2)
    col4 = plan.binary("-", col3, plan.lit(2))
    l1 = len(plan)
    col5 = plan.binary("+", col1, col2)  # dedupped
    assert l1 == len(plan)
    col6 = plan.unary("-", col5)
    # a, b, a+b as x, a+b-2 as y, -(a+b) as z
    plan.output(col1, col2, (col3, "x"), (col4, "y"), (col6, "z"))
    plan.execute(ctx)
    assert_duck_eq(
        ctx.output,
        "SELECT a, b, a+b AS x, a+b-2 AS y, -(a+b) AS z FROM a",
        a=pdf,
        check_order=False,
    )
