import pandas as pd
from slide.operators.execution_plan import ExecutionPlan
from slide.operators.graph import Context
from slide_pandas import PandasUtils
from slide_test.utils import assert_duck_eq
from triad import Schema


def test_select():
    plan = ExecutionPlan()
    df = plan.df("a", Schema("a:long,b:double").pa_schema)
    assert Schema(df.output_schema) == "a:long,b:double"
    a = plan.col(df, "a")
    ln = len(plan)
    plan.col(df, "a")
    assert ln == len(plan)  # dedup
    b = plan.col(df, "b")
    c = plan.binary("+", a, b)
    df = plan.cols_to_df(a, b, (c, "c"), reference=df)
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


def test_union():
    plan = ExecutionPlan()
    df1 = plan.df("a", Schema("a:uint32,b:int,c:bool").pa_schema)
    df2 = plan.df("b", Schema("aa:int8,bb:double,cc:str").pa_schema)
    df = plan.union(df1, df2)
    assert Schema(df.output_schema) == "a:long,b:double,c:str"
    plan.output(df)
    assert Schema(plan.output_schema) == "a:long,b:double,c:str"

    ctx = Context(PandasUtils())
    pdf1 = pd.DataFrame([[0, None, True], [2, 3, False]], columns=["a", "b", "c"])
    ctx["a"] = pdf1
    pdf2 = pd.DataFrame([[-1, 1.1, "x"], [-2, 3.1, None]], columns=["aa", "bb", "cc"])
    ctx["b"] = pdf2
    plan.execute(ctx)

    assert_duck_eq(
        ctx.output,
        "SELECT a, b, c FROM a UNION SELECT aa,bb,cc FROM b",
        a=pdf1,
        b=pdf2,
        check_order=False,
    )


def test_union_all():
    plan = ExecutionPlan()
    df1 = plan.df("a", Schema("a:uint32,b:int,c:bool").pa_schema)
    df = plan.union(df1, df1, distinct=False)
    assert Schema(df.output_schema) == Schema("a:uint32,b:int,c:bool")
    plan.output(df)
    assert Schema(plan.output_schema) == Schema("a:uint32,b:int,c:bool")

    ctx = Context(PandasUtils())
    pdf1 = pd.DataFrame([[0, None, True], [2, 3, False]], columns=["a", "b", "c"])
    ctx["a"] = pdf1
    plan.execute(ctx)

    assert_duck_eq(
        ctx.output,
        "SELECT a, b, c FROM a UNION ALL SELECT a, b, c FROM a",
        a=pdf1,
        check_order=False,
    )


def test_except():

    plan = ExecutionPlan()
    df1 = plan.df("a", Schema("a:uint32,b:int").pa_schema)
    df2 = plan.df("b", Schema("a:long,b:long").pa_schema)
    df = plan.except_df(df1, df2)
    assert Schema(df.output_schema) == Schema("a:uint32,b:int")
    plan.output(df)
    assert Schema(plan.output_schema) == Schema("a:uint32,b:int")

    ctx = Context(PandasUtils())
    pdf1 = pd.DataFrame([[1, 2], [2, 3], [1, 2], [2, 3]], columns=["a", "b"])
    ctx["a"] = pdf1
    pdf2 = pd.DataFrame([[2, 3], [4, 5]], columns=["a", "b"])
    ctx["b"] = pdf2
    plan.execute(ctx)

    assert_duck_eq(
        ctx.output,
        "SELECT a, b FROM a EXCEPT SELECT a, b FROM b",
        a=pdf1,
        b=pdf2,
        check_order=False,
    )


def test_intersect():
    plan = ExecutionPlan()
    df1 = plan.df("a", Schema("a:uint32,b:int").pa_schema)
    df2 = plan.df("b", Schema("a:long,b:long").pa_schema)
    df = plan.intersect(df1, df2)
    assert Schema(df.output_schema) == Schema("a:uint32,b:int")
    plan.output(df)
    assert Schema(plan.output_schema) == Schema("a:uint32,b:int")

    ctx = Context(PandasUtils())
    pdf1 = pd.DataFrame([[1, 2], [2, 3], [1, 2], [2, 3]], columns=["a", "b"])
    ctx["a"] = pdf1
    pdf2 = pd.DataFrame([[2, 3], [4, 5]], columns=["a", "b"])
    ctx["b"] = pdf2
    plan.execute(ctx)

    assert_duck_eq(
        ctx.output,
        "SELECT a, b FROM a INTERSECT SELECT a, b FROM b",
        a=pdf1,
        b=pdf2,
        check_order=False,
    )


def test_filter():
    plan = ExecutionPlan()
    df = plan.df("a", Schema("a:long,b:bool,c:bool").pa_schema)
    df = plan.filter_df(df, "c", drop=False)
    assert Schema(df.output_schema) == Schema("a:long,b:bool,c:bool")
    plan.output(df)
    assert Schema(plan.output_schema) == Schema("a:long,b:bool,c:bool")

    ctx = Context(PandasUtils())
    pdf1 = pd.DataFrame([[1, True, False], [2, True, True]], columns=["a", "b", "c"])
    ctx["a"] = pdf1
    plan.execute(ctx)

    assert_duck_eq(
        ctx.output,
        "SELECT a,b,c FROM a WHERE c",
        a=pdf1,
        check_order=False,
    )

    plan = ExecutionPlan()
    df = plan.df("a", Schema("a:long,b:bool,c:bool").pa_schema)
    df = plan.filter_df(df, "b")
    assert Schema(df.output_schema) == Schema("a:long,c:bool")
    plan.output(df)
    assert Schema(plan.output_schema) == Schema("a:long,c:bool")

    ctx = Context(PandasUtils())
    pdf1 = pd.DataFrame([[1, True, False], [2, True, True]], columns=["a", "b", "c"])
    ctx["a"] = pdf1
    plan.execute(ctx)

    assert_duck_eq(
        ctx.output,
        "SELECT a,c FROM a WHERE b",
        a=pdf1,
        check_order=False,
    )


def test_join():
    plan = ExecutionPlan()
    df1 = plan.df("a", Schema("a:uint32,b:int").pa_schema)
    df2 = plan.df("b", Schema("a:long,c:long").pa_schema)
    df = plan.join(df1, df2, "inner")
    assert Schema(df.output_schema) == Schema("b:int,c:long")
    plan.output(df)
    assert Schema(plan.output_schema) == Schema("b:int,c:long")

    ctx = Context(PandasUtils())
    pdf1 = pd.DataFrame([[1, 2], [2, 3], [1, 2], [2, 3]], columns=["a", "b"])
    ctx["a"] = pdf1
    pdf2 = pd.DataFrame([[2, 3], [-4, 5]], columns=["a", "c"])
    ctx["b"] = pdf2
    plan.execute(ctx)

    assert_duck_eq(
        ctx.output,
        "SELECT a.b, b.c FROM a INNER JOIN b ON a.a = b.a",
        a=pdf1,
        b=pdf2,
        check_order=False,
    )
