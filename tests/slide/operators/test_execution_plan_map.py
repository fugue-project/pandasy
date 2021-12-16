import pandas as pd
import pyarrow as pa
from pytest import raises
from slide.operators.execution_plan import ExecutionPlan
from slide.operators.graph import Context, Operator
from slide_pandas import PandasUtils
from slide_test.utils import assert_duck_eq
from triad import Schema


def test_col_op():
    def f(plan: ExecutionPlan, df: Operator):
        col0 = plan.col(df, "c")
        assert pa.bool_() == col0.output_type
        assert "c" == col0.output_name
        col1 = plan.col(df, "a")
        assert pa.uint32() == col1.output_type
        assert "a" == col1.output_name
        col2 = plan.col(df, "b")
        assert pa.float32() == col2.output_type
        assert "b" == col2.output_name

        return col0, col1, col2

    pdf = pd.DataFrame([[0, 1.1, True], [3, 4.1, False]], columns=["a", "b", "c"])
    assert_duck_eq(
        run_plan(pdf, "a:uint,b:float32,c:bool", f),
        "SELECT c, a, b FROM a",
        a=pdf,
        check_order=False,
    )


def test_lit_op():
    def f(plan: ExecutionPlan, df: Operator):
        col0 = plan.lit(None)
        assert pa.null() == col0.output_type
        col1 = plan.lit("abc")
        assert pa.string() == col1.output_type
        col2 = plan.lit(1, pa.uint8())
        assert pa.uint8() == col2.output_type
        col3 = plan.col(df, "a")

        return (col1, "x"), (col2, "y"), col3

    pdf = pd.DataFrame([[0, 1.1, True], [3, 4.1, False]], columns=["a", "b", "c"])
    assert_duck_eq(
        run_plan(pdf, "a:uint,b:float32,c:bool", f),
        "SELECT 'abc' AS x, 1 AS y, a FROM a",
        a=pdf,
        check_order=False,
    )


def test_pure_lit_op():
    def f(plan: ExecutionPlan, df: Operator):
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

        return (col1, "a"), (col2, "b"), (col3, "c"), (col4, "d")

    pdf = pd.DataFrame([[0, 1.1, True], [3, 4.1, False]], columns=["a", "b", "c"])
    res = run_plan(pdf, "a:uint,b:float32,c:bool", f)
    expected = [["abc", 1, b"\0abc", [1, 2]], ["abc", 1, b"\0abc", [1, 2]]]

    assert expected == res.astype(object).values.tolist()


def test_unary_op():
    def f(plan: ExecutionPlan, df: Operator):
        col0 = plan.col(df, "c")
        col1 = plan.col(df, "a")
        col2 = plan.col(df, "b")
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
        return (col3, "c3"), (col4, "c4"), (col5, "c5"), (col6, "c6"), (col10, "c10")

    pdf = pd.DataFrame([[0, 1.1, True], [3, 4.1, False]], columns=["a", "b", "c"])
    assert_duck_eq(
        run_plan(pdf, "a:uint,b:float32,c:bool", f),
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


def test_binary_op_num():
    def f(plan: ExecutionPlan, df: Operator):
        col1 = plan.col(df, "a")
        col2 = plan.col(df, "b")
        cola = plan.binary("+", col1, col1)
        assert pa.int64() == cola.output_type
        colb = plan.binary("-", col1, col1)
        assert pa.int64() == colb.output_type
        colc = plan.binary("*", col1, col1)
        assert pa.int64() == colc.output_type
        cold = plan.binary("/", col1, plan.lit(2))
        assert pa.int64() == cold.output_type

        cole = plan.binary("+", col1, col2)
        assert pa.float64() == cole.output_type
        colf = plan.binary("-", col1, col2)
        assert pa.float64() == colf.output_type
        colg = plan.binary("*", col1, col2)
        assert pa.float64() == colg.output_type
        colh = plan.binary("/", col1, col2)
        assert pa.float64() == colh.output_type

        coli = plan.binary("+", col2, col1)
        assert pa.float64() == coli.output_type
        colj = plan.binary("-", col2, col1)
        assert pa.float64() == colj.output_type
        colk = plan.binary("*", col2, col1)
        assert pa.float64() == colk.output_type
        coll = plan.binary("/", col2, col1)
        assert pa.float64() == coll.output_type

        colm = plan.binary("+", col2, col2)
        assert pa.float64() == colm.output_type
        coln = plan.binary("-", col2, col2)
        assert pa.float64() == coln.output_type
        colo = plan.binary("*", col2, col2)
        assert pa.float64() == colo.output_type
        colp = plan.binary("/", col2, col2)
        assert pa.float64() == colp.output_type

        return (
            (cola, "a"),
            (colb, "b"),
            (colc, "c"),
            (cold, "d"),
            (cole, "e"),
            (colf, "f"),
            (colg, "g"),
            (colh, "h"),
            (coli, "i"),
            (colj, "j"),
            (colk, "k"),
            (coll, "l"),
            (colm, "m"),
            (coln, "n"),
            (colo, "o"),
            (colp, "p"),
        )

    pdf = pd.DataFrame([[1, 1.1], [3, 4.1]], columns=["a", "b"])
    assert_duck_eq(
        run_plan(pdf, "a:uint,b:float32", f),
        """
        SELECT
            a+a AS a, a-a AS b, a*a AS c, a/2 AS d,
            a+b AS e, a-b AS f, a*b AS g, a/b AS h,
            b+a AS i, b-a AS j, b*a AS k, b/a AS l,
            b+b AS m, b-b AS n, b*b AS o, b/b AS p
        FROM a
        """,
        a=pdf,
        check_order=False,
    )


def test_binary_op_logical():
    def f(plan: ExecutionPlan, df: Operator):
        col1 = plan.col(df, "a")
        col2 = plan.col(df, "b")
        cola = plan.binary("&", col1, col2)
        assert pa.bool_() == cola.output_type
        colb = plan.binary("|", col1, col2)
        assert pa.bool_() == colb.output_type
        colc = plan.binary("&", col1, plan.lit(True))
        assert pa.bool_() == colc.output_type
        cold = plan.binary("&", col1, plan.lit(False))
        assert pa.bool_() == cold.output_type
        cole = plan.binary("&", col1, plan.lit(None))
        assert pa.bool_() == cole.output_type
        colf = plan.binary("|", col1, plan.lit(True))
        assert pa.bool_() == colf.output_type
        colg = plan.binary("|", col1, plan.lit(False))
        assert pa.bool_() == colg.output_type
        colh = plan.binary("|", col1, plan.lit(None))
        assert pa.bool_() == colh.output_type

        return (
            (cola, "a"),
            (colb, "b"),
            (colc, "c"),
            (cold, "d"),
            (cole, "e"),
            (colf, "f"),
            (colg, "g"),
            (colh, "h"),
        )

    pdf = pd.DataFrame(
        [
            [True, True],
            [True, False],
            [True, None],
            [False, True],
            [False, False],
            [False, None],
            [None, True],
            [None, False],
            [None, None],
        ],
        columns=["a", "b"],
    )
    assert_duck_eq(
        run_plan(pdf, "a:bool,b:bool", f),
        """
        SELECT
            a AND b AS a, a OR b AS b,
            a AND TRUE AS c, a AND FALSE AS d, a AND NULL AS e,
            a OR TRUE AS f, a OR FALSE AS g, a OR NULL AS h
        FROM a
        """,
        a=pdf,
        check_order=False,
    )


def test_binary_op_logical_2():
    def f(plan: ExecutionPlan, df: Operator, sql):
        output = []
        n = 0
        for op in ["&", "|"]:
            for left in [True, False, None]:
                for right in [True, False, None]:
                    name = f"_{n}"
                    col = plan.binary(op, plan.lit(left), plan.lit(right))
                    assert pa.bool_() == col.output_type
                    output.append((col, name))
                    ls = "NULL" if left is None else str(left).upper()
                    rs = "NULL" if right is None else str(right).upper()
                    o = "AND" if op == "&" else "OR"
                    sql.append(f"{ls} {o} {rs} AS {name}")
                    n += 1
        return output

    pdf = pd.DataFrame(
        [
            [True, True],
            [True, False],
        ],
        columns=["a", "b"],
    )
    sql = []
    res = run_plan(pdf, "a:bool,b:bool", lambda a, b: f(a, b, sql))
    _sql = ", ".join(sql)
    assert_duck_eq(
        res,
        f"SELECT {_sql} FROM a",
        a=pdf,
        check_order=False,
    )


def run_plan(pdf, schema, plan_func):
    plan = ExecutionPlan()
    df = plan.df("a", Schema(schema).pa_schema)
    args = plan_func(plan, df)
    res = plan.cols_to_df(*args, reference=df)
    plan.output(res)

    ctx = Context(PandasUtils())
    ctx["a"] = pdf
    plan.execute(ctx)

    return ctx.output
