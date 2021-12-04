from datetime import datetime, timedelta
from typing import Any, Dict

import numpy as np
import pandas as pd
import duckdb
import pyarrow as pa


def assert_duck_eq(
    df: pd.DataFrame,
    sql: Any,
    digits=8,
    check_col_order: bool = False,
    check_order: bool = False,
    check_content: bool = True,
    throw: bool = True,
    debug: bool = False,
    **tables: pd.DataFrame,
) -> bool:
    conn = duckdb.connect()
    try:
        for k, v in tables.items():
            conn.register_arrow(k, pa.Table.from_pandas(v))
        df2 = conn.execute(sql).fetchdf()
    finally:
        conn.close()

    return assert_pdf_eq(
        df,
        df2,
        digits == digits,
        check_col_order=check_col_order,
        check_order=check_order,
        check_content=check_content,
        throw=throw,
        debug=debug,
    )


def assert_pdf_eq(
    df: pd.DataFrame,
    data: Any,
    columns: Any = None,
    digits=8,
    check_col_order: bool = False,
    check_order: bool = False,
    check_content: bool = True,
    throw=True,
    debug: bool = False,
) -> bool:
    df1 = df
    df2 = (
        data if isinstance(data, pd.DataFrame) else pd.DataFrame(data, columns=columns)
    ).convert_dtypes()
    cols = list(df1.columns)
    try:
        if not check_col_order:
            assert sorted(cols) == sorted(
                df2.columns
            ), f"columns mismatch {sorted(cols)}, {sorted(df2.columns)}"
            df2 = df2[cols]
        else:
            assert cols == list(
                df2.columns
            ), f"columns mismatch {cols}, {list(df2.columns)}"
        assert df1.shape == df2.shape, f"shape mismatch {df1.shape}, {df2.shape}"

        if not check_content:
            return True
        if not check_order:
            df1 = df1.sort_values(cols)
            df2 = df2.sort_values(cols)
        df1 = df1.reset_index(drop=True)
        df2 = df2.reset_index(drop=True)

        if debug:
            print(df.dtypes)
            print(df)
            print(df2.dtypes)
            print(df2)

        pd.testing.assert_frame_equal(
            df1, df2, check_less_precise=digits, check_dtype=False
        )
        return True
    except AssertionError:
        if throw:
            raise
        return False


def make_rand_df(  # pragma: no cover  # noqa: C901
    size: int,
    **kwargs: Any,
) -> pd.DataFrame:
    np.random.seed(0)
    data: Dict[str, np.ndarray] = {}
    for k, v in kwargs.items():
        if not isinstance(v, tuple):
            v = (v, 0.0)
        dt, null_ct = v[0], v[1]
        if dt is int:
            s = np.random.randint(10, size=size)
        elif dt is bool:
            s = np.where(np.random.randint(2, size=size), True, False)
        elif dt is float:
            s = np.random.rand(size)
        elif dt is str:
            r = [f"ssssss{x}" for x in range(10)]
            c = np.random.randint(10, size=size)
            s = np.array([r[x] for x in c])
        elif dt is datetime:
            rt = [datetime(2020, 1, 1) + timedelta(days=x) for x in range(10)]
            c = np.random.randint(10, size=size)
            s = np.array([rt[x] for x in c])
        else:
            raise NotImplementedError
        ps = pd.Series(s)
        if null_ct > 0:
            idx = np.random.choice(size, null_ct, replace=False).tolist()
            ps[idx] = None
            if dt is bool:
                ps = ps.astype("boolean")
            if dt is int:
                ps = ps.astype("Int64")
            if dt is str:
                ps = ps.astype("string")
        data[k] = ps
    return pd.DataFrame(data)
