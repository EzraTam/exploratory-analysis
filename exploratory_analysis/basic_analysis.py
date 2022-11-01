from typing import List
from types import MethodType
from functools import reduce
import pandas as pd


def join_on_col(df_1, df_2, on):
    return pd.concat(
        [df_1.set_index(on), df_2.set_index(on)], axis=1, join="inner"
    ).reset_index()


def merge_columns(li_df: List[pd.DataFrame], key: str) -> pd.DataFrame:
    return reduce(
        lambda df_current, df_add: join_on_col(df_current, df_add, key), li_df
    )


def basic_analysis(df_input: pd.DataFrame) -> dict:

    info_to_extract = dict(
        type=["dtypes"],
        num_unique=["nunique"],
        num_null=["isna", "sum"],
    )

    li_df = []

    for info_nm, li_attr in info_to_extract.items():
        df_temp = df_input
        for attr in li_attr:
            if isinstance(getattr(df_temp, attr), MethodType):
                df_temp = getattr(df_temp, attr)()
            else:
                df_temp = getattr(df_temp, attr)
        df_temp = pd.DataFrame(df_temp, columns=[info_nm]).reset_index(names=["col_nm"])
        df_temp["col_nm"] = df_temp["col_nm"].astype(str)
        li_df.append(df_temp)

    df_info = merge_columns(li_df=li_df, key="col_nm")
    num_rows = df_input.shape[0]
    num_cols = df_input.shape[1]
    return dict(df_info=df_info, num_rows=num_rows, num_cols=num_cols)
