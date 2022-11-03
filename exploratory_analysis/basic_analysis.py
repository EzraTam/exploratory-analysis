from typing import List, Union
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

def set_type_col_pd(
    df_to_set: pd.DataFrame, col_to_set: str, type: str
) -> pd.DataFrame:
    df_to_set[col_to_set] = {
        "str": df_to_set[col_to_set].astype(str),
        "int": df_to_set[col_to_set].astype(int),
    }[type]
    return df_to_set

def recurse_list_get_attribute_pd(
    input: Union[pd.DataFrame, pd.Series], list_to_recurse: List[str]
) -> pd.Series:

    if not list_to_recurse:
        return input

    attr = list_to_recurse[0]
    fg_attr_method_type = isinstance(getattr(input, attr), MethodType)

    return recurse_list_get_attribute_pd(
        input=getattr(input, attr)() if fg_attr_method_type else getattr(input, attr),
        list_to_recurse=list_to_recurse[1:],
    )

def basic_analysis(df_input: pd.DataFrame) -> dict:

    info_to_extract = dict(
        type=["dtypes"],
        num_unique=["nunique"],
        num_null=["isna", "sum"],
    )

    li_df = [
        set_type_col_pd(
            df_to_set=pd.DataFrame(
                recurse_list_get_attribute_pd(df_input, li_attr), columns=[info_nm]
            ).reset_index(names=["col_nm"]),
            col_to_set="col_nm",
            type="str",
        )
        for info_nm, li_attr in info_to_extract.items()
    ]

    df_info = merge_columns(li_df=li_df, key="col_nm")
    num_rows = df_input.shape[0]
    num_cols = df_input.shape[1]
    return dict(df_info=df_info, num_rows=num_rows, num_cols=num_cols)
