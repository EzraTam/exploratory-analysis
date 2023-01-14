""" Module for executing basic analysis of a DF
"""

from typing import List, Union, Dict
from types import MethodType
from functools import reduce
import pandas as pd


def join_on_col(df_1: pd.DataFrame, df_2: pd.DataFrame, on: str) -> pd.DataFrame:
    """Function for joining two DFs

    Args:
        df_1 (pd.DataFrame): First DF
        df_2 (pd.DataFrame): Second DF
        on (str): The key for joining the DFs

    Returns:
        pd.DataFrame: Resulting DF
    """
    return pd.concat(
        [df_1.set_index(on), df_2.set_index(on)], axis=1, join="inner"
    ).reset_index()


def merge_columns(li_df: List[pd.DataFrame], key: str) -> pd.DataFrame:
    """Function for merging/joining DFs

    Args:
        li_df (List[pd.DataFrame]): List of DFs to be merged
        key (str): Merging key

    Returns:
        pd.DataFrame: Resulting DF
    """
    return reduce(
        lambda df_current, df_add: join_on_col(df_current, df_add, key), li_df
    )


def set_type_col_pd(
    df_to_set: pd.DataFrame, col_to_set: str, type_col: str
) -> pd.DataFrame:
    """Function for setting the type of a column in a DF

    Args:
        df_to_set (pd.DataFrame): DF to set
        col_to_set (str): Column, whose type to be set
        type_col (str): Corresponding desired type of the column

    Returns:
        pd.DataFrame: Desired DF
    """
    if type_col == "str":
        df_to_set[col_to_set].astype(str)

    if type_col == "int":
        df_to_set[col_to_set].astype(str)

    return df_to_set


def recurse_list_get_attribute_pd(
    input: Union[pd.DataFrame, pd.Series], list_to_recurse: List[str]
) -> pd.Series:
    """_summary_

    Args:
        input (Union[pd.DataFrame, pd.Series]): _description_
        list_to_recurse (List[str]): _description_

    Returns:
        pd.Series: _description_
    """

    if not list_to_recurse:
        return input

    attr = list_to_recurse[0]
    fg_attr_method_type = isinstance(getattr(input, attr), MethodType)

    return recurse_list_get_attribute_pd(
        input=getattr(input, attr)() if fg_attr_method_type else getattr(input, attr),
        list_to_recurse=list_to_recurse[1:],
    )


def basic_analysis_df(df_input: pd.DataFrame) -> Dict[str, Union[pd.DataFrame, int]]:
    """Function for giving a basic analysis of a DF
        Implemented:
            * type of a DF
            * number of unique values
            * number of null elements

    Args:
        df_input (pd.DataFrame): DF to be analyzed

    Returns:
        Dict[str,Union[pd.DataFrame,int]]: Dictionary containing
            * result DF
            * number of columns
            * number of rows
    """

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
            type_col="str",
        )
        for info_nm, li_attr in info_to_extract.items()
    ]

    df_info = merge_columns(li_df=li_df, key="col_nm")
    num_rows = df_input.shape[0]
    num_cols = df_input.shape[1]
    return dict(df_info=df_info, num_rows=num_rows, num_cols=num_cols)
