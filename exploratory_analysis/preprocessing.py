"""Preprocessing Module
"""

from typing import List, Dict, Union, Optional
from itertools import product
import pandas as pd
import exploratory_analysis.basic_functions as bf


def transform_col_nm(
    dataframe: pd.DataFrame,
    how: str,
    what_char: Optional[str] = None,
    by_char: Optional[str] = None,
) -> pd.DataFrame:
    """Transform column names

    Args:
        dataframe (pd.DataFrame): DF to be transformed
        how (str): By which method should the column names be transformed.
            Existing methods:
                'to_snake' - transform the column names to snake case
                'replace' - replace a certain character in the column names by another character.
                    Choosing this option requires changing the what_char and the by_char variables
        what_char (str): Need to be specified if how == 'replace'.
            Here you can specify which character in the column name should be replaced.
        by_char (str): Need to be specified if how == 'replace'.
            Here you can specify by which character the character specified in the variable what
            in the column name should be replaced.

    Returns:
        pd.DataFrame: Output DF
    """

    assert how in ["to_snake", "replace"], f"The method how={how} not yet implemented!"

    if how == "replace":
        assert (
            what_char is not None
        ), "Please specify in the variable what, which character should be replaced!"
        assert (
            what_char is not None
        ), "Please specify in the variable by, by which character what should be replaced!"

    transform_func = {
        "to_snake": bf.snake_case,
        "replace": lambda col: col.replace(what_char, by_char),
    }[how]

    return dataframe.rename(
        columns={col_nm: transform_func(col_nm) for col_nm in dataframe.columns}
    )


def new_time_col(dataframe: pd.DataFrame, col: str, how: str) -> pd.DataFrame:
    """Create formatted time column

    Args:
        dataframe (pd.DataFrame): Input DF
        col (str): Name of the time column to be transformed
        how (str): Format of the new time column

    Returns:
        pd.DataFrame: _description_
    """

    # Configuration dictionary with key how
    # and value the corresponding expression
    _conf_dict = dict(year="%Y", month="%m", day="%d", weekday="%w")

    dataframe[how] = dataframe[col].apply(lambda x: x.strftime(_conf_dict[how]))
    return dataframe


def year_month_day_col(dataframe: pd.DataFrame, col: str) -> pd.DataFrame:
    """Create year, month, and date day column

    Args:
        dataframe (pd.DataFrame): Input DF
        col (str): Name of the given time column

    Returns:
        pd.DataFrame: Output DF
    """

    li_to_extract = ["year", "month", "day"]

    for how in li_to_extract:
        dataframe = new_time_col(dataframe=dataframe, col=col, how=how)
        dataframe[how] = dataframe[how].astype(int)

    return dataframe


def one_hot_encode(
    df_input: pd.DataFrame, li_one_hot: List[str], drop: Optional[bool] = True
) -> Dict[str, Union[pd.DataFrame, Dict[str, List[str]]]]:
    """Drop and one hot encode columns of a dataframe
    Args:
        df_input (pd.DataFrame): Input DF
        li_drop (List[str]): List of colums to drop
        li_one_hot (List[str]): List of colums to encode one-hot

    Returns:
        dict:
                df_result: Resulting dataframe
                dummies_dict: Dictionaries giving information about names of columns
                resulting from one hot encoding. Form:
                    one_hot_encoded_column: List of resulting columns
    """
    df_result = df_input
    dummies_dict = {}
    for feat in li_one_hot:
        dummies_df = pd.get_dummies(df_result[feat], prefix=feat)
        dummies_dict[feat] = list(dummies_df.columns)
        df_result = df_result.join(dummies_df)
    if drop:
        df_result = df_result.drop(columns=li_one_hot)
    return dict(df_result=df_result, dummies_dict=dummies_dict)


def drop_and_one_hot(
    df_input: pd.DataFrame, li_drop: List[str], li_one_hot: List[str]
) -> dict:
    """Drop and one hot encode columns of a dataframe

    Args:
        df_input (pd.DataFrame): Input DF
        li_drop (List[str]): List of colums to drop
        li_one_hot (List[str]): List of colums to encode one-hot

    Returns:
        dict:
                df_result: Resulting dataframe
                dummies_dict: Dictionaries giving information about names of columns
                resulting from one hot encoding. Form:
                    one_hot_encoded_column: List of resulting columns
    """
    df_result = df_input.drop(columns=li_drop)
    dummies_dict = {}
    for feat in li_one_hot:
        dummies_df = pd.get_dummies(df_result[feat], prefix=feat)
        dummies_dict[feat] = list(dummies_df.columns)
        df_result = df_result.join(dummies_df)
    return dict(df_result=df_result, dummies_dict=dummies_dict)


def create_time_cols(
    df: pd.DataFrame,
    time_col: str,
    to_create: Union[List[str], str],
    new_col_names: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Create additional time columns from a date_time column of a pandas DF
    Args:
        df (pd.DataFrame): DF to process
        time_col (str): Time column to be extracted from
        to_create (Union[List[str],str]): Columns to extract.
            Choices are: ["day_name","month","year","week","hour","day","month_year"]
        new_col_names (Optional[List[str]], optional): List of new column names. Defaults to None.

    Returns:
        pd.DataFrame: _description_
    """

    if to_create == "all":
        to_create = ["day_name", "month", "year", "week", "hour", "day", "month_year"]
    if new_col_names is None:
        new_col_names = to_create
    for method_extract, new_col_name in zip(to_create, new_col_names):
        df[new_col_name] = {
            "day_name": df[time_col].dt.day_name(),
            "month": df[time_col].dt.month,
            "year": df[time_col].dt.year,
            "week": df[time_col].apply(lambda x: x.isocalendar()[1]),
            "hour": df[time_col].dt.hour,
            "day": df[time_col].apply(lambda x: x.timetuple().tm_yday),
            "month_year": df[time_col].dt.strftime("%b %Y"),
        }[method_extract]
    return df


def concate_columns(
    df: pd.DataFrame, columns: List[str], new_col_nm: str
) -> pd.DataFrame:
    """Concatenate two columns into one as a dict with the first column as key
    and second column as value

    Args:
        df (pd.DataFrame): DF to process
        columns (List[str]): Columns to concate
        new_col_nm (str): Name of the resulting column

    Returns:
        pd.DataFrame: Result
    """
    list_df_cols = [df[col] for col in columns]
    df[new_col_nm] = [(key, val) for key, val in zip(*list_df_cols)]
    return df


def pad_complete_cat_value(
    df: pd.DataFrame,
    group_col: str,
    cat_col: str,
    value_for_completion: Optional[Union[int, float]] = 0,
) -> pd.DataFrame:
    """Complete df by non-existing subitems in a group. We assign then a default value for the non-existing items

    Args:
        df (pd.DataFrame): DF to process
        group_col (str): Column name of the group
        cat_col (str): Column name of the item
        value_for_completion (Optional[Union[int,float]], optional): Which value to be assigned if non-existent. Defaults to 0.

    Returns:
        pd.DataFrame: Resulting DF
    """

    df_group_cat_complete = pd.DataFrame(
        list(product(df[group_col].unique(), df[cat_col].unique())),
        columns=[group_col, cat_col],
    )
    return df.merge(df_group_cat_complete, on=[group_col, cat_col], how="right").fillna(
        value_for_completion
    )
