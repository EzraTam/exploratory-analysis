"""Preprocessing Module
"""

from typing import List, Dict, Union
import pandas as pd
import exploratory_analysis.basic_functions as bf


def transform_col_nm_to_snake(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Transform column names to snake case

    Args:
        dataframe (pd.DataFrame): Input DF

    Returns:
        pd.DataFrame: Output DF
    """
    return dataframe.rename(
        columns={col_nm: bf.snake_case(col_nm) for col_nm in dataframe.columns}
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
    df_input: pd.DataFrame, li_one_hot: List[str]
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
