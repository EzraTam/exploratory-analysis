"""Module for basic functions
"""
from collections import defaultdict
from typing import Union, List, Optional, Any, Tuple, Dict
from re import sub
import pandas as pd


def distinct(input_list: List) -> List:
    """Return list of distinct element of a list"""
    return list(set(input_list))


def raise_error(condition: bool, error_class: Any, msg: Optional[str] = "") -> None:
    """Function for raising an error

    Args:
        condition (bool): Condition for an error
        error_class (Any): Class of the corresponding error
        msg (Optional[str], optional): Message to throw if error

    Raises:
        error_class: _description_
    """
    if condition:
        raise error_class(msg)


def set_zero_if_below(list_input: List[float], limit: float) -> List[float]:
    """Given a list of numbers. Set all element less than a limit to zero

    Args:
        list_input (List[float]): Input list
        limit (float): Limit value

    Returns:
        List[float]: Resulting List
    """
    return [val if val >= limit else 0 for val in list_input]


def return_obj_val_in_list_order(object_to_order: pd.Series, key_list: List) -> List:
    """Function for returning values of an object in a given key order

    Args:
        object_to_order (Series): Object with values to order
        key_list (List): Key to order

    Returns:
        List: Resulting list
    """
    return [object_to_order[key] for key in key_list]


def snake_case(string: str) -> str:
    """Transform a string into snake_case
    Args:
        string (str): Input string

    Returns:
        str: Input string in snake case
    """
    return "_".join(
        sub(
            "([A-Z][a-z]+)", r" \1", sub("([A-Z]+)", r" \1", string.replace("-", " "))
        ).split()
    ).lower()


def revert_key_val(dict_input: Union[dict, pd.Series]) -> dict:
    """Revert key and value in a dictionary
    Args:
        dict_input (Union[dict,Series]): Input dictionary

    Returns:
        dict: Input dictionary with key and value reverted
    """
    return {val: ind for ind, val in dict_input.items()}


def to_int(val: float, round_num: Optional[int] = None) -> Union[int, float]:
    """Transform an integer in float to int.
    Else the value remains in float.
    Args:
        val (float): Value to transform
        round_num (int): To how many digit should it be rounded

    Returns:
        Union[int, float]: Result
    """
    if val.is_integer():
        return int(val)

    if round_num:
        return round(val, round_num)

    return val


def adjust_display_names(
    df: pd.DataFrame,
    cat_rows_name: str,
    cat_columns_name: str,
    order_cat_columns: Optional[str] = None,
) -> pd.DataFrame:
    """Adjust display name in pandas multiindex DF"""

    # Make adjustment for display
    df.columns = df.columns.set_names(cat_columns_name, level=1)

    df.index = df.index.set_names(cat_rows_name)

    if order_cat_columns is not None:
        df = df.reindex(columns=order_cat_columns, level=1)

    return df


def df_to_list(df: pd.DataFrame) -> List[Tuple[Any]]:
    """Function for extracting DF to list of tuples"""
    return list(df.itertuples(index=False, name=None))


def extract_from_tuple(list_tuple, idx) -> List[Any]:
    """Function for extracting an element of tuples list"""
    return [_tuple[idx] for _tuple in list_tuple]


def merge_dicts_into_list_values(list_dicts: List[Dict]) -> Dict[str, List]:
    """Merge list of dicts with key and list of values"""
    _dd = defaultdict(list)
    for _dict in list_dicts:
        for key, value in _dict.items():
            _dd[key].extend(value)

    return _dd


# Functions for sorting


def sort_by_list(list_input: List, order: List) -> List:
    """Sort a list by a given list of order

    Args:
        list_input (List): List to sort
        order (List): Order of the elements

    Returns:
        List: Ordered List
    """
    _dict_key_order_idx = {_key: _idx for _idx, _key in enumerate(order)}
    _dict_key_order_idx_inv = {_idx: _key for _key, _idx in _dict_key_order_idx.items()}

    _result = sorted(map(lambda _key: _dict_key_order_idx[_key], list_input))
    return list(map(lambda idx: _dict_key_order_idx_inv[idx], _result))


def sort_week_day(list_week_day: List[str]) -> List[str]:
    """Sort a list of week days"""
    _week_days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]

    return sort_by_list(list_week_day, _week_days)


def sort_month(list_months: List[str]) -> List[str]:
    """Sort a list of months"""
    _months = [
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December",
    ]
    return sort_by_list(list_months, _months)
