"""Module for basic functions
"""

from typing import Union, List, Optional, Any
from re import sub
from pandas import Series, DataFrame


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


def return_obj_val_in_list_order(object_to_order: Series, key_list: List) -> List:
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


def revert_key_val(dict_input: Union[dict, Series]) -> dict:
    """Revert key and value in a dictionary
    Args:
        dict_input (Union[dict,Series]): Input dictionary

    Returns:
        dict: Input dictionary with key and value reverted
    """
    return {val: ind for ind, val in dict_input.items()}

def to_int(val: float) -> Union[int, float]:
    """Transform an integer in float to int.
    Else the value remains in float.
    Args:
        val (float): Value to transform

    Returns:
        Union[int, float]: Result
    """
    return int(val) if val.is_integer() else val

def adjust_display_names(
    df:DataFrame, cat_rows_name: str, cat_columns_name: str, 
    order_cat_columns: Optional[str] = None
    )->DataFrame:
    """Adjust display name in pandas multiindex DF
    """

    # Make adjustment for display
    df.columns = df.columns.set_names(cat_columns_name, level=1)

    df.index = df.index.set_names(cat_rows_name)

    if order_cat_columns is not None:
        df = df.reindex(columns=order_cat_columns, level=1)

    return df
