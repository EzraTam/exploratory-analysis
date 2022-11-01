"""Module for basic functions
"""

from typing import Union
from re import sub
from pandas import Series


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
