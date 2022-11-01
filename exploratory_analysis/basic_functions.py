from re import sub
from pandas import Series
from typing import Union


def snake_case(s):
    return "_".join(
        sub(
            "([A-Z][a-z]+)", r" \1", sub("([A-Z]+)", r" \1", s.replace("-", " "))
        ).split()
    ).lower()

def revert_key_val(dict_input:Union[dict,Series])->dict:
    return {val:ind for ind,val in dict_input.items()}