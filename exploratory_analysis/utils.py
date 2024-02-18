"""Module for Utils"""

from typing import Dict, List, Union
from itertools import product
import pandas as pd

def fill_df_full_cat(df:pd.DataFrame,cats:Dict[str,List[Union[str,int,float]]])->pd.DataFrame:
    """Function for completing a dataframe by full categories and set values equal
    to zero if no value exist.

    Args:
        df (pd.DataFrame): DF to complete
        cats (Dict[str,List[Union[str,int,float]]]): Categories to complete
            form --> category_key: full list of categories
    Returns:
        pd.DataFrame: _description_
    """
    _df_fill=pd.DataFrame(list(product(*cats.values())),columns=cats.keys())
    return df.merge(_df_fill,on=list(cats.keys()),how="right").fillna(0)
