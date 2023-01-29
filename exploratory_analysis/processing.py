"""
Module for processing data.
Functionalities:
    * Show statistics by grouping
"""

from typing import List, Dict, Union, Optional
from functools import reduce
import pandas as pd
from exploratory_analysis.preprocessing import pad_complete_cat_value, concate_columns
from exploratory_analysis.basic_functions import to_int, adjust_display_names


def group_and_fill(
    df: pd.DataFrame,
    cat_nm: str,
    sub_cat_nm: str,
    cols_values: List[str],
    value_for_completion: Union[str, int, float],
    aggregator: str,
    additional_columns: Union[List[str], str],
) -> pd.DataFrame:
    """Group a data frame respective a category then fill the category with complete list of subcategories

    Args:
        df (pd.DataFrame): DF to be processed
        cat_nm (str): Category column name
        sub_cat_nm (str): Subcategory column names
        cols_values (List[str]): Column names of the values to aggregate
        value_for_completion (Union[str,int,float]): Default values for the missing subcategories
        aggregator (str): Aggregation method
        additional_columns (Union[List[str],str]): additional columns to add to the result table from the original table.
            Need to be 1-1 to the cat_nm column

    Returns:
        pd.DataFrame: _description_‚
    """

    additional_columns = (
        [additional_columns]
        if isinstance(additional_columns, str)
        else additional_columns
    )

    # Check whether additional columns is 1-1 to cat_nm
    _df_additional = df[[*additional_columns, cat_nm]].drop_duplicates()

    assert (
        _df_additional.groupby(cat_nm).nunique().max().max() == 1
    ), f"the additional columns {additional_columns} are not 1-1 to cat_nm {cat_nm}"

    _df = getattr(
        df.groupby([cat_nm, sub_cat_nm], as_index=False)[cols_values], aggregator
    )()
    _df = pad_complete_cat_value(
        df=_df,
        group_col=cat_nm,
        cat_col=sub_cat_nm,
        value_for_completion=value_for_completion,
    )

    return _df.merge(_df_additional, on=cat_nm)


def separate_df_by_cat(
    df: pd.DataFrame, cat: str
) -> Dict[Union[str, int, float], pd.DataFrame]:
    """Separate DF by categories

    Args:
        df (pd.DataFrame): DF to process
        cat (str): category column name

    Returns:
        Dict[Union[str,int,float],pd.DataFrame]: Result Dict
    """
    # Collect possible subcategories of cat_tables in DF
    cat_elements = df[cat].unique()

    # Initialize the tables in a dict
    dict_df = {
        cat_choice: df[df[cat] == cat_choice].reset_index(drop=True).drop(columns=[cat])
        for cat_choice in cat_elements
    }

    return dict_df


def compute_stat_aggregate(
    df: pd.DataFrame,
    cats: List[str],
    data_distinguisher: str,
    col_val: str,
    aggregator: str,
    stat_method: str,
) -> pd.DataFrame:
    """Aggregate data then compute stats

    Args:
        df (pd.DataFrame): DF to be processed
        cats (List[str]): Categories for which statistics are computed
        data_distinguisher (str): distinguisher of the data. Aggregation is respective to this columns
        col_val (str): Columns containing values to be analyzed
        aggregator (str): Method for aggregation
        stat_method (str): Method for statistics

    Returns:
        pd. DataFrame: Result DF
    """
    # Group data respective to a distinguisher
    _df = getattr(
        df.groupby([*cats, *data_distinguisher], as_index=False)[col_val],
        aggregator,
    )()

    # Compute statistics respective to the desired columns and row categories
    _df = getattr(_df.groupby(cats)[col_val], stat_method)()

    # Formatting the numbers
    _df = _df.apply(lambda x: str(int(x)) if x.is_integer() else x)

    return _df


def stat_agg(
    df: pd.DataFrame,
    cat_tables: str,
    cat_rows: str,
    cat_columns: str,
    data_distinguisher: Union[str, List[str]],
    col_data: str,
    aggregator: str,
    stat_method: str,
    cat_columns_name: Optional[str] = None,
    cat_rows_name: Optional[str] = None,
    quantity_name: Optional[str] = None,
    nan_name: Optional[str] = "No Data",
    order_cat_columns: Optional[str] = None,
) -> pd.DataFrame:
    """
    Statistically aggregate data respective two three categories:
        * First the data is grouped by the categories and additional data_distinguisher feature
            Example: Categorize sales data by year, hour, and week day of the purchase with the purchase-id as distinguisher
                as in a purchase-id may have several rows in a data (e.g. receipt-id)
        * Second distinguished grouped data is aggregated respective a method. E.g. sum or nunique
            Example: We collect the total sales in a purchase-id by using the method sum
        * Third aggregated data are computed statistically respective to the groups. E.g. compute the median
    --> Tables of the statistics respective to the categories are created
    Args:
        df (pd.DataFrame): Dataframe
        cat_tables (str): Categories of the resulting tables, e.g. year
        cat_rows (str): Categories of the rows, e.g. hour
        cat_columns (str): Categories of the columns, e.g. week day
        data_distinguisher (Union[str,List[str]]): Distinguisher of the data, e.g. purchase-id
        col_data (str): Quantity to analyze, e.g. Gross sales
        aggregator (str): Method to aggregate the data. E.g. sum, nunique, ...
        stat_method (str): Method of the statistics to compute. E.g. median, average, ...
        cat_columns_name (str): Display name of the columns title
        cat_rows_name (str): Display name of the rows title
        quantity_name (str): Display name of the quantity
        nan_name (Optional[str], optional): _description_. Defaults to "No Data".

    Returns:
        pd.DataFrame: Multiindex DF
    """

    cat_columns_name = cat_columns if cat_columns_name is None else cat_columns_name
    cat_rows_name = cat_rows if cat_rows_name is None else cat_rows_name
    quantity_name = (
        f"{stat_method} {col_data} per {data_distinguisher} ({aggregator})"
        if quantity_name is None
        else quantity_name
    )

    # Initialize the tables in a dict
    results = separate_df_by_cat(df=df, cat=cat_tables)

    # Needed in order to handle the possibility of string and list of strings input
    if isinstance(data_distinguisher, str):
        data_distinguisher = [data_distinguisher]

    results = map(
        lambda x: (
            x[0],
            compute_stat_aggregate(
                df=x[1],
                cats=[cat_columns, cat_rows],
                data_distinguisher=data_distinguisher,
                col_val=col_data,
                aggregator=aggregator,
                stat_method=stat_method,
            ),
        ),
        results.items(),
    )

    # Unstack multi index pandas series to multi index DF
    dict_to_list_df = map(
        lambda dict_item: pd.DataFrame(dict_item[1].rename(dict_item[0])), results
    )

    # Add the DFs
    result_df = (
        reduce(lambda x, y: x.join(y, how="outer"), dict_to_list_df)
        .unstack(level=0)
        .fillna(nan_name)
    )

    # # Make adjustment for display
    # result_df.columns = result_df.columns.set_names(cat_rows_name, level=1)
    # result_df.index = result_df.index.set_names(cat_columns_name)
    # if order_cat_columns is not None:
    #     result_df = result_df.reindex(columns=order_cat_columns, level=1)

    return adjust_display_names(
        df=result_df, 
        cat_rows_name=cat_rows_name, cat_columns_name=cat_columns_name, 
        order_cat_columns=order_cat_columns
        )

def agg_cat_stat_in_cells(
    df: pd.DataFrame,
    cat_tables: str,
    cat_row: str,
    cat_col: str,
    cat_in_cell_col: str,
    col_val: str,
    stat_method: str,
    cat_columns_name: Optional[str] = None,
    cat_rows_name: Optional[str] = None,
    nan_name: Optional[str] = "No Data",
    order_cat_columns: Optional[str] = None,
) -> pd.DataFrame:
    _df = getattr(
        df.groupby([cat_tables, cat_col, cat_row, cat_in_cell_col], as_index=False)[
            col_val
        ],
        stat_method,
    )()

    _dict_df = separate_df_by_cat(df=_df, cat=cat_tables)

    _li_df_concated = map(
        lambda x: (x[0], x[1][x[1][col_val] > 0].reset_index(drop=True)),
        _dict_df.items(),
    )

    _li_df_concated = map(
        lambda x: (
            x[0],
            concate_columns(x[1], [cat_in_cell_col, col_val], x[0]).drop(
                columns=[cat_in_cell_col, col_val]
            ),
        ),
        _li_df_concated,
    )

    _li_df_concated = map(
        lambda x: x[1].groupby([cat_col, cat_row])[x[0]].apply(list), _li_df_concated
    )

    _li_df_concated = [
        _df_concated.apply(lambda x: list(map(lambda y: f"{y[0]}: {to_int(y[1])}", x)))
        for _df_concated in _li_df_concated
    ]

    _li_df_concated = [
        _df_concated.apply(", ".join) for _df_concated in _li_df_concated
    ]

    _li_df_concated = [
        pd.DataFrame(_df_concated).unstack(level=0) for _df_concated in _li_df_concated
    ]

    result_df=reduce(lambda x, y: x.join(y, how="outer"), _li_df_concated).fillna(
        nan_name
    )

    return adjust_display_names(
        df=result_df, 
        cat_rows_name=cat_rows_name, cat_columns_name=cat_columns_name, 
        order_cat_columns=order_cat_columns
        )
