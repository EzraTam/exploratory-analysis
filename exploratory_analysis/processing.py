from typing import List, Dict, Union, Optional
import pandas as pd

def stat_agg(
    df:pd.DataFrame, 
    cat_tables:str, cat_rows:str, cat_columns:str, 
    data_distinguisher: Union[str,List[str]], col_data: str, 
    aggregator:str, stat_method:str, 
    cat_columns_name: Optional[str]=None, cat_rows_name: Optional[str]=None, quantity_name: Optional[str]=None,
    nan_name: Optional[str] = "No Data"
    )->Dict[Union[str,int,float],pd.DataFrame]:
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
        Dict[str,pd.DataFrame]: Dict of tables with key equal to subcategories in cat tables
    """

    cat_columns_name = cat_columns if cat_columns_name is None else cat_columns_name
    cat_rows_name = cat_rows if cat_rows_name is None else cat_rows_name
    quantity_name = f"{stat_method} {col_data} per {data_distinguisher} ({aggregator})" if quantity_name is None else quantity_name
    
    # Collect possible subcategories of cat_tables in DF
    cat_tables_elements= df[cat_tables].unique()

    # Initialize the tables in a dict
    results={cat_tables_choice: df[df[cat_tables]==cat_tables_choice] for cat_tables_choice in cat_tables_elements}

    # Needed in order to handle the possibility of string and list of strings input
    if isinstance(data_distinguisher, str):
        data_distinguisher=[data_distinguisher]

    for cat_tables_choice in cat_tables_elements:

        _df=results[cat_tables_choice]

        # Group data respective to a distinguisher
        _df_number_data=getattr(_df.groupby([cat_columns,cat_rows,*data_distinguisher],as_index=False)[col_data],aggregator)()

        # Compute statistics respective to the desired columns and row categories 
        _df_number_data=getattr(_df_number_data.groupby([cat_columns, cat_rows])[col_data],stat_method)()

        # Formatting the numbers
        _df_number_data=_df_number_data.apply(lambda x: str(int(x)) if x.is_integer() else x)

        # From a multiindex to a row-column table and rename nan values
        _result=pd.DataFrame(_df_number_data).unstack(level=0).fillna(nan_name)

        # Set the display names
        _result.columns=_result.columns.set_levels([quantity_name],level=0)
        _result.columns=_result.columns.set_names(cat_columns_name,level=1)
        _result.index=_result.index.set_names(cat_rows_name)

        results[cat_tables_choice]=_result
    
    return results