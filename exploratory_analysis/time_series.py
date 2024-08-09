"""
Module for time series analysis
"""
from typing import Dict, List, Optional, Union

import pandas as pd
import plotly.express as px
from plotly.graph_objects import Figure


def plot_time_series_plotly(
    df_time_series: pd.DataFrame,
    time_col: str,
    val_cols: Union[List[str], str],
    display_name_val_cols: Optional[Dict[str, str]] = None,
    hover_nm_x_col: Optional[str] = None,
    hover_nm_y_col: Optional[str] = None,
    plot_title: Optional[str] = None,
    xaxis_title: Optional[str] = None,
    yaxis_title: Optional[str] = None,
    legend_title: Optional[str] = None,
    other_hovers_in_plot: Optional[Dict[str, str]] = None,
) -> Figure:
    """Function for plotting time-series in plotly
    Args:
        df_time_series (pd.DataFrame): DF with time-Series to plot
        time_col (str): Name of the time Column
        val_cols (Union[List[str], str]): Name of the column(s),
            wherein the time series to be plotted is given
        display_name_val_cols (Optional[Dict[str, str]], optional): Display names of the values.
            Defaults to None.
        hover_nm_x_col (Optional[str], optional): Display name of the x-Axis in Hover.
            Defaults to None.
        hover_nm_y_col (Optional[str], optional): Display name of the y-Axis in Hover
            Defaults to None.
        plot_title (Optional[str], optional): Title of the plot.
            Defaults to None.
        xaxis_title (Optional[str], optional): Title of the xaxis.
            Defaults to None.
        yaxis_title (Optional[str], optional): Title of the yaxis.
            Defaults to None.
        legend_title (Optional[str], optional): Title of the legend.
            Defaults to None.
        other_hovers_in_plot (Optional[Dict[str, str]], optional): Column names of the values
            to be shown in t
    Returns:
        Figure: Resulting plotly figure
    """

    if display_name_val_cols is not None:
        df_time_series = df_time_series.rename(columns=display_name_val_cols)

    plotly_line_arguments = {
        "data_frame": df_time_series, "x": time_col, "y": val_cols}

    if other_hovers_in_plot is not None:
        plotly_line_arguments = {
            **plotly_line_arguments,
            "custom_data": list(other_hovers_in_plot.keys()),
        }

    fig = px.line(**plotly_line_arguments)
    fig.update_layout(
        title=plot_title,
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        legend_title=legend_title,
        autosize=True,
    )

    _hover_nm_x_col = hover_nm_x_col if hover_nm_x_col is not None else time_col
    _hover_nm_y_col = hover_nm_y_col if hover_nm_y_col is not None else "Values"

    # Add x and y values in hover
    hover_text = f"{_hover_nm_x_col}: %{{x}}<br>{_hover_nm_y_col}: %{{y}}<br>"

    if other_hovers_in_plot is not None:
        for _ind, _display_text in enumerate(other_hovers_in_plot.values()):
            hover_text += f"{_display_text}: %{{customdata[{_ind}]}}<br>"

    fig.update_traces(hovertemplate=hover_text)
    fig.show()
    return fig


# Functions for simple aggregation of time-series an plotting with smoothing functionalities
def extract_time_series(
    raw_data: pd.DataFrame,
    col_time: str,
    col_val: str,
    norm_val: Optional[float] = None,
    smooth_lengths: Optional[List[int]] = None,
    display_nm_col_val: Optional[str] = None,
    display_nm_col_smoothed: Optional[List[str]] = None,
    aggregator_data: Optional[str] = "sum",
    method_smoother: Optional[str] = "ewm",
    aggregator_smoother: Optional[str] = "mean",
) -> pd.DataFrame:
    """Function for extracting time related quantity from a DF
    by grouping. Moreover, one can provide smoothed time-series (ewm),
    by inserting a list of the length of smoothing windows in smooth_lengths variable

    Args:
        raw_data (pd.DataFrame): DF from which the time-series should be extracted
        col_time (str): Column name of the time feature
        col_val (str): Column name of the value to be extracted
        norm_val (Optional[float], optional): Desired normalization of the time-series value.
            Time-series values will be then divided by the given value
            Defaults to None, here no normalization is desired.
        smooth_lengths (Optional[List[int]], optional): Length of list of smoothing windows,
           for each value in the list, a smoothed time-series will be generated,
           with smoothing window equal to the value. Smoothing is done by EWM
           Defaults to None, here no smoothing is provided
        display_nm_col_val (Optional[str], optional): Display name of the value column.
            Defaults to None.
        display_nm_col_smoothed (Optional[List[str]], optional): Display names of
            the smoothed Columns.
            Defaults to None.
        aggregator_data (Optional[str], optional): Method for aggregating the data.
            Defaults to "sum".
        method_smoother (Optional[str], optional): Method for creating sliding window.
            Options are exponentially weighted (ewm) or usual sliding window (rolling)
            Defaults to "ewm".
        aggregator_smoother (Optional[str], optional): Method for aggregating data
                in sliding window.
            Defaults to "mean".        

    Returns:
        pd.DataFrame: Time-series resulting from aggregating the raw data and eventual smoothing
    """

    _df = getattr(raw_data.groupby([col_time])[col_val], aggregator_data)()

    if norm_val is not None:
        _df = _df / norm_val

    _df = pd.DataFrame(_df).reset_index()

    if display_nm_col_smoothed is None:
        display_nm_col_smoothed = [
            f"{col_val}__smoothed_{_length}" for _length in smooth_lengths
        ]

    for _ind, _length in enumerate(smooth_lengths):
        _arg_smooth = {
            "ewm": {"span": _length},
            "rolling": {"window": _length, "min_periods": 1}

        }[method_smoother]
        _smooth = getattr(_df[col_val], method_smoother)(**_arg_smooth)
        _df[display_nm_col_smoothed[_ind]] = getattr(
            _smooth, aggregator_smoother)()

    if display_nm_col_val is not None:
        _df = _df.rename(columns={col_val: display_nm_col_val})

    return _df


def plot_smoothed_time_series_agg(
    raw_data: pd.DataFrame,
    col_time: str,
    col_val: str,
    smooth_lengths: List[int],
    display_nm_col_val: str,
    display_nm_col_smoothed: List[str],
    plot_title: str,
    xaxis_title: str,
    yaxis_title: str,
    legend_title: str,
    display_nm_col_time: str,
    extract_weekday: Optional[bool] = False,
    aggregator_data: Optional[str] = "sum",
    norm_val: Optional[int] = None,
    method_smoother: Optional[str] = "ewm",
    aggregator_smoother: Optional[str] = "mean",
) -> Dict[str, Union[pd.DataFrame, Figure]]:
    """Function for plotting the time-series,
    resulting by aggregation and subsequent smoothing (EWM)

    Args:
        raw_data (pd.DataFrame):DF from which the time-series should be extracted
        col_time (str): Column name of the time feature
        col_val (str): Column name of the value to be extracted
        smooth_lengths (List[int]): Length of list of smoothing windows,
           for each value in the list, a smoothed time-series will be generated,
           with smoothing window equal to the value. Smoothing is done by EWM.
        display_nm_col_val (str): Display name of the value column.
        display_nm_col_smoothed (List[str]): Display names of
            the smoothed Columns.
        plot_title (str): Title of the plot
        xaxis_title (str): Title of the x-Axis
        yaxis_title (str): Title of the y-Axis
        legend_title (str): Title of the legend
        display_nm_col_time (str): Display name of the time Column
        extract_weekday (Optional[bool], optional): Whether to extract weekday in the data
            and add to hover. Defaults to False.
        aggregator_data (Optional[str], optional): Method for data aggregation. Defaults to "sum".
        norm_val (Optional[int], optional): Desired normalization of the time-series value.
            Time-series values will be then divided by the given value
            Defaults to None, here no normalization is desired.
        method_smoother (Optional[str], optional): Method for creating sliding window.
            Options are exponentially weighted (ewm) or usual sliding window (rolling)
            Defaults to "ewm".
        aggregator_smoother (Optional[str], optional): Method for aggregating data
                in sliding window.
            Defaults to "mean".

    Returns:
        Dict[str,Union[pd.DataFrame,Figure]]: Results - DF of the smoothed time series
            and plotly figure
    """
    _df = extract_time_series(
        raw_data=raw_data,
        col_time=col_time,
        col_val=col_val,
        norm_val=norm_val,
        smooth_lengths=smooth_lengths,
        display_nm_col_val=display_nm_col_val,
        display_nm_col_smoothed=display_nm_col_smoothed,
        aggregator_data=aggregator_data,
        method_smoother=method_smoother,
        aggregator_smoother=aggregator_smoother
    )

    _arg_plot = {
        "df_time_series": _df,
        "time_col": col_time,
        "val_cols": [display_nm_col_val, *display_nm_col_smoothed],
        "hover_nm_x_col": display_nm_col_time,
        "hover_nm_y_col": display_nm_col_val,
        "plot_title": plot_title,
        "xaxis_title": xaxis_title,
        "yaxis_title": yaxis_title,
        "legend_title": legend_title,
    }

    if extract_weekday:
        _df["weekday"] = _df["date"].apply(lambda x: x.strftime("%A"))
        _arg_plot = {**_arg_plot, "other_hovers_in_plot": {"weekday": "Day"}}

    fig = plot_time_series_plotly(**_arg_plot)

    return {"result_df": _df, "plotly_figure": fig}


def extract_cat_time_series(
    raw_data: pd.DataFrame,
    col_time: str,
    col_cat: str,
    col_val: str,
    norm_val: Optional[float] = None,
    smooth_length: Optional[int] = None,
    aggregator_data: Optional[str] = "sum",
    fill_na: Optional[bool] = False,
) -> pd.DataFrame:
    """
    Function for extracting time series of some predefined category
    Args:
        raw_data (pd.DataFrame): DF, from where the time series is extracted
        col_time (str): Column name of the time
        col_cat (str): Column name of the categories
        col_val (str): Column name of the values
        norm_val (Optional[float], optional): Desired normalization of the time-series value.
            Time-series values will be then divided by the given value
            Defaults to None, here no normalization is desired.
        smooth_length (Optional[int], optional): Length of of the smoothing window,
            smoothing is done by EWM.
            Defaults to None, here no smoothing is provided. Defaults to None.
        aggregator_data (Optional[str], optional): Method for the data aggregation respective to time.
            Defaults to "sum".
        fill_na (Optional[bool], optional): Whether to fill na with 0.
            Defaults to False.

    Returns:
        pd.DataFrame: Resulting DF
    """

    _df = getattr(
        raw_data.groupby([col_time, col_cat])[col_val], aggregator_data
    )().reset_index()

    if norm_val is not None:
        _df[col_val] = _df[col_val] / norm_val

    _df_cat_series = _df.pivot(index=col_time, columns=col_cat, values=col_val)

    if fill_na:
        _df_cat_series = _df_cat_series.fillna(0)

    if smooth_length is not None:
        for _col in _df[col_cat].unique():
            _df_cat_series[_col] = _df_cat_series[_col].ewm(
                span=smooth_length).mean()
    return _df_cat_series


def plot_cat_time_series_plotly(
    raw_data: pd.DataFrame,
    col_time: str,
    col_cat: str,
    col_val: str,
    plot_title: str,
    x_axis_title: str,
    y_axis_title: str,
    legend_title: str,
    display_nm_col_time: str,
    display_nm_col_val: str,
    aggregator_data: Optional[str] = "sum",
    extract_weekday: Optional[bool] = True,
    norm_val: Optional[int] = None,
    smooth_length: Optional[int] = None,
) -> Dict[str, Union[pd.DataFrame, Figure]]:
    """Plotting time-series with some categories
     raw_data (pd.DataFrame): DF, from where the time series is extracted
        col_time (str): Column name of the time
        col_cat (str): Column name of the categories
        col_val (str): Column name of the values
        norm_val (Optional[float], optional): Desired normalization of the time-series value.
            Time-series values will be then divided by the given value
            Defaults to None, here no normalization is desired.
        smooth_length (Optional[int], optional): Length of of the smoothing window,
            smoothing is done by EWM.
            Defaults to None, here no smoothing is provided. Defaults to None.
        aggregator_data (Optional[str], optional): Method for the data aggregation respective to time.
            Defaults to "sum".
        fill_na (Optional[bool], optional): Whether to fill na with 0.
            Defaults to False.

    Returns:
        Dict[str, Union[pd.DataFrame, Figure]]: Result with DF and figure
    """
    _df_cat_series = extract_cat_time_series(
        raw_data=raw_data,
        col_time=col_time,
        col_cat=col_cat,
        col_val=col_val,
        fill_na=True,
        aggregator_data=aggregator_data,
        norm_val=norm_val,
        smooth_length=smooth_length,
    )
    _col_cats = list(_df_cat_series.columns)
    _df_cat_series = _df_cat_series.reset_index()

    if extract_weekday:
        _df_cat_series["Weekday"] = _df_cat_series["date"].apply(
            lambda x: x.strftime("%A")
        )

    _fig = plot_time_series_plotly(
        df_time_series=_df_cat_series,
        time_col=col_time,
        val_cols=_col_cats,
        hover_nm_x_col=display_nm_col_time,
        hover_nm_y_col=display_nm_col_val,
        other_hovers_in_plot={"Weekday": "Weekday"},
        plot_title=plot_title,
        xaxis_title=x_axis_title,
        yaxis_title=y_axis_title,
        legend_title=legend_title,
    )
    return {"df": _df_cat_series, "fig": _fig}
