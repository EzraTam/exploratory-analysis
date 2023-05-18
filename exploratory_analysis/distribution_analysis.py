"""Module for exploring the distribution
of data
"""
from typing import Optional, Dict, List

import numpy as np
import pandas as pd
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
from scipy.signal import argrelextrema

import plotly.express as px



def find_local_kde_extrema(
    data: pd.Series,
    resolution: int,
    how: Optional[str] = "maxima",
    bandwidth_kdw: Optional[float] = 0.6,
    kernel_kde: Optional[str] = "gaussian",
    optimize_bandwidth: Optional[bool] = False
) -> Dict[str, np.array]:
    """Function for finding local extrema in the KDE-smoothed data distribution.

    Args:
        data (pd.Series): Data tto explore
        resolution (int): The resolution of Values - linearly interpolated between
            min data value and max data value with number equal to resolution
        how (Optional[str], optional): Whether we want to find local
            maxima or minima. Defaults to "maxima".
        bandwidth_kdw (Optional[float], optional): Bandwidth of the KDE. Defaults to 0.6.
        kernel_kde (Optional[str], optional): Kernel of the KDE. Defaults to "gaussian".

    Returns:
        Dict[str,np.array]: {
            "local_extrema": The local extremum - Most likely occurence of the data,
            "kde_values": Value of the corresponding KDE
        }
    """
    _data_for_kde=np.array(data).reshape(-1, 1)
    # Fit a kernel density estimation to the data
    if optimize_bandwidth:
        grid = GridSearchCV(KernelDensity(kernel=kernel_kde),
                        {'bandwidth': np.linspace(0.1, 1.0, 30)},
                        cv=20) # 20-fold cross-validation
        grid.fit(_data_for_kde)
        bandwidth_kdw=grid.best_params_["bandwidth"]
        print(f"Obtained bandwidth after optimization: {bandwidth_kdw}")


    kde = KernelDensity(kernel=kernel_kde, bandwidth=bandwidth_kdw).fit(
        _data_for_kde
    )

    # Generate x values for evaluating the KDE
    x_values = np.linspace(min(data), max(data), resolution)

    kde_values = np.exp(kde.score_samples(x_values.reshape(-1, 1)))
    _comparator = {"maxima": np.greater, "minima": np.less}[how]
    _pos_extrema = argrelextrema(data=kde_values, comparator=_comparator)
    return {
        "local_extrema": x_values[_pos_extrema],
        "kde_values": kde_values[_pos_extrema],
    }


def plot_violin_binary(
    data_frame: pd.DataFrame,
    val_col: str,
    cats: List[str],
    prefix: Optional[str] = "",
    annot_prefix: Optional[str] = "",
    optimize_bandwidth: Optional[bool] = False
) -> None:
    """Violin Plot of boolean categorical variables
    Local maxima of KDE is also given

    Args:
        data_frame (pd.DataFrame): Data
        val_col (str): Column name of the values
        cats (List[str]): Categories to be plotted
        prefix (Optional[str], optional): Prefix for displaying the categories. Defaults to "".
        annot_prefix (Optional[str], optional): Annotation for the prefix. Defaults to "".

    Returns:
        None: None
    """
    for cat in cats:

        cat_col = prefix + cat

        _kde_extrema = find_local_kde_extrema(
            data_frame[data_frame[cat_col] == 1][val_col], resolution=100,optimize_bandwidth=optimize_bandwidth
        )

        _df_fig = data_frame.rename(columns={cat_col: cat})
        _df_fig[cat] = _df_fig[cat].apply(lambda x: "No" if x == 0 else "Yes")

        _arg_violin = dict(
            data_frame=_df_fig,
            x=val_col,
            color=cat,
            range_x=[0, data_frame[val_col].max()],
            points="all",
            box=True,
            category_orders={cat: ["No", "Yes"]},
        )

        fig = px.violin(**_arg_violin)

        for _loc_max, _size in zip(
            _kde_extrema["local_extrema"], _kde_extrema["kde_values"]
        ):
            _color = "red"
            _opacity = _size / sum(_kde_extrema["kde_values"])
            _annotation_text = annot_prefix + str(np.round(_loc_max, 2))
            fig.add_vline(
                x=_loc_max,
                line_width=10 * _size,
                line_color=_color,
                opacity=_opacity,
                annotation_text=_annotation_text,
                annotation=dict(opacity=_opacity, font=dict(color=_color, size=8)),
            )
        fig.show()

    return None


def plot_violin_with_kde(
    data_frame: pd.DataFrame,
    val_col: str,
    cats: List[str],
    prefix: Optional[str] = "",
    annot_prefix: Optional[str] = "",
    resolution: Optional[int]=100,
    optimize_bandwidth: Optional[bool]= False,
    bandwidth_kdw: Optional[float] = 0.6,
    kernel_kde: Optional[str] = "gaussian"

) -> None:
    """Violin Plot of boolean categorical variables
    Local maxima of KDE is also given

    Args:
        data_frame (pd.DataFrame): Data
        val_col (str): Column name of the values
        cats (List[str]): Categories to be plotted
        prefix (Optional[str], optional): Prefix for displaying the categories. Defaults to "".
        annot_prefix (Optional[str], optional): Annotation for the prefix. Defaults to "".

    Returns:
        None: None
    """
    for cat in cats:
        cat_col = prefix + cat
        _df = data_frame.rename(columns={cat: cat_col})
        _cats = _df[cat_col].unique()
        _kde_extrema = {
            _cat: find_local_kde_extrema(
                _df[_df[cat_col] == _cat][val_col], resolution=resolution,
                bandwidth_kdw=bandwidth_kdw,kernel_kde=kernel_kde,optimize_bandwidth=optimize_bandwidth
            )
            for _cat in _cats
        }

        _cat_order = dict(enumerate(_cats))
        _colors = [
            "#1f77b4",  # muted blue
            "#ff7f0e",  # safety orange
            "#2ca02c",  # cooked asparagus green
            "#d62728",  # brick red
            "#9467bd",  # muted purple
            "#8c564b",  # chestnut brown
            "#e377c2",  # raspberry yogurt pink
            "#7f7f7f",  # middle gray
            "#bcbd22",  # curry yellow-green
            "#17becf",  # blue-teal
        ][: len(_cats)]

        _arg_violin = dict(
            data_frame=_df,
            x=val_col,
            color=cat_col,
            range_x=[0, _df[val_col].max()],
            points="all",
            box=True,
            category_orders=_cat_order,
            color_discrete_sequence=_colors,
        )

        fig = px.violin(**_arg_violin)
        for _cat_nm, _color, _kde_ext in zip(_cats, _colors, _kde_extrema.values()):
            for _loc_max, _size in zip(
                _kde_ext["local_extrema"], _kde_ext["kde_values"]
            ):
                _opacity = _size / sum(_kde_ext["kde_values"])
                _annotation_text = (
                    annot_prefix
                    + f"Local Max. KDE<br> {_cat_nm}: {str(np.round(_loc_max,2))}"
                )
                fig.add_vline(
                    x=_loc_max,
                    line_width=10 * _size,
                    line_color=_color,
                    opacity=_opacity,
                    annotation_text=_annotation_text,
                    annotation=dict(opacity=_opacity, font=dict(color=_color, size=8)),
                )
        fig.show()

    return None
