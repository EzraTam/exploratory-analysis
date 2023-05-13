"""Module for exploring the distribution
of data
"""
from typing import Optional

import pandas as pd
from sklearn.neighbors import KernelDensity
from scipy.signal import argrelextrema
import numpy as np


def find_local_kde_extrema(
    data: pd.Series,
    resolution: int,
    how: Optional[str] = "maxima",
    bandwidth_kdw: Optional[float] = 0.6,
    kernel_kde: Optional[str] = "gaussian",
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
    # Fit a kernel density estimation to the data
    kde = KernelDensity(kernel=kernel_kde, bandwidth=bandwidth_kdw).fit(
        np.array(data).reshape(-1, 1)
    )
    # Generate x values for evaluating the KDE
    x_values = np.linspace(min(data), max(data), 100)

    kde_values = np.exp(kde.score_samples(x_values.reshape(-1, 1)))
    _comparator = {"maxima": np.greater, "minima": np.less}[how]
    _pos_extrema = argrelextrema(data=kde_values, comparator=_comparator)
    return {
        "local_extrema": x_values[_pos_extrema],
        "kde_values": kde_values[_pos_extrema],
    }
