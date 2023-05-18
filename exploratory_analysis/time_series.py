from typing import Optional, List
import pandas as pd

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
) -> pd.DataFrame:

    _df = getattr(raw_data.groupby([col_time])[col_val], aggregator_data)()

    if norm_val is not None:
        _df = _df / norm_val

    _df = pd.DataFrame(_df).reset_index()

    if display_nm_col_smoothed is None:
        display_nm_col_smoothed = [
            f"{col_val}__smoothed_{_length}" for _length in smooth_lengths
        ]

    for _ind, _length in enumerate(smooth_lengths):
        _df[display_nm_col_smoothed[_ind]] = _df[col_val].ewm(span=_length).mean()

    if display_nm_col_val is not None:
        _df = _df.rename(columns={col_val: display_nm_col_val})

    return _df