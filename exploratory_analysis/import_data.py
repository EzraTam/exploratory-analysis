"""
Importing module
"""

import pathlib
from typing import Dict
import pandas as pd


class DataImport:
    """Class for importing data
    from a file specified by the
    path.
    """

    def __init__(self, file_path: str):
        # Read input∏
        self.file_path = file_path

        # check file format
        implemented_file_format = ["xls"]
        assert (
            pathlib.PurePath(self.file_path).name.split(".")[1]
            in implemented_file_format
        ), "DataImport not yet impemented for this file format!"

    def load_file_to_pds(self) -> Dict[str, pd.DataFrame]:
        """Method for loading file in DF-format

        Returns:
            Dict[str, pd.DataFrame]: Dict with keys equal
            to the sheet names and values equal to the
            corresponding DF
        """
        _excel_object = pd.ExcelFile(self.file_path)
        return {
            sheet_name: _excel_object.parse(sheet_name)
            for sheet_name in _excel_object.sheet_names
        }


def import_excel_to_pds(path: str) -> Dict[str, pd.DataFrame]:
    """Function for importing DF

    Args:
        path (str): _description_

    Returns:
        Dict[str, pd.DataFrame]: _description_
    """
    return DataImport(path).load_file_to_pds()
