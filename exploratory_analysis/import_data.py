"""
Importing module
"""

import pathlib
from typing import Dict,Union
import pandas as pd


class DataImport:
    """Class for importing data
    from a file specified by the
    path.
    """

    def __init__(self, file_path: str):
        # Read input∏
        self.file_path = file_path

        self.file_format=pathlib.PurePath(self.file_path).name.split(".")[1]

        # check file format
        implemented_file_format = ["xls","csv"]
        assert (
            self.file_format
            in implemented_file_format
        ), "DataImport not yet impemented for this file format!"

    def load_file_to_pds(self) -> Union[Dict[str, pd.DataFrame],pd.DataFrame]:
        """Method for loading file in DF-format

        Returns:
            Dict[str, pd.DataFrame]: Dict with keys equal
            to the sheet names and values equal to the
            corresponding DF
        """
        if self.file_format == "xls":
            _excel_object = pd.ExcelFile(self.file_path)
            return {
                sheet_name: _excel_object.parse(sheet_name)
                for sheet_name in _excel_object.sheet_names
            }

        if self.file_format == "csv":
            return pd.read_csv(self.file_path)


def import_to_pds(path: str) -> Dict[str, pd.DataFrame]:
    """Function for importing DF from certain file

    Args:
        path (str): Path of the data

    Returns:
        Dict[str, pd.DataFrame]: Dictionary for DF for each sheets
    """
    return DataImport(path).load_file_to_pds()
