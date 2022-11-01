import pathlib
from typing import Dict
import pandas as pd


class DataImport:
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
        _excel_object = pd.ExcelFile(self.file_path)
        return {
            sheet_name: _excel_object.parse(sheet_name)
            for sheet_name in _excel_object.sheet_names
        }


def import_excel_to_pds(path: str) -> Dict[str, pd.DataFrame]:
    return DataImport(path).load_file_to_pds()
