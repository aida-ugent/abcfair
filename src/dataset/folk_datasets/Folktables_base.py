import os
from folktables import ACSDataSource as FolkACSDataSource

from ..base import DataSource, Dataset
from utils.constants import DATA_DIR


class ACSDataSource(DataSource):
    name = "FolkTables-base"
    binary_sens_cols = None

    def __init__(self, states=None, **kwargs):
        super().__init__(**kwargs)
        self.states = states  # If None, take all states

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if cls.name is None:
            raise ValueError(f"Data class {cls} must set a 'name'.")

    def setup(self) -> [Dataset, Dataset, Dataset]:
        raise NotImplementedError

    def get_data(self):
        data_source = FolkACSDataSource(survey_year='2018', horizon='1-Year', survey='person', root_dir=DATA_DIR)
        data = data_source.get_data(download=True, states=self.states)
        return data

    @property
    def data_dir(self):
        return os.path.join(DATA_DIR, "folktables",self.__class__.name)