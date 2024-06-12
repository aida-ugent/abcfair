from abc import ABC, abstractmethod
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset as TorchDataset
import torch
import numpy as np
from typing import Dict

from utils import DATA_DIR


SENS_FORMATS = ["binary", "intersectional", "parallel"]


class Dataset(TorchDataset):
    def __init__(self, feat: torch.Tensor, sens: torch.Tensor, label: torch.Tensor,
                 sens_formats: Dict[str, torch.Tensor] = None):
        self.feat = feat
        self.sens = sens
        self.label = label
        self.sens_formats = sens_formats

        if sens_formats is None:
            self.sens_formats = {}

        # Check that the lengths of all tensors match.
        assert feat.shape[0] == sens.shape[0] == label.shape[0]
        nb_samples = feat.shape[0]
        assert all(t.shape[0] == nb_samples for t in self.sens_formats.values())

    @staticmethod
    def from_numpy(feat, sens, label, sens_formats=None):
        if sens_formats is None:
            sens_formats = {}
        if isinstance(feat, np.ndarray):
            feat = torch.from_numpy(feat).float()
        if isinstance(sens, np.ndarray):
            sens = torch.from_numpy(sens).float()
        if isinstance(label, np.ndarray):
            label = torch.from_numpy(label).float()
        sens_formats = {key: torch.from_numpy(sens_formats[key]).float() for key in sens_formats}
        return Dataset(feat, sens, label, sens_formats)

    def to(self, device):
        for attr in ['feat', 'sens', 'label']:
            setattr(self, attr, getattr(self, attr).to(device))
        for key in self.sens_formats:
            self.sens_formats[key] = self.sens_formats[key].to(device)
        return self

    def __getitem__(self, index):
        return self.feat[index], self.sens[index], self.label[index], {
            key: self.sens_formats[key][index] for key in self.sens_formats
        }

    def __len__(self):
        return self.feat.shape[0]


class DataSource(ABC):
    name = None
    binary_sens_cols = None

    @abstractmethod
    def setup(self) -> [Dataset, Dataset, Dataset]:
        """Load the data and return the train, val, and test datasets."""
        pass

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if cls.name is None:
            raise ValueError(f"Data class {cls} must set a 'name'.")

    def __init__(self,
                 main_sens_format="parallel",
                 batch_size=32,
                 drop_sens_feat=True,
                 device='cpu'):
        if main_sens_format not in SENS_FORMATS:
            raise ValueError(
                f"The main_sens_format {main_sens_format} should be in {SENS_FORMATS}.")
        elif main_sens_format == "binary" and self.binary_sens_cols is None:
            raise ValueError("binary_sens_cols must be set if main_sens_format='binary'.")

        self.main_sens_format = main_sens_format
        self.batch_size = batch_size
        self.drop_sens_feat = drop_sens_feat
        self.device = device

        self.train_data = None
        self.val_data = None
        self.test_data = None

        self.binary_sens_col_names = None
        self.intersect_col_names = None
        self.parallel_sens_col_names = None

    def dataloader(self, stage: str) -> DataLoader:
        dataset = getattr(self, f"{stage}_data")
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=stage == 'train')

    def train_dataloader(self):
        return self.dataloader("train")

    def val_dataloader(self):
        return self.dataloader("val")

    def test_dataloader(self):
        return self.dataloader("test")

    @property
    def data_dir(self):
        return os.path.join(DATA_DIR, self.__class__.name)

    @property
    def feat_dim(self):
        return self.train_data.feat.shape[1]

    @property
    def sens_dim(self):
        return self.train_data.sens.shape[1]

    @property
    def sens_formats_dims(self):
        return {key: self.train_data.sens_formats[key].shape[1] for key in self.train_data.sens_formats}

    def info(self) -> dict:
        return {
            'feat_dim': self.feat_dim,
            'sens_dim': self.sens_dim,
            'sens_formats_dims': self.sens_formats_dims
        }

    def _preprocess_df(self,
                       df,
                       sens_columns=None,
                       label_column=None,
                       label_column_unbiased=None,
                       drop_columns=None,
                       categorical_values=None,
                       mapping=None,
                       mapping_label=None,
                       normalise_columns=None,
                       drop_rows=None):
        if self.binary_sens_cols is None:
            raise ValueError("binary_sens_cols must be set in order to use this method")

        if not label_column_unbiased:
            label_column_unbiased = label_column

        if drop_rows:
            for column in drop_rows:
                for value in drop_rows[column]:
                    df = df.drop(df[df[column] == value].index)

        if drop_columns:
            df = df.drop(columns=drop_columns)

        if mapping:
            for column in mapping:
                df[column] = df[column].squeeze().replace(mapping[column])

        df = df.dropna()

        if categorical_values:
            df = pd.get_dummies(df, columns=categorical_values)
            expanded_sens_columns = []
            for column in sens_columns:
                if column in categorical_values:
                    expanded_sens_columns += list(df.filter(regex=('^' + column + "_")).columns)
                else:
                    expanded_sens_columns += [column]
            sens_columns = expanded_sens_columns

        if mapping_label:
            df[label_column] = df[label_column].applymap(mapping_label)

        if normalise_columns:
            for column in normalise_columns:
                if column in sens_columns:
                    df[column] = (df[column] - df[column].min()) / (df[column].max() - df[column].min())
                else:
                    df[column] = (df[column] - df[column].mean()) / df[column].std()

        print(f"Sensitive features: {sens_columns}")

        # Having a set of binary sensitive attributes
        sens_df = df[sens_columns]
        binary_sens_df = sens_df[sens_df.columns[self.binary_sens_cols].tolist()]
        print(f"Binary sensitive features set: {binary_sens_df.columns}")
        self.binary_sens_col_names = sens_df.columns[self.binary_sens_cols].tolist()
        # Mostly a sanity check.
        # It is expected that binary sens columns are one-hot encodings.
        if (binary_sens_df.sum(axis=1) != 1).any():
            raise ValueError("Binary sensitive features are not mutually exclusive.")

        # Create set for intersectional fairness
        _, np_array_group_number, counts = np.unique(df[sens_columns].round().values.astype(np.bool8), axis=0,
                                                     return_inverse=True, return_counts=True)
        small_count = np.where(counts < 10)[0]
        if len(np.where(counts < len(np_array_group_number) * 0.001)[0]) > 0:
            for i in range(1, len(small_count)):
                np_array_group_number[np.where(np_array_group_number == small_count[i])[0]] = small_count[0]
        intersectional_df = pd.get_dummies(np_array_group_number).add_prefix("intersect_")
        self.intersect_col_names = intersectional_df.columns.to_list()
        df = pd.concat((df, intersectional_df), axis=1)

        self.parallel_sens_col_names = sens_columns

        # If the method only uses one sensitive attribute
        if self.main_sens_format == "binary":
            sens_columns = sens_df.columns[self.binary_sens_cols].tolist()

        # If the method uses intersectional groups
        elif self.main_sens_format == "intersectional":
            sens_columns = self.intersect_col_names

        train_val, test = train_test_split(df, test_size=0.2)
        train, val = train_test_split(train_val, test_size=0.2)

        if self.drop_sens_feat:
            not_feat_cols = (label_column +
                             self.parallel_sens_col_names +
                             self.intersect_col_names +
                             self.binary_sens_col_names)
        else:
            not_feat_cols = label_column

        self.train_data = Dataset.from_numpy(
            train.drop(columns=not_feat_cols).to_numpy(dtype="float64"),
            train[sens_columns].to_numpy(dtype="float64"),
            train[label_column].to_numpy(dtype="float64"),
            {
                "parallel": train[self.parallel_sens_col_names].to_numpy(dtype="float64"),
                "binary": train[self.binary_sens_col_names].to_numpy(dtype="float64"),
                "intersectional": train[self.intersect_col_names].to_numpy(dtype="float64")
            }
        )

        self.val_data, self.test_data = (Dataset.from_numpy(
            sub_df.drop(columns=not_feat_cols).to_numpy(dtype="float64"),
            sub_df[sens_columns].to_numpy(dtype="float64"),
            sub_df[label_column_unbiased].to_numpy(dtype="float64"),
            {
                "parallel": sub_df[self.parallel_sens_col_names].to_numpy(dtype="float64"),
                "binary": sub_df[self.binary_sens_col_names].to_numpy(dtype="float64"),
                "intersectional": sub_df[self.intersect_col_names].to_numpy(dtype="float64")
            }
        ) for sub_df in [val, test])

        self.train_data = self.train_data.to(self.device)
        self.val_data = self.val_data.to(self.device)
        self.test_data = self.test_data.to(self.device)
        return self.train_data, self.val_data, self.test_data


class CSVDataSource(DataSource):
    name = "csv"
    delimiter = ";"

    def setup(self):
        pass

    def read(self, data_set, **preprocess_kwargs):
        df = pd.read_csv(os.path.join(DATA_DIR, data_set), delimiter=self.delimiter)
        return self._preprocess_df(df, **preprocess_kwargs)
