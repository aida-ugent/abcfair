from aif360.datasets.binary_label_dataset import BinaryLabelDataset

from utils.utils import dataset_to_pandas_df
from dataset.base import Dataset


# This method comes from the AIF360 package, use of this package is free, more information about its licensing can be
# found at https://github.com/Trusted-AI/AIF360?tab=Apache-2.0-1-ov-file#

def dataset_to_aif360_dataset(dataset: Dataset):
    df, col_names = dataset_to_pandas_df(dataset)
    sens_col = col_names[1]
    label_col = col_names[2]
    aif360_dataset = BinaryLabelDataset(df=df, label_names=[label_col], protected_attribute_names=sens_col)
    return aif360_dataset, col_names[0], col_names[1]
