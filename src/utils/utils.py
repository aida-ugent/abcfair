import pandas as pd
import numpy as np
import os
import yaml
import wandb

from .constants import CONFIG_DIR
from .secrets import ENTITY, PROJECT
from dataset.base import Dataset


def import_nested_dict(dict_to_keep, dict_to_import):
    """
    If the key is already in dict_to_keep, then it is not overwritten with values from dict_to_import!
    """

    for import_key, import_value in dict_to_import.items():
        if import_key in dict_to_keep.keys():
            if isinstance(import_value, dict):
                if not isinstance(dict_to_keep[import_key], dict):
                    raise ValueError(f"Importing dict into non-dict value at key {import_key}!")
                dict_to_keep[import_key] = import_nested_dict(dict_to_keep[import_key], import_value)
        elif import_key == "value" and "values" in dict_to_keep.keys():
            continue
        else:
            dict_to_keep[import_key] = import_value
    return dict_to_keep


def import_config(existing_config, imported_config_path):
    if imported_config_path is None:
        return

    if not os.path.exists(imported_config_path):
        imported_config_path = os.path.join(CONFIG_DIR, imported_config_path)

    with open(imported_config_path, 'r') as file:
        config_fairness_method = yaml.safe_load(file)
    import_nested_dict(existing_config, config_fairness_method)


def setup_logging(config):
    wandb.init(
        entity=ENTITY,
        project=PROJECT,
        allow_val_change=True,
        config=config
    )


def dataset_to_pandas_df(dataset: Dataset):
    feat = dataset.feat
    sens = dataset.sens
    label = dataset.label

    feat_col = [f"feat{i}" for i in range(feat.shape[1])]
    sens_col = [f"sens{i}" for i in range(sens.shape[1])]
    label_col = "label"
    df = pd.DataFrame(np.hstack((feat.numpy(), sens.numpy(), label.numpy())),
                      columns=feat_col + sens_col + [label_col])
    return df, (feat_col, sens_col, label_col)
