from torch import from_numpy
import numpy as np
import pandas as pd

from aif360.datasets.binary_label_dataset import BinaryLabelDataset
from aif360.algorithms.preprocessing import LFR

from .common import dataset_to_aif360_dataset
from fairness_method.base import FairnessMethod
from dataset.base import Dataset


# This method comes from the AIF360 package, use of this package is free, more information about its licensing can be
# found at https://github.com/Trusted-AI/AIF360?tab=Apache-2.0-1-ov-file#
class LearningFairRepr(FairnessMethod):
    name = "LearningFairRepr"
    preprocess_model = None

    def __init__(self, fairness_strength=0.5, **kwargs):
        super().__init__(**kwargs)
        self.fairness_strength = fairness_strength
        self.preprocessing_model = None

    def preprocess(self, dataset):
        aif360_dataset, feat_col, sens_col = dataset_to_aif360_dataset(dataset)

        preprocessing_model = LFR(unprivileged_groups=[{sens_col[0]: 0}], privileged_groups=[{sens_col[0]: 1}],
                                  k=5, Ax=0.1, Ay=1.0, Az=self.fairness_strength, verbose=1)
        self.preprocessing_model = preprocessing_model.fit(aif360_dataset, maxiter=2000, maxfun=600)
        transformed_features = self.preprocessing_model.transform(aif360_dataset)

        preprocessed_dataset = Dataset.from_numpy(
            np.float32(transformed_features.convert_to_dataframe()[0][feat_col]),
            np.float32(transformed_features.convert_to_dataframe()[0][sens_col]),
            dataset.label
        )
        preprocessed_dataset.sens_formats = dataset.sens_formats
        return preprocessed_dataset

    def feat_transform(self, feat, sens):
        labels = np.zeros((sens.shape[0], 1))
        feat_col = [f"feat{i}" for i in range(feat.shape[1])]
        sens_col = [f"sens{i}" for i in range(sens.shape[1])]
        df = pd.DataFrame(np.hstack((feat.numpy(), sens.numpy(), labels)), columns=feat_col + sens_col + ["label"])
        aif360_dataset = BinaryLabelDataset(df=df, label_names=["label"], protected_attribute_names=sens_col)
        transformed_features = self.preprocessing_model.transform(aif360_dataset)
        return from_numpy(np.float32(transformed_features.convert_to_dataframe()[0][feat_col]))
