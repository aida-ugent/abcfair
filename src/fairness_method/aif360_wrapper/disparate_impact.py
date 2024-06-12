from torch import from_numpy
import numpy as np
from aif360.algorithms.preprocessing import DisparateImpactRemover

from fairness_method.base import FairnessMethod
from .common import dataset_to_aif360_dataset
from dataset.base import Dataset


# This method comes from the AIF360 package, use of this package is free, more information about its licensing can be
# found at https://github.com/Trusted-AI/AIF360?tab=Apache-2.0-1-ov-file#
class DisparateImpact(FairnessMethod):
    name = "disparate_impact"

    def __init__(self, fairness_strength=0.5, **kwargs):
        super().__init__(**kwargs)
        self.fairness_strength = fairness_strength

    def preprocess(self, dataset):
        aif360_dataset, feat_col, sens_col = dataset_to_aif360_dataset(dataset)

        preprocessing_model = DisparateImpactRemover(self.fairness_strength, sens_col[0])
        transformed_features = preprocessing_model.fit_transform(aif360_dataset)

        preprocessed_dataset = Dataset.from_numpy(
            np.float32(transformed_features.convert_to_dataframe()[0][feat_col]),
            np.float32(transformed_features.convert_to_dataframe()[0][sens_col]),
            dataset.label
        )
        preprocessed_dataset.sens_formats = dataset.sens_formats
        return preprocessed_dataset
