from torch import from_numpy
import numpy as np
import pandas as pd
from aequitas.flow.methods.preprocessing.data_repairer import DataRepairer
from aequitas.flow.methods.preprocessing.label_flipping import LabelFlipping
from aequitas.flow.methods.preprocessing.prevalence_sample import PrevalenceSampling

# The three below are not in aequitas 1.0.0.
# from aequitas.flow.methods.preprocessing.massaging import Massaging
# from aequitas.flow.methods.preprocessing.correlation_suppression import CorrelationSuppression
# from aequitas.flow.methods.preprocessing.feature_importance_suppression import FeatureImportanceSuppression

from fairness_method.base import FairnessMethod
from utils.utils import dataset_to_pandas_df
from dataset.base import Dataset

# The following methods come from the Aequitas package, which is free to use, more information on its license can be
# found at https://github.com/dssg/aequitas?tab=MIT-1-ov-file#readme


class AequitasWrapper(FairnessMethod):
    name = "aequitas"

    def __init__(self, fairness_strength=0.5, statistic='pr', **kwargs):
        super().__init__(**kwargs)
        self.fairness_strength = fairness_strength
        assert statistic == 'pr', "Only PR is supported for now"

        self.aequitas_method = None

    def preprocess(self, dataset: Dataset):
        df, (feat_col, sens_col, label_col) = dataset_to_pandas_df(dataset)

        df['sens'] = np.argmax(df[sens_col].values, axis=1)
        self.aequitas_method.fit(df[feat_col], df[label_col], df['sens'])
        feat, label, sens = self.aequitas_method.transform(df[feat_col], df[label_col], df['sens'])

        preprocessed_dataset = Dataset.from_numpy(
            np.float32(feat.values),
            np.float32(pd.get_dummies(sens).values),
            np.float32(label.values)[..., np.newaxis]
        )
        preprocessed_dataset.sens_formats = dataset.sens_formats
        return preprocessed_dataset


class DataRepairerWrapper(AequitasWrapper):
    name = "data_repairer"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.aequitas_method = DataRepairer(repair_level=self.fairness_strength)


class LabelFlippingWrapper(AequitasWrapper):
    name = "label_flipping"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Setting seed to none because it otherwise has 42 as default.
        self.aequitas_method = LabelFlipping(max_flip_rate=self.fairness_strength, seed=None)


class PrevalenceSamplingWrapper(AequitasWrapper):
    name = "prevalence_sampling"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Setting seed to none because it otherwise has 42 as default.
        self.aequitas_method = PrevalenceSampling(alpha=self.fairness_strength, seed=None)
