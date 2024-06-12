from .aif360_wrapper import DisparateImpact, LearningFairRepr
from .aequitas_wrapper import DataRepairerWrapper, LabelFlippingWrapper, PrevalenceSamplingWrapper
from .fairlearn_wrapper import ExpGradient
from .ffb_wrapper import PRLoss, HSIC, LAFTR
from .fairret_wrapper import Fairret
from .error_parity import ErrorParity
from .no_method import NoMethod

CLASSES = [DisparateImpact, Fairret, NoMethod, LearningFairRepr, ExpGradient, PRLoss, HSIC, LAFTR, DataRepairerWrapper,
           LabelFlippingWrapper, PrevalenceSamplingWrapper, ErrorParity]
CLASS_DICT = {name: fairness_method for name, fairness_method in
              ((fairness_method.name, fairness_method) for fairness_method in CLASSES)}


def build_method(name=None, **kwargs):
    try:
        cls = CLASS_DICT[name]
    except KeyError:
        raise ValueError(f"Data class {name} not found. Available classes: {list(CLASS_DICT.keys())}")

    return cls(**kwargs)
