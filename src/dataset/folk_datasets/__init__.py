__all__ = ['build_data']

from .ACSIncome import ACSIncome
from .ACSEmployment import ACSEmployment
from .ACSMobility import ACSMobility
from .ACSPublicCoverage import ACSPublicCoverage
from .ACSTravelTime import ACSTravelTime

CLASSES = [ACSIncome, ACSEmployment, ACSMobility, ACSPublicCoverage, ACSTravelTime]
CLASS_DICT = { name:datasetclass for name, datasetclass in ((datasetclass.name, datasetclass) for datasetclass in CLASSES)}

def build_data(name=None, **kwargs):
    try:
        cls = CLASS_DICT[name]
    except KeyError:
        raise ValueError(f"Data class {name} not found. Available classes: {list(CLASS_DICT.keys())}")

    return cls(**kwargs)