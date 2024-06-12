__all__ = ['build_data']


from .folk_datasets import CLASS_DICT as FOLK_CLASS_DICT
from .SchoolPerformanceUnbiased import SchoolPerformanceUnbiased
from .SchoolPerformanceBiased import SchoolPerformanceBiased


CLASSES = [SchoolPerformanceUnbiased, SchoolPerformanceBiased]
CLASS_DICT = {cls.name: cls for cls in CLASSES} | FOLK_CLASS_DICT

def build_data(name=None, **kwargs):
    try:
        cls = CLASS_DICT[name]
    except KeyError:
        raise ValueError(f"Data class {name} not found. Available classes: {list(CLASS_DICT.keys())}")

    return cls(**kwargs)
