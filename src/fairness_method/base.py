from typing import Callable

from dataset.base import Dataset
from model.model_tuner import Tuner


class FairnessMethod:
    name = None

    def __init__(self, device='cpu'):
        self.device = device

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if cls.name is None:
            raise ValueError(f"Data class {cls} must set a 'name'.")

    def preprocess(self, dataset: Dataset) -> Dataset:
        raise NotImplementedError

    def inprocess(self, tuner) -> Tuner:
        raise NotImplementedError

    def postprocess(self, model, dataloader) -> Callable:
        raise NotImplementedError
