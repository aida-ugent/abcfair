import sklearn as sk
import torch
from torch import sigmoid
import numpy as np
from tqdm import tqdm
import wandb

from evaluator.evaluator import Evaluator


class Tuner(sk.base.BaseEstimator):
    def fit(self, *args, **kwargs):
        raise NotImplementedError

    def predict(self, *args, **kwargs):
        raise NotImplementedError


# Since we use the sklearn API for PyTorch, we could have used skorch. For clarity purposes, we will not use it here.
class TorchTuner(Tuner):
    def __init__(self, device='cpu'):
        self.device = device

    def fit(self, model, optimizer, train_loader, val_loader, nr_epochs, data_info, with_metrics=True, **kwargs):
        model = model.to(self.device)
        for epoch in tqdm(range(nr_epochs)):
            train_metrics = self.loop(model, optimizer, train_loader, data_info,
                                      stage="training", with_metrics=with_metrics)
            with torch.no_grad():
                val_metrics = self.loop(model, optimizer, val_loader, data_info,
                                        stage="val", with_metrics=with_metrics)

            if with_metrics:
                wandb.log(train_metrics | val_metrics | {"epoch": epoch})
        return model

    def predict(self, model, feat, sens, **kwargs):
        return sigmoid(model(feat))

    def calculate_loss(self, logit, label, sens, **kwargs):
        return torch.nn.functional.binary_cross_entropy_with_logits(logit, label)

    def loop(self, model, optimizer, dataloader, data_info=None, stage=None, with_metrics=True):
        if with_metrics:
            evaluator = Evaluator(stage, device=self.device, data_info=data_info)
        else:
            evaluator = None

        losses = []
        for feat, sens, label, sens_formats in dataloader:
            if stage == "training":
                optimizer.zero_grad()

            logit = model(feat)
            loss = self.calculate_loss(logit, label=label, sens=sens, feat=feat)
            losses.append(loss.item())

            if with_metrics:
                predictions = sigmoid(logit)
                evaluator.update(predictions, label, sens_formats)

            if stage == "training":
                loss.backward()
                optimizer.step()

        if with_metrics:
            metrics = evaluator.compute()
            metrics.update({f"{stage}/loss": np.mean(losses)})
            return metrics
