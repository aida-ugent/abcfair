from torch import from_numpy
import torch
from torch.functional import F
from torch.utils.data import TensorDataset, DataLoader
from torch.distributions import Bernoulli
from sklearn.base import BaseEstimator
import numpy as np

# The following methods come from the fairlearn package, which is free to use, more information on its license can be
# found at https://github.com/fairlearn/fairlearn?tab=MIT-1-ov-file#readme

# All fairness constraints.
from fairlearn.reductions import DemographicParity, TruePositiveRateParity, FalsePositiveRateParity, ErrorRateParity

# Only use Exponentiation Gradient algorithm (with ErrorRate as objective).
from fairlearn.reductions import ExponentiatedGradient

from fairness_method.base import FairnessMethod
from model.neural_net import build_optimizer


STAT_TO_REDUCTIONS = {
    "pr": DemographicParity,
    "tpr": TruePositiveRateParity,
    "fpr": FalsePositiveRateParity,
    "acc": ErrorRateParity
}
SAMPLE_WEIGHT_NAME = "sample_weight"


class TorchModel(BaseEstimator):
    times_trained = 0

    def __init__(self, model, optimizer, train_loader, nr_epochs, device='cpu', **kwargs):
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.nr_epochs = nr_epochs
        self.device = device

    def fit(self, X, y, sample_weight=None):
        self.model.apply(
            lambda m: torch.nn.init.kaiming_uniform_(m.weight) if isinstance(m, torch.nn.Linear) else None
        )
        self.model.to(self.device)
        optimizer = build_optimizer(self.model,
                                    lr=self.optimizer.param_groups[0]["lr"],
                                    weight_decay_str=self.optimizer.param_groups[0]['weight_decay'])

        # It kinda sucks that we can't reuse the dataset class here, but it would probably be too complicated to
        # adjust it just for this method...
        dataset = TensorDataset(from_numpy(X).to(self.device),
                                from_numpy(y.to_numpy()).to(self.device),
                                from_numpy(sample_weight.to_numpy()).to(self.device))
        train_loader = DataLoader(dataset, batch_size=self.train_loader.batch_size, shuffle=True)

        for epoch in range(self.nr_epochs):
            losses = []
            for feat, label, sample_weight in train_loader:
                optimizer.zero_grad()

                logit = self.model(feat).squeeze()
                loss = F.binary_cross_entropy_with_logits(logit, label.float(), weight=sample_weight)
                loss.backward()

                optimizer.step()
                losses.append(loss.item())
            print(f"fit {TorchModel.times_trained}, epoch: {epoch}, loss: {np.mean(losses)}")
        TorchModel.times_trained += 1
        self.model.cpu()

    def predict(self, X):
        X = from_numpy(X)
        pred = Bernoulli(logits=self.model(X)).sample()
        return pred.squeeze().numpy()


class ExpGradient(FairnessMethod):
    name = "exp_gradient"

    def __init__(self, fairness_strength=0.9, statistic="pr", grid_size=10, **kwargs):
        super().__init__(**kwargs)
        self.fairness_strength = fairness_strength
        self.statistic = statistic
        self.grid_size = grid_size  # Note: this is the amount of models that are trained.

        self.exp_grad = None

    def inprocess(self, tuner, **kwargs):
        tuner.fit = self.train
        tuner.predict = self.predict
        return tuner

    def train(self, model, optimizer, train_loader, val_loader, nr_epochs, data_info, **kwargs):
        model_as_estimator = TorchModel(model, optimizer, train_loader, nr_epochs, device=self.device, **kwargs)

        constraint = STAT_TO_REDUCTIONS[self.statistic](ratio_bound=self.fairness_strength)
        self.exp_grad = ExponentiatedGradient(model_as_estimator, constraints=constraint, max_iter=self.grid_size)
        feat = train_loader.dataset.feat
        sens = train_loader.dataset.sens
        label = train_loader.dataset.label
        feat = feat.cpu().numpy()
        sens = sens.argmax(dim=1).cpu().numpy()
        label = label.squeeze().cpu().numpy()
        self.exp_grad.fit(feat, label, sensitive_features=sens)
        return model

    def predict(self, _model, feat, _sens, **kwargs):
        feat = feat.cpu().numpy()
        pred = from_numpy(self.exp_grad._pmf_predict(feat))[:, [1]]  # Output gives positive and negative distr.
        return pred.to(self.device)
