import torch
from fairness_method.base import FairnessMethod

from error_parity import RelaxedThresholdOptimizer

MAPPER_FAIRNESS_NAMES = {"pr": "demographic_parity", "tpr": "true_positive_rate_parity",
                         "fpr": "false_positive_rate_parity", "eo": "equalized_odds"}


# The following methods come from the error-parity package, which is free to use, more information on its license can be
# found at https://github.com/socialfoundations/error-parity?tab=MIT-1-ov-file#readme.


class ErrorParity(FairnessMethod):
    name = "error_parity"

    def __init__(self, fairness_strength=1, statistic="pr", **kwargs):
        super().__init__(**kwargs)
        if statistic not in MAPPER_FAIRNESS_NAMES.keys():
            raise ValueError(f"{statistic} is not a valid statistic for the error-parity method")
        self.statistic = MAPPER_FAIRNESS_NAMES[statistic]
        self.tolerance = fairness_strength

    def postprocess(self, model, dataloader):
        wrapped_model = ErrorParityModelWrapper(model)
        self.thresholder = RelaxedThresholdOptimizer(
            predictor=wrapped_model,
            constraint=self.statistic,
            tolerance=self.tolerance
        )
        feat = dataloader.dataset.feat
        sens = dataloader.dataset.sens
        label = dataloader.dataset.label

        y_scores = model(feat)

        self.thresholder.fit(X=feat.numpy(), y=label.numpy(), group=torch.argmax(sens, dim=1).numpy(),
                             y_scores=y_scores.detach().numpy())
        return self.predict

    def predict(self, model, feat, sens, **kwargs):
        post_processed_predictions = self.thresholder.predict(X=feat.numpy(), group=torch.argmax(sens, dim=1).numpy())
        return torch.unsqueeze(torch.tensor(post_processed_predictions), 1)


class ErrorParityModelWrapper:
    def __init__(self, model):
        self.model = model

    def __call__(self, input):
        return self.model(torch.tensor(input)).detach().numpy()
