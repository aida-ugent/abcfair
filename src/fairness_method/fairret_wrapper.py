from fairret.statistic import PositiveRate, TruePositiveRate, FalsePositiveRate, PositivePredictiveValue, \
    FalseOmissionRate, Accuracy, FalseNegativeFalsePositiveFraction, FScore
from fairret.loss import NormLoss, KLProjectionLoss, JensenShannonProjectionLoss, TotalVariationProjectionLoss
from fairret.loss import SquaredEuclideanProjectionLoss, LSELoss

from fairness_method.base import FairnessMethod


# The following methods come from the fairret package, which is free to use, more information on its license can be found
# at https://github.com/aida-ugent/fairret?tab=MIT-1-ov-file#readme.


class Fairret(FairnessMethod):
    name = "fairret"
    is_positive_rate = False

    def __init__(self, fairness_strength=1, statistic="pr", fairret="norm", **kwargs):
        super().__init__(**kwargs)
        self.fairness_strength = fairness_strength
        statistic_dict = {"pr": PositiveRate(), 'tpr': TruePositiveRate(), "fpr": FalsePositiveRate(),
                          "ppv": PositivePredictiveValue(),
                          "for": FalseOmissionRate(), "acc": Accuracy(), "fn_fp": FalseNegativeFalsePositiveFraction(),
                          "f1_score": FScore()}
        try:
            self.statistic = statistic_dict[statistic]
        except KeyError:
            raise ValueError(f"{statistic} is not a valid statistic for fairret configuration")

        if statistic == "pr":
            self.is_positive_rate = True

        fairret_dict = {"norm": NormLoss(self.statistic), "KL_proj": KLProjectionLoss(self.statistic),
                        "JS_proj": JensenShannonProjectionLoss(self.statistic),
                        "tot_var_proj": TotalVariationProjectionLoss(self.statistic),
                        "sq_eucl": SquaredEuclideanProjectionLoss(self.statistic), "lse": LSELoss(self.statistic)}

        try:
            self.fairret = fairret_dict[fairret]
        except KeyError:
            raise ValueError(f"{fairret} is not a valid for fairret type in fairret configuration")

        self.bce_loss = None

    def inprocess(self, tuner):
        self.bce_loss = tuner.calculate_loss
        tuner.calculate_loss = self.calculate_loss
        return tuner

    def calculate_loss(self, logit, label, sens, **kwargs):
        loss = self.bce_loss(logit, label, None)
        if self.is_positive_rate:
            loss += self.fairness_strength * self.fairret(logit, sens)
        else:
            loss += self.fairness_strength * self.fairret(logit, sens, label)
        return loss
