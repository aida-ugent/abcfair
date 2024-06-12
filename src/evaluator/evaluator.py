from fairret import statistic
from fairret.metric import LinearFractionalParity
from torchmetrics.classification import BinaryAccuracy, BinaryAUROC, BinaryF1Score

# Defining all the fairness metric on which we evaluate
metrics_dict = {"dem_par": statistic.PositiveRate(), "eq_opp": statistic.TruePositiveRate(),
                "pred_eq": statistic.FalsePositiveRate(),
                "pred_par": statistic.PositivePredictiveValue(), "forp": statistic.FalseOmissionRate(),
                "acc_eq": statistic.Accuracy(), "f1_score_eq": statistic.FScore()}


def dem_par_evaluator(sens_dim):
    return LinearFractionalParity(statistic.PositiveRate(), sens_dim)


def other_fairness_statistic_evaluator(sens_dim, hard_class=False):
    if hard_class:
        return {metric_name: LinearFractionalParity(metric_statistic, sens_dim)
                for metric_name, metric_statistic in metrics_dict.items() if metric_name != "tr_eq"}

    return {metric_name: LinearFractionalParity(metric_statistic, sens_dim)
            for metric_name, metric_statistic in metrics_dict.items()}


def performance_evaluator(device="cpu"):
    return {"accuracy": BinaryAccuracy().to(device),
            "auroc": BinaryAUROC().to(device),
            "f1_score": BinaryF1Score().to(device)}


class Evaluator:
    calc_metric_dict = {}

    def __init__(self, stage, device="cpu", data_info=None):
        self.stage = stage
        for sens_type in ["all_sens", "one_sens", "intersect_sens"]:
            self.calc_metric_dict[sens_type] = {}
            if sens_type == "all_sens":
                relevant_sens_dim = data_info['sens_formats_dims']['parallel']
            elif sens_type == "one_sens":
                relevant_sens_dim = data_info['sens_formats_dims']['binary']
            else:
                relevant_sens_dim = data_info['sens_formats_dims']['intersectional']

            for pred_type in ["soft", "hard"]:
                self.calc_metric_dict[sens_type][pred_type] = {}
                for metric_name, metric_statistic in metrics_dict.items():
                    if pred_type == "hard":
                        if metric_name != "tr_eq":
                            self.calc_metric_dict[sens_type][pred_type][metric_name] = LinearFractionalParity(
                                metric_statistic, relevant_sens_dim).to(device)
                    else:
                        self.calc_metric_dict[sens_type][pred_type][metric_name] = LinearFractionalParity(
                            metric_statistic, relevant_sens_dim).to(device)
        self.performance_metrics = performance_evaluator(device)

    def update(self, predictions, batch_label, batch_sens_formats):
        for pred_type in ["soft", "hard"]:
            if pred_type == "soft":
                batch_pred = predictions
            else:
                batch_pred = predictions.round()
            for sens_type in ["all_sens", "one_sens", "intersect_sens"]:
                if sens_type == "all_sens":
                    batch_sens = batch_sens_formats['parallel']
                elif sens_type == "one_sens":
                    batch_sens = batch_sens_formats['binary']
                else:
                    batch_sens = batch_sens_formats['intersectional']

                for metric_name in self.calc_metric_dict[sens_type][pred_type].keys():
                    if metric_name == "dem_par":
                        self.calc_metric_dict[sens_type][pred_type][metric_name].update(batch_pred, batch_sens)
                    else:
                        self.calc_metric_dict[sens_type][pred_type][metric_name].update(batch_pred, batch_sens,
                                                                                        batch_label)
        for metric_name, metric in self.performance_metrics.items():
            metric.update(predictions, batch_label)

    def compute(self):
        result_dict = {}
        for pred_type in ["soft", "hard"]:
            for sens_type in ["all_sens", "one_sens", "intersect_sens"]:
                for metric_name in self.calc_metric_dict[sens_type][pred_type].keys():
                    result_dict[f"{self.stage}/{pred_type}/{sens_type}/{metric_name}"] = \
                    self.calc_metric_dict[sens_type][pred_type][metric_name].compute()

        for metric_name, metric in self.performance_metrics.items():
            result_dict[f"{self.stage}/{metric_name}"] = metric.compute()

        return result_dict

    def reset(self):
        for sens_type in ["all_sens", "one_sens", "intersect_sens"]:
            for pred_type in ["soft", "hard"]:
                for metric in self.calc_metric_dict[sens_type][pred_type].values():
                    metric.reset()

        for metric in self.performance_metrics.values():
            metric.reset()
