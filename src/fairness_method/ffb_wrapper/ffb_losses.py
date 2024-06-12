import torch

from ..base import FairnessMethod


# The following methods come from the FFB package, which is free to use, more information on its license can be found
# at https://github.com/ahxt/fair_fairness_benchmark?tab=MIT-1-ov-file#readme


class FFBWrapper(FairnessMethod):
    name = "ffb"

    def __init__(self, fairness_strength=1, statistic="pr", **kwargs):
        super().__init__(**kwargs)
        self.fairness_strength = fairness_strength
        self.statistic = statistic

        if statistic != 'pr':
            raise ValueError(f"FFBWrapper only supports 'pr' statistic, not {statistic}.")

    def inprocess(self, tuner, **kwargs):
        self.bce_loss = tuner.calculate_loss
        tuner.calculate_loss = self.calculate_loss
        return tuner

    def forward(self, y_pred, s):
        raise NotImplementedError("Subclasses must implement this method.")

    def calculate_loss(self, logit, label, sens, **kwargs):
        loss = self.bce_loss(logit, label, None)

        if (sens.sum(dim=1) != 1).any():
            raise ValueError("Binary sensitive features must be one-hot encodings of demographic group.")
        if sens.shape[1] != 2:
            raise ValueError("FFBWrapper only supports binary sensitive features.")
        sens = torch.argmax(sens, dim=1, keepdim=True).float()
        pred = torch.sigmoid(logit)

        ffb_loss = self.forward(pred, sens)
        loss += self.fairness_strength * ffb_loss
        return loss


class HSIC(FFBWrapper):  # using linear
    name = "hsic"

    def __init__(self, s_x=1, s_y=1, **kwargs):
        super().__init__(**kwargs)
        self.s_x = s_x
        self.s_y = s_y

    def forward(self, x, y):
        m, _ = x.shape  # batch size
        K = HSIC.GaussianKernelMatrix(x, self.s_x)
        L = HSIC.GaussianKernelMatrix(y, self.s_y)
        H = torch.eye(m) - 1.0 / m * torch.ones((m, m))
        return torch.trace(torch.mm(L, torch.mm(H, torch.mm(K, H)))) / ((m - 1) ** 2)

    @staticmethod
    def pairwise_distances(x):
        # x should be two dimensional
        instances_norm = torch.sum(x ** 2, -1).reshape((-1, 1))
        return -2 * torch.mm(x, x.t()) + instances_norm + instances_norm.t()

    @staticmethod
    def GaussianKernelMatrix(x, sigma=1):
        pairwise_distances_ = HSIC.pairwise_distances(x)
        return torch.exp(-pairwise_distances_ / sigma)


class PRLoss(FFBWrapper):
    name = "prejudice_remover"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, y_pred, s):
        output_f = y_pred[s == 0]
        output_m = y_pred[s == 1]

        # For the mutual information,
        # Pr[y|s] = sum{(xi,si),si=s} sigma(xi,si) / #D[xs]

        # D[xs]
        N_female = torch.tensor(output_f.shape[0])
        N_male = torch.tensor(output_m.shape[0])

        # male sample, #female sample
        Dxisi = torch.stack((N_male, N_female), dim=0)

        # Pr[y|s]
        y_pred_female = torch.sum(output_f)
        y_pred_male = torch.sum(output_m)
        P_ys = torch.stack((y_pred_male, y_pred_female), dim=0) / Dxisi

        # Pr[y]
        P = torch.cat((output_f, output_m), 0)
        P_y = torch.sum(P) / y_pred.shape[0]

        # P(siyi)
        P_s1y1 = torch.log(P_ys[1]) - torch.log(P_y)
        P_s1y0 = torch.log(1 - P_ys[1]) - torch.log(1 - P_y)
        P_s0y1 = torch.log(P_ys[0]) - torch.log(P_y)
        P_s0y0 = torch.log(1 - P_ys[0]) - torch.log(1 - P_y)

        # PI
        PI_s1y1 = output_f * P_s1y1
        PI_s1y0 = (1 - output_f) * P_s1y0
        PI_s0y1 = output_m * P_s0y1
        PI_s0y0 = (1 - output_m) * P_s0y0
        PI = (
                torch.sum(PI_s1y1)
                + torch.sum(PI_s1y0)
                + torch.sum(PI_s0y1)
                + torch.sum(PI_s0y0)
        )
        return PI
