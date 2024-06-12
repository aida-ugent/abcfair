import torch
from ..base import FairnessMethod

from model.neural_net import build_neural_net


class GradReverse(torch.autograd.Function):
    """
    borrwed from https://github.com/hanzhaoml/ICLR2020-CFair/blob/master/models.py
    Implement the gradient reversal layer for the convenience of domain adaptation neural network.
    The forward part is the identity function while the backward part is the negative function.
    """

    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg()


def grad_reverse(x):
    return GradReverse.apply(x)

class LAFTR(FairnessMethod):
    name = 'laftr'

    """
    Modified from https://github.com/ahxt/fair_fairness_benchmark/
    which in turn modified it from https://github.com/hanzhaoml/ICLR2020-CFair/blob/master/models.py
    "Multi-layer perceptron with adversarial training for fairness".
    """

    def __init__(self,
                 fairness_strength=0.5,
                 reconstruction_strength=0.,
                 adv_dim=None,
                 rec_dim=None,
                 label_given=False,
                 **kwargs):
        super().__init__(**kwargs)

        self.fairness_strength = fairness_strength
        self.reconstruction_strength = reconstruction_strength  # Using reconstruction loss means using LAFTR.
        if adv_dim is None:
            adv_dim = []
        self.adv_dim = adv_dim
        if rec_dim is None:
            rec_dim = []
        self.rec_dim = rec_dim
        self.label_given = label_given

        self.base_fit = None
        self.base_loop = None
        self.encoder = None
        self.classifier = None
        self.adversary = None
        self.decoder = None

    def inprocess(self, tuner, **kwargs):
        self.base_fit = tuner.fit
        tuner.fit = self.train
        tuner.calculate_loss = self.calculate_loss
        return tuner

    def train(self, model, optimizer, train_loader, val_loader, nr_epochs, data_info, **kwargs):
        linear_layers = [m for m in model if isinstance(m, torch.nn.Linear)]
        if len(linear_layers) < 2:
            raise ValueError(f"Adversarial debiasing only works for networks with at least one hidden layer, but only "
                             f"found {len(linear_layers)} linear layers.")

        self.encoder = model[:-1]
        self.classifier = torch.nn.Sequential(model[-1])
        encoding_dim = linear_layers[-2].out_features
        adv_input_dim = encoding_dim + (1 if self.label_given else 0)
        sens_dim = train_loader.dataset.sens.shape[1]
        self.adversary = build_neural_net(adv_input_dim, self.adv_dim, output_dim=sens_dim)
        optimizer.add_param_group({"params": self.adversary.parameters()})
        if self.reconstruction_strength != 0.:
            self.decoder = build_neural_net(encoding_dim, self.rec_dim, output_dim=linear_layers[0].in_features)
            optimizer.add_param_group({"params": self.decoder.parameters()})

        trained_encoder = self.base_fit(self.encoder, optimizer, train_loader, val_loader, nr_epochs, data_info,
                                        with_metrics=False)
        return torch.nn.Sequential(trained_encoder, *self.classifier)

    def calculate_loss(self, encoded_repr, label, sens, feat=None, **kwargs):
        logit = self.classifier(encoded_repr)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logit, label.float())

        adv_feat = grad_reverse(encoded_repr)
        if self.label_given:
            adv_feat = torch.cat([adv_feat, label], dim=-1)
        adv_logit = self.adversary(adv_feat)
        sens_feat = torch.argmax(sens, dim=1).long()
        adv_loss = torch.nn.functional.cross_entropy(adv_logit, sens_feat)
        loss += self.fairness_strength * adv_loss

        if self.reconstruction_strength != 0.:
            assert self.decoder is not None
            rec = self.decoder(encoded_repr)
            rec_loss = torch.sqrt(torch.nn.functional.mse_loss(rec, feat))
            loss += self.reconstruction_strength * rec_loss

        return loss

    def loop(self, model, optimizer, dataloader, data_info=None, stage=None, _with_metrics=None):
        self.base_loop(model, optimizer, dataloader, data_info, stage, with_metrics=False)
