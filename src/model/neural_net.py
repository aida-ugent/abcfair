import torch


def build_neural_net(input_dim, dim, output_dim=1, dropout_strength=0.2):
    neural_net = []
    for hidden_dim in dim:
        neural_net.append(torch.nn.Linear(input_dim, hidden_dim))
        neural_net.append(torch.nn.ReLU())
        neural_net.append(torch.nn.Dropout(dropout_strength))
        input_dim = hidden_dim
    neural_net.append(torch.nn.Linear(input_dim, output_dim))
    neural_net = torch.nn.Sequential(*torch.nn.ModuleList(neural_net))
    return neural_net


def build_optimizer(model, lr=0.0001, weight_decay_str=0.01):
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay_str)
