import pathlib
import torch
import numpy as np
import random
import time
import wandb
import argparse

from dataset import build_data
from fairness_method import build_method
from model.model_tuner import TorchTuner
from model.neural_net import build_neural_net, build_optimizer
from evaluator.evaluator import Evaluator
from utils.utils import import_config, setup_logging


def configure(args):
    # Load config and setup logging (with wandb).
    config = {}
    import_config(config, args.config)
    if 'data_config' in config:
        # Config relating to the data was modularized to a separate file, but the use of this is optional.
        import_config(config, config['data_config'])
    setup_logging(config)

    # Load config back from wandb, e.g. in case we are running a sweep.
    config = wandb.config

    if config is None or len(vars(config)) == 0:
        raise ValueError("A config file was not provided, neither as an argument nor through wandb.")
    else:
        print(config)

    if 'num_threads' in config:
        torch.set_num_threads(config['num_threads'])

    if 'seed' in config:
        random.seed(config["seed"])
        np.random.seed(config["seed"])
        torch.manual_seed(config["seed"])
        torch.cuda.manual_seed(config["seed"])

    if 'device' not in config:
        config['device'] = 'cpu'

    return config


def run(config):
    # Setting up the dataset.
    data = build_data(**config["data"], device=config['device'])
    data.setup()

    # Define the method.
    fairness_method = build_method(**config["method"], device=config['device'])

    # Execute fairness preprocessing.
    try:
        start_time = time.perf_counter()
        data.train_data = fairness_method.preprocess(data.train_data)
        if hasattr(fairness_method, 'feat_transform'):
            # Some preprocessing methods transform all features.
            data.val_data.feat = fairness_method.feat_transform(data.val_data.feat, data.val_data.sens)
            data.test_data.feat = fairness_method.feat_transform(data.test_data.feat, data.test_data.sens)
        wandb.log({"preprocess_time": time.perf_counter() - start_time})
    except NotImplementedError:
        pass

    # Initialize the model and the tuning loop.
    model = build_neural_net(data.feat_dim, **config["model"])
    optimizer = build_optimizer(model, **config["optim"])
    tuner = TorchTuner(device=config['device'])
    data_info = data.info()

    # Execute training (possibly with modifications from inprocessing).
    try:
        tuner = fairness_method.inprocess(tuner)
    except NotImplementedError:
        pass
    start_time = time.perf_counter()
    model = tuner.fit(model, optimizer, data.train_dataloader(), data.val_dataloader(),
                      config["training"]["max_epochs"], data_info)
    wandb.log({"training_time": time.perf_counter() - start_time})

    # Execute postprocessing.
    try:
        start_time = time.perf_counter()
        predict_fn = fairness_method.postprocess(model, data.train_dataloader())
        wandb.log({"postprocess_time": time.perf_counter() - start_time})
    except NotImplementedError:
        predict_fn = tuner.predict


    # Testing the model
    test_dataloader = data.test_dataloader()
    model = model.to(config['device'])
    evaluator = Evaluator('test', device=config['device'], data_info=data_info)
    for feat, sens, label, sens_formats in test_dataloader:
        predictions = predict_fn(model, feat, sens)
        evaluator.update(predictions, label, sens_formats)
    wandb.log(evaluator.compute())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=pathlib.Path, help="Path to .yaml config file.", nargs='?')
    args = parser.parse_args()
    config = configure(args)
    run(config)


if __name__ == "__main__":
    main()
