"""opacus: Training with Sample-Level Differential Privacy using Opacus Privacy Engine."""

import logging
from typing import List, Tuple

from opacus_fl.task import Net, get_weights

from flwr.common import Context, Metrics, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg
from opacus_fl.my_strategy import CustomFedAvg
from opacus_fl.task import Net, get_weights, set_weights, test, get_transforms
import toml 
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader

# Load config
config = toml.load("/home/devaganthan.ss/Desktop/DPO-HPO/Baselines/opacus-non-private-baseline/pyproject.toml")
fraction_ft = config["train-settings"]["fraction-fit"]


# Opacus logger seems to change the flwr logger to DEBUG level. Set back to INFO
logging.getLogger("flwr").setLevel(logging.INFO)


def get_evaluate_fn(testloader, device):
    """Return an evaluation function which can be used to evaluate the global model"""

    def evaluate(server_round, parameters_ndarrays, config):

        net = Net()
        set_weights(net, parameters_ndarrays)
        net.eval()
        net.to(device)
        loss, accuracy = test(net, testloader, device)

        return loss, {"cen_accuracy": accuracy}
    
    return evaluate

# CHANGE
def fit_config_fn(rnd: int) -> dict:
    """Return a configuration with the requested number of rounds."""
    return {"rnd": rnd}


def weighted_average_eval(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    return {"accuracy": sum(accuracies) / sum(examples)}

def weighted_average_fit(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    losses = [num_examples * m["loss"] for num_examples, m in metrics]
    sum_examples = sum([num_examples for num_examples, _ in metrics])
    return {"train_loss": sum(losses)/sum_examples, "train_accuracy": sum(accuracies) / sum_examples}


def server_fn(context: Context) -> ServerAppComponents:
    num_rounds = context.run_config["num-server-rounds"]

    ndarrays = get_weights(Net())
    parameters = ndarrays_to_parameters(ndarrays)

    # Load Global Test Set
    testset = load_dataset("uoft-cs/cifar10")["test"]
    trainset = load_dataset("uoft-cs/cifar10")["train"]

    testloader = DataLoader(testset.with_transform(get_transforms(train=False)), batch_size=32)


    strategy = CustomFedAvg(
        fraction_fit= fraction_ft, 
        evaluate_metrics_aggregation_fn=weighted_average_eval,
        initial_parameters=parameters,
        evaluate_fn=get_evaluate_fn(testloader, device= "cuda:0" if torch.cuda.is_available() else "cpu"),
        on_fit_config_fn=fit_config_fn, # CHANGE
        fit_metrics_aggregation_fn=weighted_average_fit
    )

    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(config=config, strategy=strategy)


app = ServerApp(server_fn=server_fn)
