from flwr.common import FitRes, Parameters, parameters_to_ndarrays, Scalar
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg

import torch
import json
import wandb
from datetime import datetime
import toml
from .task import Net, set_weights
from typing import Union, Optional

train_config = toml.load("/home/devaganthan.ss/Desktop/DPO-HPO/Baselines/opacus-non-private-baseline/pyproject.toml")
wandb_logging = train_config["train-settings"]["wandb_log"]
private = train_config["train-settings"]["private"]

class CustomFedAvg(FedAvg):
    """A strategy that keeps the core functionality of FedAvg unchanged but enables
    additional features such as: Saving global checkpoints, saving metrics to the local
    file system as a JSON, pushing metrics to Weight & Biases.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # A dictionary that will store the metrics generated on each round
        self.results_to_save = {}

        if private:
            train_type = "private"
        else:
            train_type = "non-private"

        # Log those same metrics to W&B
        if wandb_logging:
            name = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
            wandb.init(project="dp_hpo_project", name=f"{train_type}-baseline-{name}")

    # def aggregate_fit(
    #     self,
    #     server_round: int,
    #     results: list[tuple[ClientProxy, FitRes]],
    #     failures: list[tuple[ClientProxy, FitRes] | BaseException],
    # ) -> tuple[Parameters | None, dict[str, bool | bytes | float | int | str]]:
    #     """Aggregate received model updates and metrics, ave global model checkpoint."""

    #     # Call the default aggregate_fit method from FedAvg
    #     parameters_aggregated, metrics_aggregated = super().aggregate_fit(
    #         server_round, results, failures
    #     )

    #     ## Save new Global Model as a PyTorch checkpoint
    #     # Convert parameters to ndarrays
    #     ndarrays = parameters_to_ndarrays(parameters_aggregated)
    #     # Instantiate model
    #     model = Net()
    #     # Apply paramters to model
    #     set_weights(model, ndarrays)
    #     # Save global model in the standard PyTorch way
    #     torch.save(model.state_dict(), f"global_model_round_{server_round}")

    #     # Return the expected outputs for `aggregate_fit`
    #     return parameters_aggregated, metrics_aggregated

    def evaluate(
        self, server_round: int, 
        parameters: Parameters
    ) -> tuple[float, dict[str, bool | bytes | float | int | str]] | None:
        """Evaluate global model, then save metrics to local JSON and to W&B."""
        # Call the default behaviour from FedAvg
        loss, metrics = super().evaluate(server_round, parameters)


        if wandb_logging:
            # Store metrics as dictionary
            my_results = {"loss": loss, **metrics}
            # Insert into local dictionary
            self.results_to_save[server_round] = my_results

            # Save metrics as json
            with open("results_p.json", "w") as json_file:
                json.dump(self.results_to_save, json_file, indent=4)

            # Log metrics to W&B
            wandb.log(my_results, step=server_round)

        # Return the expected outputs for `evaluate`
        return loss, metrics
    
    # def evaluate_metrics_aggregation_fn(
    #     self, metrics: list[tuple[int, dict[str, bool | bytes | float | int | str]]]
    # ) -> dict[str, bool | bytes | float | int | str]:
    #     """Combine metrics from multiple clients into a single aggregated metric."""
    #     # Call the default behaviour from FedAvg
    #     metrics_aggregated = super().evaluate_metrics_aggregation_fn(metrics)

    #     # Return the expected outputs for `evaluate_metrics_aggregation_fn`
    #     return metrics_aggregated


    # CHANGE
    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list[Union[tuple[ClientProxy, FitRes], BaseException]],
    ) -> tuple[Optional[Parameters], dict[str, Scalar]]:
        parameters_aggregated, metrics_aggregated = super().aggregate_fit(server_round, results, failures)

        
        if wandb_logging:
            # Store metrics as dictionary
            my_results = metrics_aggregated
            # Insert into local dictionary
            self.results_to_save[server_round] = my_results

            # Save metrics as json
            with open("results_train.json", "w") as json_file:
                json.dump(self.results_to_save, json_file, indent=4)

            # Log metrics to W&B
            wandb.log(my_results, step=server_round)



        return parameters_aggregated, metrics_aggregated