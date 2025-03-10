"""opacus: Training with Sample-Level Differential Privacy using Opacus Privacy Engine."""

import warnings

import torch
from opacus import PrivacyEngine
from opacus_fl.task import Net, get_weights, load_data, set_weights, test, train
import logging

from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
import toml

warnings.filterwarnings("ignore", category=UserWarning)

train_config = toml.load("/home/devaganthan.ss/Desktop/DPO-HPO/Baselines/opacus-non-private-baseline/pyproject.toml")
private = train_config["train-settings"]["private"]
wandb_logging = train_config["train-settings"]["wandb_log"]

class FlowerClient(NumPyClient):
    def __init__(
        self,
        train_loader,
        test_loader,
        target_delta,
        noise_multiplier,
        max_grad_norm,
    ) -> None:
        super().__init__()
        self.model = Net()
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.target_delta = target_delta
        self.noise_multiplier = noise_multiplier
        self.max_grad_norm = max_grad_norm

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def fit(self, parameters, config):
        model = self.model
        set_weights(model, parameters)

        # optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

        # CHANGE
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9) 
        server_round = config["rnd"]
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
        for _ in range(server_round):
            scheduler.step()
        epsilon = 0
        if private:
            print('Yes it is getting into the private training')
            privacy_engine = PrivacyEngine(secure_mode=False)
            (
                model,
                optimizer,
                self.train_loader,
            ) = privacy_engine.make_private(
                module=model,
                optimizer=optimizer,
                data_loader=self.train_loader,
                noise_multiplier=self.noise_multiplier,
                max_grad_norm=self.max_grad_norm,
            )

            epsilon, loss = train(
                model,
                self.train_loader,
                privacy_engine,
                optimizer,
                self.target_delta,
                device=self.device,
            )

            if epsilon is not None:
                print(f"Epsilon value for delta={self.target_delta} is {epsilon:.2f}")
            else:
                print("Epsilon value not available.")
            print('Private training Loss:', loss)
        else: 
            print('Non Private Training')
            epsilon, loss = train(
                model,
                self.train_loader,
                None,
                optimizer,
                None,
                device=self.device,
            )
            print('Non-private training Loss:', loss) 

        # CHANGE
        loss, accuracy = test(model, self.train_loader, self.device)
          
        return (get_weights(model), len(self.train_loader.dataset), {"loss": loss, "accuracy": accuracy, "epsilon": epsilon})

    def evaluate(self, parameters, config):
        set_weights(self.model, parameters)
        loss, accuracy = test(self.model, self.test_loader, self.device)

        return loss, len(self.test_loader.dataset), {"accuracy": accuracy}

# Change: Trying to have statefullness 
def client_fn(context: Context):
    partition_id = context.node_config["partition-id"]
    noise_multiplier = 1.0 if partition_id % 2 == 0 else 1.5

    train_loader, test_loader = load_data(
        partition_id=partition_id, num_partitions=context.node_config["num-partitions"]
    )
    return FlowerClient(
        train_loader,
        test_loader,
        context.run_config["target-delta"],
        noise_multiplier,
        context.run_config["max-grad-norm"],
    ).to_client()


app = ClientApp(client_fn=client_fn)
