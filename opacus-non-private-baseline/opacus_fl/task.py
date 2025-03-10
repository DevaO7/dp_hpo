"""opacus: Training with Sample-Level Differential Privacy using Opacus Privacy Engine."""

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor, RandomCrop, RandomHorizontalFlip
from tqdm import tqdm
import toml

fds = None  # Cache FederatedDataset

train_config = toml.load("/home/devaganthan.ss/Desktop/DPO-HPO/Baselines/opacus-non-private-baseline/pyproject.toml")
private = train_config["train-settings"]["private"]


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# CHANGE
def get_transforms(train=True):
    
    if train:
        pytorch_transforms = Compose([
        RandomCrop(32, padding=4),
        RandomHorizontalFlip(),
        ToTensor(),
        Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    else:
        pytorch_transforms = Compose([
        ToTensor(),
        Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    def apply_transforms(batch):
        batch["img"] = [pytorch_transforms(img) for img in batch["img"]]
        return batch
    
    return apply_transforms



def get_weights(net):
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_weights(net, parameters):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)


def load_data(partition_id: int, num_partitions: int):
    print('Num partitions:', num_partitions)
    global fds
    if fds is None:
        partitioner = IidPartitioner(num_partitions=num_partitions)
        fds = FederatedDataset(
            dataset="uoft-cs/cifar10",
            partitioners={"train": partitioner},
        )

    partition = fds.load_partition(partition_id)
    # Divide data on each node: 80% train, 20% test
    partition_train_test = partition.train_test_split(test_size=0.2, seed=42)


    partition_train_test = partition_train_test.with_transform(get_transforms(train=True))
    train_loader = DataLoader(
        partition_train_test["train"], batch_size=32, shuffle=True
    )
    test_loader = DataLoader(partition_train_test["test"], batch_size=32)
    return train_loader, test_loader


def train(net, train_loader, privacy_engine, optimizer, target_delta, device, epochs=1):
    criterion = torch.nn.CrossEntropyLoss().to(device)
    net.to(device)
    net.train()
    running_loss = 0.0
    epsilon = None
    for _ in range(epochs):
        for batch in tqdm(train_loader, "Training"):
            images = batch["img"]
            labels = batch["label"]
            optimizer.zero_grad()
            loss = criterion(net(images.to(device)), labels.to(device))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    if private:
        epsilon = privacy_engine.get_epsilon(delta=target_delta)
    return epsilon, avg_loss


def test(net, test_loader, device):
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    with torch.no_grad():
        for batch in tqdm(test_loader, "Testing"):
            images = batch["img"].to(device)
            labels = batch["label"].to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
    accuracy = correct / len(test_loader.dataset)
    loss /= len(test_loader)
    return loss, accuracy
