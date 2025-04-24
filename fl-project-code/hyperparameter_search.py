import copy
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import MNIST
from torch.utils.data import random_split
from torch import nn, optim
from torch.utils.data import random_split, DataLoader, Subset
import numpy as np
import torch.nn.functional as F
from opacus.accountants import RDPAccountant
import wandb
import yaml
import matplotlib.pyplot as plt
import random
from datetime import datetime
import json

random.seed(42)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Net(nn.Module):
    def __init__(self, num_classes=10):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        
        self.pool = nn.MaxPool2d(2, 2)
        # self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(64 * 7 * 7, num_classes)  # for 28x28 MNIST input

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool(x)
        x = F.relu(self.conv5(x))
        # x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    

def partition_data(dataset, n_clients, iid=True):
    data_indices = np.arange(len(dataset))
    labels = np.array([dataset[i][1] for i in data_indices])

    client_data_indices = []

    if iid:
        # Shuffle and split equally
        np.random.shuffle(data_indices)
        split_sizes = [len(dataset) // n_clients] * n_clients
        for i in range(len(dataset) % n_clients):
            split_sizes[i] += 1
        split_indices = np.array_split(data_indices, np.cumsum(split_sizes)[:-1])
        client_data_indices = [list(indices) for indices in split_indices]
    else:
        # Non-IID partitioning: each client gets data from only 2 random digits
        classes = np.unique(labels)
        class_indices = {cls: np.where(labels == cls)[0] for cls in classes}

        for _ in range(n_clients):
            chosen_classes = np.random.choice(classes, 2, replace=False)
            indices = []
            for cls in chosen_classes:
                cls_indices = class_indices[cls]
                selected = np.random.choice(cls_indices, len(cls_indices) // n_clients, replace=False)
                indices.extend(selected)
            client_data_indices.append(indices)

    # Create Subset datasets for each client
    client_datasets = [Subset(dataset, indices) for indices in client_data_indices]
    return client_datasets

def load_data(n_clients=10, iid=True):
    # Define the transform
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Load the MNIST dataset
    train_dataset = MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = MNIST(root='./data', train=False, download=True, transform=transform)

    # Split into train/test

    # Partition the training data among clients
    client_datasets = partition_data(train_dataset, n_clients, iid=iid)

    return client_datasets, test_dataset


def train(model, train_loader, criterion, optimizer, epochs=1):
    avg_loss = 0.0
    for epoch in range(epochs):
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            avg_loss += loss.item()
            optimizer.step()
    avg_loss /= len(train_loader)
    return avg_loss

def server_test(model, test_dataset):
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    with torch.no_grad():
        correct = 0
        total = 0
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    accuracy = correct / total
    return accuracy  # Return accuracy to log it later


def create_model(model_config=None):
    model = Net()
    if model_config:
        model.load_state_dict(model_config)
    model.to(device)
    model.train()
    return model


def add_dp_noise(aggregated_delta, noise_multiplier, clip_threshold):
    noisy_delta = copy.deepcopy(aggregated_delta)
    for key in noisy_delta.keys():
        param = noisy_delta[key]
        noise = torch.normal(
            mean=0.0,
            std=noise_multiplier * clip_threshold,
            size=param.shape,
            device=param.device
        )
        noisy_delta[key] = param + noise
    return noisy_delta


def fed_avg_dp(client_weights, local_size, noise_multiplier, clip_threshold):
    empty_weight = copy.deepcopy(client_weights[0])
    for key in empty_weight.keys():
        empty_weight[key] = torch.zeros_like(empty_weight[key])
    tot_size = sum(local_size)
    for i, weight in enumerate(client_weights):
        for key in weight.keys():
            empty_weight[key] += weight[key] * (local_size[i] / tot_size)

    noisy_delta = add_dp_noise(empty_weight, noise_multiplier, clip_threshold)

    
    return noisy_delta

def compute_delta(local_model, global_model):
    delta = {}
    for key in local_model.keys():
        delta[key] = local_model[key] - global_model[key]
    return delta

def clip_delta(delta, clip_threshold):
    total_norm_sq = 0.0
    for key in delta:
        total_norm_sq += torch.sum(delta[key] ** 2)
    total_norm = torch.sqrt(total_norm_sq)
    
    scaling_factor = min(1.0, clip_threshold / (total_norm + 1e-6))
    
    for key in delta:
        delta[key] = delta[key] * scaling_factor
    return delta


def sample_from_distribution(mu):
    k = np.random.poisson(mu)
    return k


def central_dp_federated_learning(learning_rate):

    # Configurations
    NOISE_MULTIPLIER = 0.5
    DELTA = 1e-5
    CLIP_THRESHOLD = 0.005
    server_rounds = 600
    num_clients = 50
    no_of_clients_sampled = 5
    SAMPLE_RATE = no_of_clients_sampled / num_clients

    # Load data
    client_datasets, test_dataset = load_data(num_clients, iid=True)


    model = create_model(None)  # Initialize the model
    # Load the model config
    global_model = model.state_dict()

    accountant = RDPAccountant()


    best_accuracy = 0
    for round in range(server_rounds):
        client_weights = []
        local_size = []
        sampled_client_ids = np.random.choice(range(num_clients), size=no_of_clients_sampled, replace=False)

        losses = []

        # Clients train the model (happens in sequential order)
        for client_id in sampled_client_ids:
            model = create_model(global_model)
            client_data = client_datasets[client_id]
            local_size.append(len(client_data))
            train_loader = DataLoader(client_data, batch_size=256, shuffle=True)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.SGD(model.parameters(), lr=learning_rate)
            loss = train(model, train_loader, criterion, optimizer)
            losses.append(loss)
            delta = compute_delta(local_model=model.state_dict(), global_model=global_model)
            clipped_delta = clip_delta(delta, CLIP_THRESHOLD)
            client_weights.append(clipped_delta)

        noisy_aggregate_deltas = fed_avg_dp(client_weights, local_size, NOISE_MULTIPLIER, CLIP_THRESHOLD)

        updated_global_model = copy.deepcopy(global_model)
        for key in updated_global_model.keys():  
            updated_global_model[key] = noisy_aggregate_deltas[key] + global_model[key]
        
        global_model = copy.deepcopy(updated_global_model)

        model = create_model(global_model)
        accountant.step(noise_multiplier=NOISE_MULTIPLIER, sample_rate=SAMPLE_RATE)
        
        # Get test accuracy from the modified server_test function
        test_accuracy = server_test(model, test_dataset)

        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy

        # Print round number and accuracy
        # print(f"Round {round + 1}/{server_rounds}, Best Accuracy: {best_accuracy:.4f}")
        
        
    
    # Compute the final Epsilon
    final_epsilon, final_alpha = accountant.get_privacy_spent(delta=DELTA)

    # Return Privacy Budget for  Q(x)
    return final_epsilon, best_accuracy, final_alpha, DELTA





if __name__ == "__main__":
    name = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    wandb_logging = True
    if wandb_logging == True:
        wandb.init(project="dp_hpo_project", name=f"learning_rate_search-baseline-{name}")

    # Parameters of the Experiment
    n = 5
    no_of_candidates = 20
    privacy_budgets = [96, 99, 102, 105]
    mu_values = [1, 3, 6, 15]
    learning_rates = np.logspace(np.log10(0.0005), np.log10(0.02), num=no_of_candidates)
    expected_epsilon = 96.05
    expected_alpha = 1.3

    
    accuracy_epsilon = {}

    # Do this 500 times
    for i, target_epsilon in enumerate(privacy_budgets):

        mu = mu_values[i]


        print('--'*50)
        print(f"Running with target epsilon: {target_epsilon}")
        print(f"Running with mu: {mu}")
        

        avg_best_accuracy = []

        for l in range(n):

            # Print trial number
            print('**'*50)
            print(f"Running Trial {l+1}/{n}")


            # Sampling K from the Distribution
            K = sample_from_distribution(mu)
            # print(f"Sampled K: {K}")

            if wandb_logging:
                wandb.log({
                    "K": K,})

            if K == 0:
                K = 1
            
            if l == 0:
                K = 1


            epsilons = []

            accuracies = []

            sample_learning_rate = []

            # Run the algorithm for K number of times
            for k in range(K):
                # Sample a random hyperparameter configuration
                j = np.random.randint(0, no_of_candidates)

                # Pick a Random Candidate
                learning_rate = learning_rates[j]
                # print(f"Running with learning rate: {learning_rate}")

                # Perform Federated Learning
                epsilon, test_accuracy, alpha, delta = central_dp_federated_learning(learning_rate)


                if wandb_logging:
                    wandb.log({
                        "epsilon": epsilon,
                        "test_accuracy": test_accuracy,
                        "learning_rate": learning_rate,
                    })

                # Store the accuracy and epsilon in the list
                epsilons.append(epsilon)
                accuracies.append(test_accuracy)
                sample_learning_rate.append(learning_rate)
        

            # Compute the average accuracy and epsilon
            avg_best_accuracy.append(np.max(accuracies))

            best_learning_rate = sample_learning_rate[np.argmax(accuracies)]
            # print(f"Best Learning Rate: {best_learning_rate}")

            if wandb_logging:
                wandb.log({
                    "best_learning_rate": best_learning_rate,
                })
            
            # Print the list of accuracies
            # print(f"Accuracies: {accuracies}")

        accuracy_epsilon[target_epsilon] = np.mean(avg_best_accuracy)
        # print(f"Average Best Accuracy: {np.mean(avg_best_accuracy)}")
        # print(f"Average Epsilon: {np.mean(epsilons)}")
        # print(f"Avg Best Accuracy List: {avg_best_accuracy}")
        # print(f"Alpha Value: {alpha}")
        # print(f"Expected Epsilon: {expected_epsilon}")
        # print(f"Expected Alpha: {expected_alpha}")

        with open('accuracy_epsilon.json', 'w') as f:
            json.dump({"accuracy_epsilon": accuracy_epsilon,
                "target_epsilon": target_epsilon,
                "avg_best_accuracy": np.mean(avg_best_accuracy),
                "avg_epsilon": np.mean(epsilons),
                "alpha": alpha,
                "expected_epsilon": expected_epsilon,
                "expected_alpha": expected_alpha,
                "avg_best_accuracy_list": avg_best_accuracy,
                "mu": mu,
            }, f)


    # Plot the average accuracy of the accuracy list (Wandb)
    plt.figure(figsize=(10, 6))
    plt.plot(accuracy_epsilon.keys(), accuracy_epsilon.values(), marker='o')
    plt.xlabel('Privacy Budget (Epsilon)')
    plt.ylabel('Average Best Accuracy')
    plt.title('Average Best Accuracy vs Privacy Budget')
    plt.grid()
    plt.savefig(f"accuracy_vs_epsilon.png")
    # Log the results to Wandb
    # wandb.log({
    #     "accuracy_vs_epsilon": wandb.Image("accuracy_vs_epsilon.png")
    # })
    # wandb.finish()
    print("Finished logging to Wandb.")

    # save the dictionary accuracy_epsilon to a yaml file
    with open('accuracy_epsilon.json', 'w') as f:
        json.dump(accuracy_epsilon, f)
    print("Saved accuracy_epsilon to accuracy_epsilon.yaml")
