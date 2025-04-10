import os
import itertools
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
from torchvision.transforms import v2
import numpy as np
import matplotlib.pyplot as plt
import wandb  # For optional experiment tracking/logging
import json

import os

from medmnist import PathMNIST, DermaMNIST, BloodMNIST, OCTMNIST

from data_augmentation import *
from resnet import CustomResNet

os.makedirs("saved_models", exist_ok=True)
os.makedirs("plots", exist_ok=True)
os.makedirs("models", exist_ok=True)

# Download datasets using MedMNIST.
print("Downloading datasets...")
path_dataset = PathMNIST(split="train", download=True)
derma_dataset = DermaMNIST(split="train", download=True)
blood_dataset = BloodMNIST(split="train", download=True)
retina_dataset = OCTMNIST(split="train", download=True)  # Using OCTMNIST as retina dataset

# Concatenate and augment datasets
print("Concatenating and augmenting datasets...")
full_dataset = ConcatDataset(path_dataset, derma_dataset, blood_dataset, retina_dataset)
dataset_size = len(full_dataset)
print(f"Total dataset size after augmentation: {dataset_size}")

# Split dataset: 70% training, 10% validation, 20% test
train_size = int(0.7 * dataset_size)
val_size = int(0.1 * dataset_size)
test_size = dataset_size - train_size - val_size

print("Splitting dataset into train, validation, and test sets...")
train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])
print(f"Train: {len(train_dataset)} | Validation: {len(val_dataset)} | Test: {len(test_dataset)}")

# Define grid search hyperparameters
grid_params = {
    "num_epochs": [3, 5],
    "lr": [0.001, 0.0005],
    "batch_size": [32, 64],
    "num_blocks": [3, 5, 10],
    "base_channels": [32, 64],
    "kernel_size": [3, 5]
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
criterion = nn.CrossEntropyLoss()

results_summary = []

# -------------------
# Begin grid search over hyperparameter combinations.
config_counter = 0
num_classes = 28
print("Everything is good. Start your engine!")
for (num_epochs, lr, batch_size, num_blocks, base_channels, kernel_size) in itertools.product(
        grid_params["num_epochs"],
        grid_params["lr"],
        grid_params["batch_size"],
        grid_params["num_blocks"],
        grid_params["base_channels"],
        grid_params["kernel_size"]):

    config = {
        "num_epochs": num_epochs,
        "lr": lr,
        "batch_size": batch_size,
        "num_blocks": num_blocks,
        "base_channels": base_channels,
        "kernel_size": kernel_size
    }
    config_str = f"config{config_counter}_epochs{num_epochs}_lr{lr}_bs{batch_size}_blocks{num_blocks}_channels{base_channels}_ks{kernel_size}"
    print(f"\nRunning configuration: {config_str}")
    config_counter += 1

    # DataLoaders for this configuration
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model with current hyperparameters.
    model = CustomResNet(in_channels=3, num_blocks=num_blocks, base_channels=base_channels,
                         kernel_size=kernel_size, num_classes=num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Lists to track accuracy (and loss if desired) for each epoch.
    train_accuracies = []
    val_accuracies = []

    # Training and validation loop for each epoch.
    for epoch in range(1, num_epochs + 1):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            labels = labels.squeeze()  # Ensure labels are of shape [batch_size]
            optimizer.zero_grad()
            outputs = model(images)
            
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)

            # Training accuracy computation
            _, predicted = torch.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        train_loss = running_loss / len(train_loader.dataset)
        train_acc = correct_train / total_train
        train_accuracies.append(train_acc)

        # Validation loop
        model.eval()
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                labels = labels.squeeze()  # Ensure labels are of shape [batch_size]
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()
        val_acc = correct_val / total_val
        val_accuracies.append(val_acc)

        print(f"{config_str} | Epoch [{epoch}/{num_epochs}] | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

        # Save model checkpoint after each epoch.
        model_save_path = os.path.join("saved_models", f"model_{config_str}_epoch{epoch}.pth")
        torch.save(model.state_dict(), model_save_path)
        print(f"Saved model: {model_save_path}")

    # Final test evaluation for this configuration.
    model.eval()
    correct_test = 0
    total_test = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            labels = labels.squeeze()  # Ensure labels are of shape [batch_size]
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total_test += labels.size(0)
            correct_test += (predicted == labels).sum().item()
    test_acc = correct_test / total_test
    print(f"{config_str} | Test Accuracy: {test_acc:.4f}")

    # Determine best validation performance for this configuration.
    best_val_acc = max(val_accuracies)
    best_epoch = val_accuracies.index(best_val_acc)

    # Save results for this configuration.
    results_summary.append({
        "params": config,
        "config_str": config_str,
        "best_val_acc": best_val_acc,
        "best_epoch": best_epoch,
        "val_accuracies": val_accuracies,
        "train_accuracies": train_accuracies,
        "test_acc": test_acc,
        "last_model_path": model_save_path  # last saved checkpoint for reference
    })

    # Optional: Log metrics to wandb
    # wandb.log({"config": config, "best_val_acc": best_val_acc, "test_acc": test_acc})

# -------------------
# After grid search, summarize all results.
# Write the full results to a JSON file.
with open("grid_search_summary.json", "w") as f:
    json.dump(results_summary, f, indent=4)
print("\nSaved grid search summary to grid_search_summary.json")

# Sort results in descending order by best validation accuracy
results_summary_sorted = sorted(results_summary, key=lambda x: x["best_val_acc"], reverse=True)
best_result = results_summary_sorted[0]

print("\nBest hyperparameter configuration:")
print(best_result["params"])
print(f"Best validation accuracy: {best_result['best_val_acc']} at epoch {best_result['best_epoch']+1}")
print(f"Test accuracy for best config: {best_result['test_acc']}")

# -------------------
# Plot validation accuracy evolution for the best model and save the plot.
plt.figure(figsize=(8, 6))
epochs = range(1, len(best_result["val_accuracies"]) + 1)
plt.plot(epochs, best_result["val_accuracies"], label="Validation Accuracy", marker='o')
plt.title("Validation Accuracy Over Epochs (Best Model)")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.grid(True)
plt.legend()
plt.tight_layout()
plot_path = os.path.join("plots", "best_model_val_accuracy.png")
plt.savefig(plot_path)
plt.close()
print(f"Saved validation accuracy plot for best model to {plot_path}")