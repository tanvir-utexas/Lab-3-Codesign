from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, Lambda, Compose
import matplotlib.pyplot as plt
from Network import MyConvNet, MyConvNet_wo_bias

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))


training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)


# Download test data from open datasets.
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)


def train(dataloader, model, loss_fn, optimizer, device):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(dataloader, model, loss_fn, device):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return 100*correct

def train_model(batch_size=64, lr=1e-3):
    # Create data loaders.
    train_dataloader = DataLoader(training_data, batch_size=batch_size)
    test_dataloader = DataLoader(test_data, batch_size=batch_size)

    model = MyConvNet_wo_bias().to(device)
    print(model)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    epochs = 10
    scheduler = MultiStepLR(optimizer, milestones=[8], gamma=0.1)
    accuracy = 0
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer, device)
        accuracy = test(test_dataloader, model, loss_fn, device)
        scheduler.step()
    print("Done!")

    torch.save(model, f"model_lr_{lr}_bs_{batch_size}_acc{accuracy:>0.1f}.pth")

if __name__ == '__main__':
    train_model(batch_size=16, lr=5e-2)
    # train_model(batch_size=32, lr=5e-2)
    # train_model(batch_size=64, lr=5e-2)
    # train_model(batch_size=16, lr=1e-2)
    # train_model(batch_size=32, lr=1e-2)
    # train_model(batch_size=64, lr=1e-2)
    # train_model(batch_size=16, lr=2e-3)
    # train_model(batch_size=32, lr=2e-3)
    # train_model(batch_size=64, lr=2e-3)
