from collections import namedtuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, Lambda, Compose
import matplotlib.pyplot as plt
from torchsummary import summary
import torch.nn.utils.prune as prune

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))


# Download test data from open datasets.
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)


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

test_dataloader = DataLoader(test_data, batch_size=16)
loss_fn = nn.CrossEntropyLoss()

#.................................................................................................#

# #Starting filter pruning

model = torch.load('/home/tmahmud/Co-Design Tasks/Lab3_renew/Lab-3-Codesign/model_lr_0.05_bs_16_acc91.7.pth')

print("Printing Summary of the Original Model")
print("..................................")
summary(model, input_size=(1, 28, 28))

amount = 0.25

for name, module in model.named_modules():
    if isinstance(module, torch.nn.Conv2d):
        prune.ln_structured(module, name="weight", amount=amount, n=1, dim=0)
        prune.remove(module, 'weight')

test_acc = test(test_dataloader,model, loss_fn, device)

print("Test accuracy: {} with filter pruning amount:{}".format(test_acc,amount))


#.................................................................................................................#
#Starting channel pruning

amount = 0.4

for name, module in model.named_modules():
    if isinstance(module, torch.nn.Conv2d):
        prune.ln_structured(module, name="weight", amount=amount, n=1, dim=1)
        prune.remove(module, 'weight')

print("Channel Pruning amount: {}".format(amount))
test_acc = test(test_dataloader,model, loss_fn, device)

print("Test accuracy: {} after channel pruning amount:{}".format(test_acc,amount))
torch.save(model, "channel_0.4_filter_0.25_pruned_model.pth")

print("Printing Summary of the Channel Pruned Model")
print("..................................")
summary(model, input_size=(1, 28, 28))
