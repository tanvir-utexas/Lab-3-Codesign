import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import argparse
from torch.autograd import Variable


# Hyper Parameters
input_size = 28
num_classes = 10
num_epochs = 1
batch_size = 32
learning_rate = 1e-3
L1_lambda = 0.9

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class MyConvNet(nn.Module):
    def __init__(self):
        super(MyConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.act1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(16, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.act2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.lin1 = nn.Linear(7*7*64, 7*7*16)
        self.lin2 = nn.Linear(7*7*16, 10)
    
    def forward(self, x):
        c1 = self.conv1(x)
        b1 = self.bn1(c1)
        a1 = self.act1(b1)
        p1 = self.pool1(a1)
        c2 = self.conv2(p1)
        b2 = self.bn2(c2)
        a2 = self.act2(b2)
        p2 = self.pool2(a2)
        flt = p2.view(p2.size(0), -1)
        l1 = self.lin1(flt)
        out = self.lin2(l1)
        return out

model = MyConvNet().to(device)

print(model)

x = torch.randn(batch_size, 1, 28, 28).to(device)


print(model(x).shape)

train_dataset = datasets.MNIST(root='./Datasets',
                               train= True,
                               transform= transforms.ToTensor(),
                               download = True) 

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = datasets.MNIST(root='./Datasets',
                              train = False,
                              transform = transforms.ToTensor(),
                              download= True)

test_loader = DataLoader(dataset=test_dataset,
                         batch_size = batch_size,
                         shuffle = False)


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)


def check_accuracy(loader, model):
  num_correct = 0
  num_samples = 0

  model.eval()

  with torch.no_grad():
    for batch, (x, y) in enumerate(loader):
      x = Variable(x)
      y = Variable(y)

      x = x.to(device=device)
      y = y.to(device=device)

      scores = model(x)

      _, predictions = scores.max(1)
      
      num_correct += (predictions == y).sum()
      num_samples += predictions.size(0)
      

    print(f"Got {num_correct}/ {num_samples} with accuray {float(num_correct)/float(num_samples)*100:0.2f}")


for epoch in range(num_epochs):
  for batch_idx, (data, targets) in enumerate(train_loader):
    data = Variable(data)
    targets = Variable(targets)

    data = data.to(device=device)
    targets = targets.to(device=device)
    
    scores = model(data)
    loss = criterion(scores, targets)

    #Adding L1_norm here
    for name, param in model.named_parameters():
      if 'weight' in name: 
        L1_1 = Variable(param, requires_grad= True)
        L1_2 = torch.norm(L1_1, 1)
        L1_3 = L1_lambda * L1_2

        loss = loss + L1_3;       
        #print(L1_2)

    optimizer.zero_grad()
    loss.backward()

    optimizer.step()

    if (batch_idx+1) % 100 == 0:
      print('Epoch: [{}/{}], Step: [{}/{}], Loss: {:0.4f}'. format(epoch + 1, num_epochs, batch_idx + 1, len(train_dataset) // batch_size, loss.data.item()))
    
  print(f"Epoch:{epoch+1}, Training accuracy:")
  check_accuracy(train_loader, model)

  print(f"Epoch:{epoch+1}, Testing accuracy:")
  check_accuracy(test_loader, model)

  torch.save(model, '/home/tmahmud/My_codes/Lab3/MyConvNet.pth')