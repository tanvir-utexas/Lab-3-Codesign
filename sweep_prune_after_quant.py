#Quantisation Functions

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
from Network import MyConvNet
import torch.nn.utils.prune as prune


QTensor = namedtuple('QTensor', ['tensor', 'scale', 'zero_point'])

def calcScaleZeroPoint(min_val, max_val, num_bits=8):
  # Calc Scale and zero point of next 
  qmin = 0.
  qmax = 2.**num_bits - 1.

  scale = (max_val - min_val) / (qmax - qmin)

  initial_zero_point = qmin - min_val / scale
  
  zero_point = 0
  if initial_zero_point < qmin:
      zero_point = qmin
  elif initial_zero_point > qmax:
      zero_point = qmax
  else:
      zero_point = initial_zero_point

  zero_point = int(zero_point)

  return scale, zero_point

def quantize_tensor(x, num_bits=8, min_val=None, max_val=None):
    
    if not min_val and not max_val: 
      min_val, max_val = x.min(), x.max()

    qmin = 0.
    qmax = 2.**num_bits - 1.

    scale, zero_point = calcScaleZeroPoint(min_val, max_val, num_bits)
    q_x = zero_point + x / scale
    q_x.clamp_(qmin, qmax).round_()
    q_x = q_x.round()
    
    return QTensor(tensor=q_x, scale=scale, zero_point=zero_point)

def dequantize_tensor(q_x):
    return q_x.scale * (q_x.tensor.float() - q_x.zero_point)


#Rework Forward pass of Linear and Conv Layers to support Quantisation

def quantizeLayer(x, layer, stat, scale_x, zp_x, num_bits=8, weights_save_name=None):
  # for both conv and linear layers

  # cache old values
  W = layer.weight.data
  B = layer.bias.data

  # quantise weights, activations are already quantised
  w = quantize_tensor(layer.weight.data, num_bits=num_bits)
  b = quantize_tensor(layer.bias.data, num_bits=num_bits)

  if weights_save_name:
      np.save(f'./weights_npy/fp_w_{weights_save_name}', W.cpu().numpy())
      qt_w = w.scale*(w.tensor - w.zero_point)
      np.save(f'./weights_npy/quant_w_{weights_save_name}', qt_w.cpu().numpy())

  layer.weight.data = w.tensor.float()
  layer.bias.data = b.tensor.float()

  # This is Quantisation Artihmetic
  scale_w = w.scale
  zp_w = w.zero_point
  scale_b = b.scale
  zp_b = b.zero_point
  
  scale_next, zero_point_next = calcScaleZeroPoint(min_val=stat['min'], max_val=stat['max'], num_bits=num_bits)

  # Preparing input by shifting
  X = x.float() - zp_x
  layer.weight.data = scale_x * scale_w*(layer.weight.data - zp_w)
  # Jeff: Here should be minus zero point
  layer.bias.data = scale_b*(layer.bias.data - zp_b)

  # All int computation
  x = (layer(X)/ scale_next) + zero_point_next

  # Reset weights for next forward pass
  layer.weight.data = W
  layer.bias.data = B
  
  return x, scale_next, zero_point_next


'''Get Max and Min Stats for Quantising Activations of Network.
This is done by running the network with around 1000 examples 
and getting the average min and max activation values before and after each layer.'''


# Get Min and max of x tensor, and stores it
def updateStats(x, stats, key):
  max_val = torch.max(x)
  min_val = torch.min(x)

  if key not in stats:
      stats[key] = {"max": max_val.sum(), "min": min_val.sum(), "total": 1}
  else:
      stats[key]['max'] += max_val.sum().item()
      stats[key]['min'] += min_val.sum().item()
      stats[key]['total'] += 1

  return stats

# Reworked Forward Pass to access activation Stats through updateStats function
def gatherActivationStats(model, x, stats):   
  stats = updateStats(x.clone().view(x.shape[0], -1), stats, 'conv1_before')
  x = model.conv1(x)
  stats = updateStats(x.clone().view(x.shape[0], -1), stats, 'conv1_after')

  x = model.act1(model.bn1(x))
  x = model.pool1(x)

  stats = updateStats(x.clone().view(x.shape[0], -1), stats, 'conv2_before')
  x = model.conv2(x)
  stats = updateStats(x.clone().view(x.shape[0], -1), stats, 'conv2_after')

  x = model.act2(model.bn2(x))
  x = model.pool2(x)

  x = x.view(x.size(0), -1) 

  stats = updateStats(x, stats, 'lin1_before')
  x = model.lin1(x)
  
  stats = updateStats(x, stats, 'lin2_before')

  x = model.lin2(x)

  stats = updateStats(x, stats, 'lin2_after')
  return stats


# Entry function to get stats of all functions.
def gatherStats(model, test_loader):
    device = 'cuda'
    
    model.eval()
    test_loss = 0
    correct = 0
    stats = {}
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            stats = gatherActivationStats(model, data, stats)
    
    final_stats = {}
    for key, value in stats.items():
      final_stats[key] = { "max" : value["max"] / value["total"], "min" : value["min"] / value["total"] }
    return final_stats


# Forward Pass for Quantised Inference

def quantForward(model, x, stats, quant_num_bits):
  
  # Quantise before inputting into incoming layers
  x = quantize_tensor(x, num_bits=quant_num_bits, min_val=stats['conv1_before']['min'],
                      max_val=stats['conv1_before']['max'])

  x, scale_next, zero_point_next = quantizeLayer(x.tensor, model.conv1, stats['conv1_after'],
                                                 x.scale, x.zero_point, num_bits=quant_num_bits,
                                                 weights_save_name='conv1')
  # Jeff: Because the input of bn should be FP, we need to convert x to FP
  x = dequantize_tensor(QTensor(tensor=x, scale=scale_next, zero_point=zero_point_next))

  x = model.act1(model.bn1(x))
  x = model.pool1(x)
  
  x = quantize_tensor(x, num_bits=quant_num_bits, min_val=stats['conv2_before']['min'], max_val=stats['conv2_before']['max'])
  x, scale_next, zero_point_next = quantizeLayer(x.tensor, model.conv2, stats['conv2_after'],
                                                 x.scale, x.zero_point, num_bits=quant_num_bits,
                                                 weights_save_name='conv2')
  # Jeff: Because the input of bn should be FP, we need to convert x to FP
  x = dequantize_tensor(QTensor(tensor=x, scale=scale_next, zero_point=zero_point_next))

  x = model.act2(model.bn2((x)))
  x = model.pool2(x)

  x = x.view(x.size(0), -1)

  x = quantize_tensor(x, num_bits=quant_num_bits, min_val=stats['lin1_before']['min'], max_val=stats['lin1_before']['max'])

  x, scale_next, zero_point_next = quantizeLayer(x.tensor, model.lin1, stats['lin2_before'],
                                                 x.scale, x.zero_point, num_bits=quant_num_bits,
                                                 weights_save_name='lin1')

  x, scale_next, zero_point_next = quantizeLayer(x, model.lin2, stats['lin2_after'], scale_next, zero_point_next,
                                                 num_bits=quant_num_bits, weights_save_name='lin2')

  x = dequantize_tensor(QTensor(tensor=x, scale=scale_next, zero_point=zero_point_next))

  return x

def NoQuantforward(model, x):
    c1 = model.conv1(x)
    b1 = model.bn1(c1)
    a1 = model.act1(b1)
    p1 = model.pool1(a1)
    c2 = model.conv2(p1)
    b2 = model.bn2(c2)
    a2 = model.act2(b2)
    p2 = model.pool2(a2)
    flt = p2.view(p2.size(0), -1)
    l1 = model.lin1(flt)
    out = model.lin2(l1)
    return out


def testQuant(model, test_loader, quant=False, stats=None, quant_num_bits=8):
    device = 'cuda'
       
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            if quant:
              pred = quantForward(model, data, stats, quant_num_bits)
            else:
              pred = NoQuantforward(model, data)

            test_loss += loss_fn(pred, target).item()
            correct += (pred.argmax(1) == target).type(torch.float).sum().item()
    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return 100. * correct / len(test_loader.dataset)

##starting the process

#if "__name__" == "__main__":
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))

# Download test data from open datasets.
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)

batch_size = 64
test_dataloader = DataLoader(test_data, batch_size=batch_size)
model_path = 'model_lr_0.05_bs_16_acc91.7.pth'
q_model = torch.load(model_path).to(device)
print("Loading model from {}".format(model_path))
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(q_model.parameters(), lr=1e-3)


#Test on non-quantized model
testQuant(q_model, test_dataloader, quant=False)

stats = gatherStats(q_model, test_dataloader)
print(stats)

#test on quantized model
quant_num_bits = 3
print('Quantization bits {}'.format(quant_num_bits))
testQuant(q_model, test_dataloader, quant=True, stats=stats, quant_num_bits=quant_num_bits)

channel_pruning_amounts = torch.range(0, 1, 0.1)

test_accuracy_filter_pruning = []

for amount in channel_pruning_amounts:
    q_model = torch.load(model_path).to(device)

    for name, module in q_model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            prune.ln_structured(module, name="weight", amount=amount.data.item(), n=1, dim=1)
            prune.remove(module, 'weight')
    stats = gatherStats(q_model, test_dataloader)
    print(stats)

    test_acc = testQuant(q_model, test_dataloader, quant=True, stats=stats, quant_num_bits=quant_num_bits)
    print('Quantization bit {} filter pruning ratio {} Acc {}'.format(quant_num_bits, amount, test_acc))

    total_zeros = 0
    for name, param in q_model.named_parameters():
        if name in ['conv1.weight', 'conv2.weight']:
            total_zeros += torch.sum(param.data == 0)

    print('Zero params in conv layer {}'.format(total_zeros))

