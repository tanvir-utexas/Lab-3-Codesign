from collections import namedtuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, Lambda, Compose
import matplotlib.pyplot as plt
from Network import MyConvNet 
from matplotlib import pyplot as plt

QTensor = namedtuple('QTensor', ['tensor', 'scale', 'zero_point'])

def calcScaleZeroPoint(min_val, max_val,num_bits=8):
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
    q_x = q_x.round().byte()
    
    return QTensor(tensor=q_x, scale=scale, zero_point=zero_point)

def dequantize_tensor(q_x):
    return q_x.scale * (q_x.tensor.float() - q_x.zero_point)
    

#Rework Forward pass of Linear and Conv Layers to support Quantisation
def quantize_layer_mine(layer, num_bits):
    w = quantize_tensor(layer.weight.data, num_bits= num_bits) 
    b = quantize_tensor(layer.bias.data, num_bits= num_bits)
    layer.weight.data = w.tensor.float()
    layer.bias.data = b.tensor.float()


def quantizeLayer(x, layer, stat, scale_x, zp_x, num_bits = 8):
  # for both conv and linear layers

  # cache old values
  W = layer.weight.data
  B = layer.bias.data

  # quantise weights, activations are already quantised
  w = quantize_tensor(layer.weight.data, num_bits= num_bits) 
  b = quantize_tensor(layer.bias.data, num_bits= num_bits)

  layer.weight.data = w.tensor.float()
  layer.bias.data = b.tensor.float()

  # This is Quantisation Artihmetic
  scale_w = w.scale
  zp_w = w.zero_point
  scale_b = b.scale
  zp_b = b.zero_point
  
  scale_next, zero_point_next = calcScaleZeroPoint(min_val=stat['min'], max_val=stat['max'], num_bits= num_bits)

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

def quantForward(model, x, stats, num_bits = 8):
  
  # Quantise before inputting into incoming layers
  x = quantize_tensor(x, min_val=stats['conv1_before']['min'], max_val=stats['conv1_before']['max'], num_bits = num_bits)

  x, scale_next, zero_point_next = quantizeLayer(x.tensor, model.conv1, stats['conv1_after'], x.scale, x.zero_point, num_bits = num_bits)
  # Jeff: Because the input of bn should be FP, we need to convert x to FP
  x = dequantize_tensor(QTensor(tensor=x, scale=scale_next, zero_point=zero_point_next))

  x = model.act1(model.bn1(x))
  x = model.pool1(x)
  
  x = quantize_tensor(x, min_val=stats['conv2_before']['min'], max_val=stats['conv2_before']['max'], num_bits = num_bits)
  x, scale_next, zero_point_next = quantizeLayer(x.tensor, model.conv2, stats['conv2_after'], x.scale, x.zero_point, num_bits = num_bits)
  # Jeff: Because the input of bn should be FP, we need to convert x to FP
  x = dequantize_tensor(QTensor(tensor=x, scale=scale_next, zero_point=zero_point_next))

  x = model.act2(model.bn2((x)))
  x = model.pool2(x)

  x = x.view(x.size(0), -1)

  x = quantize_tensor(x, min_val=stats['lin1_before']['min'], max_val=stats['lin1_before']['max'], num_bits = num_bits)

  x, scale_next, zero_point_next = quantizeLayer(x.tensor, model.lin1, stats['lin2_before'], x.scale, x.zero_point, num_bits = num_bits)
  
  # Back to dequant for final layer
  x = dequantize_tensor(QTensor(tensor=x, scale=scale_next, zero_point=zero_point_next))
   
  x = model.lin2(x)

  return x


def draw_FP_32_histogram(model):
    weights = []
    for name, layer in model.named_modules():
        if isinstance(layer, nn.Conv2d) | isinstance(layer, nn.Linear):
            weights.append(layer.weight.data.reshape(-1))
            weights.append(layer.bias.data.reshape(-1))
    weights = torch.cat(weights).data

    weights = weights.cpu().numpy()

    plt.style.use('fivethirtyeight')
    plt.hist(weights)

    plt.title('Original FP32 Model Weights')
    plt.xlabel('Values')
    plt.ylabel('Number')

    plt.tight_layout()

    plt.savefig('Fp32_weights')

    plt.show()


def draw_quantized_histogram(model, num_bits = 8):
    weights = []
    for name, layer in model.named_modules():
        if isinstance(layer, nn.Conv2d) | isinstance(layer, nn.Linear):
              w = quantize_tensor(layer.weight.data, num_bits= num_bits) 
              b = quantize_tensor(layer.bias.data, num_bits= num_bits)
              weights.append(w.tensor.float().reshape(-1))
              weights.append(b.tensor.float().reshape(-1))
    
    weights = torch.cat(weights)

    weights = weights.cpu().numpy()

    plt.style.use('fivethirtyeight')
    plt.hist(weights)

    plt.title('Quantized Weights-{} bits'.format(num_bits))
    plt.xlabel('Values')
    plt.ylabel('Number')

    plt.tight_layout()

    plt.savefig('Quantized Weights-{} bits'.format(num_bits))

    plt.show()

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


def testQuant(model, test_loader, quant=False, stats=None, num_bits = 8):
    device = 'cuda'
       
    model.eval()
    test_loss = 0
    correct = 0
    loss_fn = nn.CrossEntropyLoss()

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            if quant:
              pred = quantForward(model, data, stats, num_bits = num_bits)
            else:
              pred = NoQuantforward(model, data)

            test_loss += loss_fn(pred, target).item()
            correct += (pred.argmax(1) == target).type(torch.float).sum().item()
    test_loss /= len(test_loader.dataset)

    test_acc = 100. * correct / len(test_loader.dataset)
    # print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    #     test_loss, correct, len(test_loader.dataset),
    #     100. * correct / len(test_loader.dataset)))
    
    return test_loss, test_acc





def quantize_model(model, num_bits):
    # Download test data from open datasets.
    test_data = datasets.FashionMNIST(
        root="data",
        train=False,
        download=True,
        transform=ToTensor(),
    )

    batch_size = 64
    test_dataloader = DataLoader(test_data, batch_size=batch_size)
    
    stats = gatherStats(model, test_dataloader)
    
    print("Evaluating No quantization performance")
    test_loss, test_accuracy = testQuant(model, test_dataloader, quant=False, stats=stats, num_bits = num_bits)    
    print('\nTest set: Average loss: {:.4f}, Accuracy: ({:.2f}%)\n'.format(
        test_loss, test_accuracy)) 
    
    
    print("Evaluating quantization performance")
    test_loss, test_accuracy = testQuant(model, test_dataloader, quant=True, stats=stats, num_bits = num_bits)
    print('\nTest set: Average loss: {:.4f}, Accuracy:({:.2f}%)\n'.format(
        test_loss, test_accuracy))    


def plot_test_accuracy(model, min_bits = 8, max_bits = 16):
    test_data = datasets.FashionMNIST(
        root="data",
        train=False,
        download=True,
        transform=ToTensor(),
    )
    batch_size = 64
    test_dataloader = DataLoader(test_data, batch_size=batch_size)
    
    stats = gatherStats(model, test_dataloader)
    
    test_accuracy = []
    
    for bits in range(min_bits, max_bits+1):
      _, accuracy = testQuant(model, test_dataloader, quant=True, stats=stats, num_bits = bits)
      test_accuracy.append(accuracy) 
    
    plt.style.use('fivethirtyeight')

    plt.plot(range(min_bits, max_bits+1), test_accuracy)
    plt.xlabel('number of bits (num_bits)')
    plt.ylabel('Test Accuracy')
    plt.title('Effect of Quantization on FashionMNIST')

    plt.legend()

    plt.tight_layout()

    plt.savefig('simple_quantization.png')

    plt.show()


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using {} device".format(device))

    model = torch.load('model.pth').to(device)
    #quantize_model(model, num_bits = 16)

    #plot_test_accuracy(model, min_bits = 1, max_bits = 16)
    draw_FP_32_histogram(model)

    draw_quantized_histogram(model, num_bits= 8)
