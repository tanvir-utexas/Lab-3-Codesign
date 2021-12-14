# Final project of ML SW-HW codesign

## Usages

- To train the model 
```
# For simplicity, we only train the CIFAR-10 models for 10 epochs
python quant_train.py --Wbits 4 --Abits 8   
```

- To check the tensorboard log 
```
tensorboard --logdir=logs   
```

### Baselines

|   Model   | W-bit | A-bit | Accuracy |
|:---------:|:-----:|:-----:|:--------:|
| ResNet 20 |   32  |   32  |   88.3   |
| ResNet 20 |   4   |   8   |   84.5   |
| ResNet 20 |   2   |   8   |   83.0   |
| ResNet 20 |   2   |   4   |   83.2   |
| ResNet 20 |   1   |   32  |   85.3   |

# Original doc: Pytorch implementation of DoReFa-Net

This repository is the pytorch implementation of [DoReFa-Net](https://arxiv.org/pdf/1606.06160.pdf) for neural network compression. 
The code is inspired by the original [tensorpack implementation](https://github.com/tensorpack/tensorpack/tree/master/examples/DoReFa-Net).
This implementation supports k-bit quantization for both weights and activations.
(I have not test the activation quantization yet, but it should work as expected) 
 
## Requirements:
- python>=3.5
- pytorch>=0.4.1
- tensorboardX


## CIFAR-10:
(Quantized models are trained from scratch.)

Model|W-bit|A-bit|Accuracy
:---:|:---:|:---:|:---:
ResNet-20|32|32|92.13
ResNet-20|4|32|91.46
ResNet-20|2|32|91.05
ResNet-20|1|32|90.54

## ImageNet2012
(Quantized models are finetuned from pretrained model.)

Model|W-bit|A-bit|Top-1 Accuracy|Top-5 Accuracy
:---:|:---:|:---:|:---:|:---:
AlexNet|32|32|56.50%|79.01%
AlexNet|1|32|53.31%|76.72%

