import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.quant_dorefa import *


# class InvertedResidual(nn.Module):
#   def __init__(self, inp, oup, stride, expand_ratio):
#     super(InvertedResidual, self).__init__()
#     assert stride in [1, 2]
#     hidden_dim = round(inp * expand_ratio)
#     self.identity = stride == 1 and inp == oup
#
#     self.conv = nn.Sequential(
#       # pw
#       nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
#       nn.BatchNorm2d(hidden_dim),
#       nn.ReLU6(inplace=True),
#       # dw
#       nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
#       nn.BatchNorm2d(hidden_dim),
#       nn.ReLU6(inplace=True),
#       # pw-linear
#       nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
#       nn.BatchNorm2d(oup),
#     )
#
#   def forward(self, x):
#     if self.identity:
#       return x + self.conv(x)
#     else:
#       return self.conv(x)
#

class InvertedBottleneck_conv_Q(nn.Module):
  '''Pre-activation version of the BasicBlock.'''

  def __init__(self, wbit, abit, in_planes, out_planes, stride=1, percentile=1.0):
    super(InvertedBottleneck_conv_Q, self).__init__()
    Conv2d = conv2d_Q_fn(w_bit=wbit, percentile=percentile)
    self.act_q = activation_quantize_fn(a_bit=abit)
    hidden_dim = in_planes * 6
    # pw
    self.conv1 = Conv2d(in_planes, hidden_dim, 1, 1, 0, bias=False)
    self.bn1 = nn.BatchNorm2d(hidden_dim)
    # dw
    self.conv2 = Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False)
    self.bn2 = nn.BatchNorm2d(hidden_dim)
    # pw-linear
    self.conv3 = Conv2d(hidden_dim, out_planes, 1, 1, 0, bias=False)
    self.bn3 = nn.BatchNorm2d(out_planes)

    self.residual = False
    if stride == 1 and in_planes == out_planes:
      self.residual = True

  def forward(self, x):
    shortcut = x
    # import ipdb
    # ipdb.set_trace()
    x = self.act_q(F.relu(self.bn1(self.conv1(x))))
    x = self.act_q(F.relu(self.bn2(self.conv2(x))))
    x = self.act_q(self.bn3(self.conv3(x)))

    if self.residual:
      x += shortcut
    return x


class MobileNetV2(nn.Module):
  def __init__(self, block, num_units, wbit, abit, num_classes, percentile):
    super(MobileNetV2, self).__init__()
    self.conv0 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)

    self.layers = nn.ModuleList()
    in_planes = 16
    strides = [1] * (num_units[0]) + \
              [2] + [1] * (num_units[1] - 1) + \
              [2] + [1] * (num_units[2] - 1)
    channels = [16] * num_units[0] + [32] * num_units[1] + [64] * num_units[2]
    for stride, channel in zip(strides, channels):
      self.layers.append(block(wbit, abit, in_planes, channel, stride, percentile))
      in_planes = channel

    self.bn = nn.BatchNorm2d(64)
    self.logit = nn.Linear(64, num_classes)

  def forward(self, x):
    out = self.conv0(x)
    for layer in self.layers:
      out = layer(out)
    out = self.bn(out)
    out = out.mean(dim=2).mean(dim=2)
    out = self.logit(out)
    return out


def mobilenetv2(wbits, abits, percentile, num_classes=10):
  return MobileNetV2(InvertedBottleneck_conv_Q, [2, 2, 2], wbits, abits, num_classes=num_classes, percentile=percentile)


