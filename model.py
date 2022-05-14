from turtle import forward
import torch.nn as nn
from torch.nn.functional import relu

class DoubleConv(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, residual=False, kernel1=3, kernel2=3):
        super(DoubleConv, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=kernel1, padding=kernel1 // 2)
        self.batch_norm1 = nn.BatchNorm2d(mid_channels)
        self.conv2 = nn.Conv2d(mid_channels, out_channels, kernel_size=kernel2, padding=kernel2 // 2)
        self.batch_norm2 = nn.BatchNorm2d(out_channels)
        self.pool = nn.MaxPool2d()
        self.residual = residual

        if residual:
            self.res_conv = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x):
        
        y = self.conv1(x)
        y = self.batch_norm1(x)
        y = self.conv2(x)
        if self.residual:
            x = self.res_conv(x)
            y = y + x
        y = self.pool(y)
        y = self.batch_norm2(y)
        y = relu(y)
    
        return y


class Model(nn.Module):
  def __init__(self, image_size, n_classes):
    super().__init__()
    self.d_conv1 = DoubleConv(3, 32, 64)
    self.d_conv2 = DoubleConv(64, 128, 128, residual=True)
    self.d_conv3 = DoubleConv(128, 256, 256, residual=True)
    self.fc1 = nn.Linear(image_size // 8 * 256, 128)

    self.fc2 = nn.Linear(128, n_classes)

  def forward(self, x):
    x = self.d_conv1(x)
    x = self.d_conv2(x)
    x = self.d_conv3(x)

    x = self.fc1(x)
    x = relu(x)
    x = self.fc2(x)

    return x
