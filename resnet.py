
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import numpy as np
from medmnist import OCTMNIST  # Import the OCTMNIST dataset
import wandb
from sklearn.metrics import confusion_matrix
from PIL import Image




class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3,
                 activation=nn.ReLU(inplace=True), use_batchnorm=True):
        super(ResidualBlock, self).__init__()

        padding = kernel_size // 2  # To keep spatial dimensions constant

        # First convolution layer
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                               padding=padding, bias=not use_batchnorm)
        self.bn1 = nn.BatchNorm2d(out_channels) if use_batchnorm else nn.Identity()

        # Second convolution layer
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size,
                               padding=padding, bias=not use_batchnorm)
        self.bn2 = nn.BatchNorm2d(out_channels) if use_batchnorm else nn.Identity()

        # Define the activation function
        self.activation = activation

        # If input and output channels differ, use a 1x1 conv to match dimensions
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        shortcut = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += shortcut
        out = self.activation(out)
        return out


class CustomResNet(nn.Module):
    def __init__(self,
                 in_channels=3,         # Input channels (e.g., 3 for RGB images)
                 num_blocks=4,          # Number of residual blocks
                 base_channels=64,      # Number of channels for the first block
                 kernel_size=3,         # Kernel size for convolutions
                 activation=nn.ReLU(inplace=True),  # Activation function
                 use_batchnorm=True,    # Whether to use BatchNorm
                 num_classes=10         # Number of classes for final output
                 ):
        super(CustomResNet, self).__init__()

        self.initial_conv = nn.Conv2d(in_channels, base_channels, kernel_size=kernel_size,
                                      padding=kernel_size//2, bias=not use_batchnorm)
        self.initial_bn = nn.BatchNorm2d(base_channels) if use_batchnorm else nn.Identity()
        self.activation = activation

        # Create a sequential container for residual blocks.
        layers = []
        # First block: input channels = base_channels, output channels = base_channels
        for _ in range(num_blocks):
            layers.append(ResidualBlock(base_channels, base_channels,
                                        kernel_size=kernel_size,
                                        activation=activation,
                                        use_batchnorm=use_batchnorm))
        self.residual_layers = nn.Sequential(*layers)

        # Global average pooling and a final linear classifier.
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(base_channels, num_classes)

    def forward(self, x):
        x = self.initial_conv(x)
        x = self.initial_bn(x)
        x = self.activation(x)

        x = self.residual_layers(x)

        x = self.global_avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


