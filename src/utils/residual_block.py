# Imports

import torch
import torch.nn as nn

# Residual Block for ResNet-like architectures


class ResidualBlock(nn.Module):
    """Residual Block for ResNet-like architectures."""

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        # Initialize the layers of the residual block

        # First convolutional layer
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )

        # Batch normalization after the first convolution
        self.bn1 = nn.BatchNorm2d(out_channels)

        # ReLU activation function
        self.relu = nn.ReLU(inplace=True)

        # Second convolutional layer
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )

        # Batch normalization after the second convolution
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Shortcut connection
        self.shortcut = nn.Sequential()

        # If stride is not 1 or input and output channels are different,
        # we need to apply a convolution to match dimensions
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        # Forward pass through the residual block

        # Apply the first convolution, batch normalization, and ReLU activation
        out = self.relu(self.bn1(self.conv1(x)))

        # Apply the second convolution and batch normalization
        out = self.bn2(self.conv2(out))

        # Add the shortcut connection
        shortcut = self.shortcut(x)

        # Combine the output of the second convolution with the shortcut
        out = out + shortcut

        # Apply ReLU activation to the output
        out = self.relu(out)

        return out
