# Imports

import torch
import torch.nn as nn

# Local imports

from src.all_moves import get_all_legal_moves
from src.utils.residual_block import ResidualBlock


class ChessResNetModel(nn.Module):
    def __init__(self, num_blocks=[2, 2, 2]):
        super(ChessResNetModel, self).__init__()

        self.in_channels = 64
        self.num_moves = len(get_all_legal_moves())

        # Initial convolutional layer
        self.conv1 = nn.Conv2d(18, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        # Define the ResNet architecture
        self.layer1 = self._make_layer(ResidualBlock, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(ResidualBlock, 256, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(ResidualBlock, 512, num_blocks[2], stride=2)

        # Adaptive pooling layer
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Shared fully connected layer
        self.brain = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(1024, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(2048, 512),
        )

        # Three output heads

        # 1. Policy head
        self.policy_head = nn.Sequential(
            nn.Linear(512, 1024), nn.ReLU(inplace=True), nn.Linear(1024, self.num_moves)
        )

        # 2. Value head
        self.value_head = nn.Sequential(
            nn.Linear(512, 256), nn.ReLU(inplace=True), nn.Linear(256, 1)
        )

        # 3. Game state head
        self.state_head = nn.Sequential(
            nn.Linear(512, 256), nn.ReLU(inplace=True), nn.Linear(256, 3)
        )

    def _make_layer(self, block, out_channels, num_blocks, stride=1):
        # Create a layer of residual blocks with the specified number of blocks and output channels
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []

        # For each stride, create a residual block
        # The first block uses the specified stride, subsequent blocks use stride 1
        for s in strides:
            layers.append(
                block(
                    self.in_channels,
                    out_channels,
                    s,
                )
            )
            self.in_channels = out_channels

        return nn.Sequential(*layers)

    def forward(self, x):

        # Forward pass through the ResNet layers
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)

        # Flatten the tensor
        x = x.view(x.size(0), -1)

        # Pass through the shared fully connected layer
        x = self.brain(x)

        # Pass through each head
        policy = self.policy_head(x)
        value = self.value_head(x)
        state = self.state_head(x)

        return {
            "best_move": policy,
            "game_state": state,
            "value": value,
        }
