import torch
import torch.nn as nn
import torch.nn.functional as F


class SequentialCNN(nn.Module):
    def __init__(self, num_classes):
        super(SequentialCNN, self).__init__()

        # ----------------------
        # Convolutional Hidden Layers
        # ----------------------
        # Conv layer 1: input 3x224x224 -> output 16x224x224 -> pool -> 16x112x112
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)

        # Conv layer 2: input 16x112x112 -> output 32x112x112 -> pool -> 32x56x56
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)

        # Conv layer 3: input 32x56x56 -> output 64x56x56 -> pool -> 64x28x28
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)

        # Pooling layer (used after every conv)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # ----------------------
        # Fully Connected Hidden Layer
        # ----------------------
        # Input: 64*28*28 = 50176
        # Output: 512 neurons
        self.fc1 = nn.Linear(64 * 28 * 28, 512)
        self.dropout = nn.Dropout(0.5)

        # ----------------------
        # Output Layer
        # ----------------------
        # Fully connected layer mapping hidden layer -> num_classes
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        # ----------------------
        # Convolutional Feature Extraction
        # ----------------------
        x = self.pool(F.relu(self.conv1(x)))  # Conv Hidden Layer 1
        x = self.pool(F.relu(self.conv2(x)))  # Conv Hidden Layer 2
        x = self.pool(F.relu(self.conv3(x)))  # Conv Hidden Layer 3

        # Flatten to feed into fully connected layers
        x = x.view(-1, 64 * 28 * 28)

        # ----------------------
        # Fully Connected Hidden Layer
        # ----------------------
        x = F.relu(self.fc1(x))  # FC Hidden Layer
        x = self.dropout(x)

        # ----------------------
        # Output Layer
        # ----------------------
        x = self.fc2(x)  # Output (no ReLU; CrossEntropyLoss expects raw logits)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1,downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        # First layer
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # Second layer
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet18(nn.Module):
        def __init__(self, num_classes):
            super(ResNet18, self).__init__()
            self.in_channels = 64
            self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.bn1 = nn.BatchNorm2d(64)
            self.relu = nn.ReLU(inplace=True)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

            self.layer1 = self._make_layer(64, 2)
            self.layer2 = self._make_layer(128, 2, stride=2)
            self.layer3 = self._make_layer(256, 2, stride=2)
            self.layer4 = self._make_layer(512, 2, stride=2)
            self.avgpool = nn.AdaptiveAvgPool2d((1,1))
            self.fc = nn.Linear(512, num_classes)

            self.apply(self._initialize_weights)
        
        def _make_layer(self, out_channels, blocks, stride=1):
            downsample = None
            if stride != 1 or self.in_channels != out_channels:
                downsample = nn.Sequential(
                    nn.Conv2d(self.in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(out_channels)
                )
            layers = [ResidualBlock(self.in_channels, out_channels, stride, downsample)]
            self.in_channels = out_channels

            for _ in range(1, blocks):
                layers.append(ResidualBlock(out_channels, out_channels))
            return nn.Sequential(*layers)

        def _initialize_weights(self, m):
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        def forward(self, x):
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)

            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            x = self.avgpool(x)

            x = torch.flatten(x, 1)
            x = self.fc(x)

            return x

            