from __future__ import annotations
from torch import nn
from torchvision.models import resnet18

class ResNet18CIFAR(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        m = resnet18(weights=None)
        m.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        m.maxpool = nn.Identity()
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        self.net = m
    def forward(self, x):
        return self.net(x)
