import torch.nn as nn
from torchvision.models import resnet18


class DifferenceRegressor(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = resnet18(weights=None)

    def forward(self, x):
        return self.net(x)