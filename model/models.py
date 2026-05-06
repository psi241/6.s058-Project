import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.datasets import CIFAR10
import torchvision.models as models
from torchvision import transforms

class SimCLRModel(nn.Module):
    def __init__(self, projection_dim=128):
        super().__init__()
        base_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        num_ftrs = base_model.fc.in_features # use only fully connected layer (no final classification head)
        base_model.fc = nn.Identity()
        self.encoder = base_model
        self.projection_dim = projection_dim
        self.projection_head = nn.Sequential( # project to embedding space
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.Linear(512, projection_dim)
        )

    def forward(self, x):
        h = self.encoder(x)
        z = self.projection_head(h)
        return z