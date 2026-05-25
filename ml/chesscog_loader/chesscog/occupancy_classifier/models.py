"""Chesscog ResNet occupancy classifier (empty vs occupied)."""

from torch import nn
from torchvision import models

from chesscog.core.registry import Registry
from chesscog.core.models import MODELS_REGISTRY

NUM_CLASSES = 2

MODEL_REGISTRY = Registry()
MODELS_REGISTRY.register_as("OCCUPANCY_CLASSIFIER")(MODEL_REGISTRY)


@MODEL_REGISTRY.register
class ResNet(nn.Module):
    input_size = (100, 100)
    pretrained = True

    def __init__(self):
        super().__init__()
        self.model = models.resnet18(weights=None)
        n = self.model.fc.in_features
        self.model.fc = nn.Linear(n, NUM_CLASSES)
        self.params = {"head": list(self.model.fc.parameters())}

    def forward(self, x):
        return self.model(x)
