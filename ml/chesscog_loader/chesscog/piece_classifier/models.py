"""Chesscog InceptionV3 piece classifier (12 classes, no empty)."""

from torch import nn
from torchvision import models

from chesscog.core.registry import Registry
from chesscog.core.models import MODELS_REGISTRY

NUM_CLASSES = 12

MODEL_REGISTRY = Registry()
MODELS_REGISTRY.register_as("PIECE_CLASSIFIER")(MODEL_REGISTRY)


@MODEL_REGISTRY.register
class InceptionV3(nn.Module):
    input_size = (299, 299)
    pretrained = True

    def __init__(self):
        super().__init__()
        self.model = models.inception_v3(weights=None, aux_logits=True)
        n = self.model.AuxLogits.fc.in_features
        self.model.AuxLogits.fc = nn.Linear(n, NUM_CLASSES)
        n = self.model.fc.in_features
        self.model.fc = nn.Linear(n, NUM_CLASSES)
        self.params = {
            "head": list(self.model.AuxLogits.fc.parameters()) + list(self.model.fc.parameters())
        }

    def forward(self, x):
        out = self.model(x)
        if self.model.training:
            return out
        return out.logits if hasattr(out, "logits") else out
