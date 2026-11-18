"""Shared preprocessing and architecture: training and serving must agree."""
import torch
from torchvision import models, transforms

LABELS = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Pleural Effusion']


def transform(training=False):
    steps = [transforms.Resize((224, 224))]
    if training:
        steps.append(transforms.RandomRotation(5))
    return transforms.Compose(steps + [transforms.ToTensor(), transforms.Normalize([.485, .456, .406], [.229, .224, .225])])


def build(pretrained=False):
    model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT if pretrained else None)
    model.classifier = torch.nn.Linear(model.classifier.in_features, len(LABELS))
    return model
