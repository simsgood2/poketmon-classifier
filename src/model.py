import torch.nn as nn
from torchvision import models


def get_weights(backbone: str, pretrained: bool):
    if not pretrained:
        return None

    weights_map = {
        "resnet18": models.ResNet18_Weights.DEFAULT,
        "resnet34": models.ResNet34_Weights.DEFAULT,
        "mobilenet_v3_small": models.MobileNet_V3_Small_Weights.DEFAULT,
        "efficientnet_b0": models.EfficientNet_B0_Weights.DEFAULT,
    }
    return weights_map[backbone]


def build_model(backbone: str, num_classes: int, pretrained: bool = True):
    weights = get_weights(backbone, pretrained)

    if backbone == "resnet18":
        model = models.resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    if backbone == "resnet34":
        model = models.resnet34(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    if backbone == "mobilenet_v3_small":
        model = models.mobilenet_v3_small(weights=weights)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
        return model

    if backbone == "efficientnet_b0":
        model = models.efficientnet_b0(weights=weights)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
        return model

    raise ValueError(f"Unsupported backbone: {backbone}")


def configure_trainable_layers(model: nn.Module, backbone: str, freeze_backbone: bool, unfreeze_last_block: bool):
    if not freeze_backbone:
        return model

    for parameter in model.parameters():
        parameter.requires_grad = False

    if backbone.startswith("resnet"):
        for parameter in model.fc.parameters():
            parameter.requires_grad = True
        if unfreeze_last_block:
            for parameter in model.layer4.parameters():
                parameter.requires_grad = True
        return model

    if backbone == "mobilenet_v3_small":
        for parameter in model.classifier.parameters():
            parameter.requires_grad = True
        if unfreeze_last_block:
            for parameter in model.features[-1].parameters():
                parameter.requires_grad = True
        return model

    if backbone == "efficientnet_b0":
        for parameter in model.classifier.parameters():
            parameter.requires_grad = True
        if unfreeze_last_block:
            for parameter in model.features[-1].parameters():
                parameter.requires_grad = True
        return model

    return model

