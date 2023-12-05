"""Encoders utilities."""

__copyright__ = """
LICENSED INTERNAL CODE. PROPERTY OF IBM.
IBM Research Licensed Internal Code
(C) Copyright IBM Corp. 2023
ALL RIGHTS RESERVED
"""
from transformers import (
    AutoImageProcessor,
    BeitImageProcessor,
    BeitModel,
    MobileNetV2Model,
    ResNetModel,
    ViTImageProcessor,
    ViTModel,
)

ENCODERS_REGISTRY = {
    "mobilenetv2_35_96": {
        "processor": AutoImageProcessor.from_pretrained("google/mobilenet_v2_0.35_96"),
        "model": MobileNetV2Model.from_pretrained("google/mobilenet_v2_0.35_96"),
        "size": 96,
    },
    "mobilenetv2_100_224": {
        "processor": AutoImageProcessor.from_pretrained("google/mobilenet_v2_1.0_224"),
        "model": MobileNetV2Model.from_pretrained("google/mobilenet_v2_1.0_224"),
        "size": 224,
    },
    "mobilenetv2_140_224": {
        "processor": AutoImageProcessor.from_pretrained("google/mobilenet_v2_1.4_224"),
        "model": MobileNetV2Model.from_pretrained("google/mobilenet_v2_1.4_224"),
        "size": 224,
    },
    "resnet_18": {
        "processor": AutoImageProcessor.from_pretrained("microsoft/resnet-18"),
        "model": ResNetModel.from_pretrained("microsoft/resnet-18"),
        "size": 224,
    },
    "resnet_50": {
        "processor": AutoImageProcessor.from_pretrained("microsoft/resnet-50"),
        "model": ResNetModel.from_pretrained("microsoft/resnet-50"),
        "size": 224,
    },
    "resnet_101": {
        "processor": AutoImageProcessor.from_pretrained("microsoft/resnet-101"),
        "model": ResNetModel.from_pretrained("microsoft/resnet-101"),
        "size": 224,
    },
    "vit_base_224": {
        "processor": ViTImageProcessor.from_pretrained("google/vit-base-patch16-224"),
        "model": ViTModel.from_pretrained("google/vit-base-patch16-224"),
        "size": 224,
    },
    "vit_base_384": {
        "processor": ViTImageProcessor.from_pretrained("google/vit-base-patch16-384"),
        "model": ViTModel.from_pretrained("google/vit-base-patch16-384"),
        "size": 384,
    },
    "vit_large_224": {
        "processor": ViTImageProcessor.from_pretrained("google/vit-large-patch16-224"),
        "model": ViTModel.from_pretrained("google/vit-large-patch16-224"),
        "size": 224,
    },
    "beit_base_224": {
        "processor": BeitImageProcessor.from_pretrained(
            "microsoft/beit-base-patch16-224-pt22k-ft22k"
        ),
        "model": BeitModel.from_pretrained(
            "microsoft/beit-base-patch16-224-pt22k-ft22k"
        ),
        "size": 224,
    },
    "beit_base_384": {
        "processor": BeitImageProcessor.from_pretrained(
            "microsoft/beit-base-patch16-384"
        ),
        "model": BeitModel.from_pretrained("microsoft/beit-base-patch16-384"),
        "size": 384,
    },
    "beit_large_224": {
        "processor": BeitImageProcessor.from_pretrained(
            "microsoft/beit-large-patch16-224-pt22k-ft22k"
        ),
        "model": BeitModel.from_pretrained(
            "microsoft/beit-large-patch16-224-pt22k-ft22k"
        ),
        "size": 224,
    },
}
