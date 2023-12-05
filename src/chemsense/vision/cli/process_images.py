"""Image processing and features extraction using pre-trained models from HuggingFace."""

__copyright__ = """
LICENSED INTERNAL CODE. PROPERTY OF IBM.
IBM Research Licensed Internal Code
(C) Copyright IBM Corp. 2023
ALL RIGHTS RESERVED
"""
from pathlib import Path

import click
import numpy as np
import pandas as pd
import torch.utils.data
from torchvision import datasets, transforms

from ..logging_configuration import setup_basic_logging_for_scripts
from ..modeling.encoders import ENCODERS_REGISTRY


@click.command()
@click.option("--task", type=str, default="red_wines", help="Dataset name identifier.")
@click.option(
    "--batch_size",
    type=int,
    default=10,
    help="Batch size for image loading and processing.",
)
@click.option(
    "--data_path",
    required=True,
    type=click.Path(path_type=Path, exists=True),
    help="Path to image directory.",
)
@click.option(
    "--features_path",
    required=True,
    type=click.Path(path_type=Path),
    help="Path to save extracted features.",
)
def main(task: str, batch_size: int, data_path: Path, features_path: Path) -> None:
    setup_basic_logging_for_scripts()
    data_path = Path.joinpath(data_path, task)

    data_transforms = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]
    )

    dataset = datasets.ImageFolder(data_path, transform=data_transforms)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False
    )

    class_names = np.array(dataset.classes)
    files = dataset.imgs
    liquid_class = [file[0].split("/")[-2] for i, file in enumerate(files)]

    for model_name in list(ENCODERS_REGISTRY.keys()):
        processor = ENCODERS_REGISTRY[model_name]["processor"]
        model = ENCODERS_REGISTRY[model_name]["model"]

        features = np.empty((0, 0))
        class_labels = np.empty(0)
        for images, labels in dataloader:
            images = transforms.Resize(
                (
                    ENCODERS_REGISTRY[model_name]["size"],
                    ENCODERS_REGISTRY[model_name]["size"],
                )
            )(images)
            inputs = processor(images=images, return_tensors="pt")
            outputs = model(**inputs)
            feat_batch = outputs.pooler_output.detach().numpy()
            if len(feat_batch.shape) > 2:
                for j in range(len(feat_batch.shape) - 2):
                    feat_batch = np.squeeze(feat_batch, axis=2)
            if np.any(features):
                features = np.concatenate((features, feat_batch))
            else:
                features = feat_batch
            labels_batch = labels.detach().numpy()
            class_labels = np.concatenate((class_labels, class_names[labels_batch]))

        feat_df = pd.DataFrame(features)
        feat_df["CLASS"] = class_labels
        feat_df["SUBCLASS"] = liquid_class
        feat_df.to_csv(
            Path.joinpath(features_path, f"Features_{model_name}_{task}.csv"),
            encoding="utf-8",
            mode="w",
        )
