"""Train and test models with few shots and image augmentation."""

__copyright__ = """
LICENSED INTERNAL CODE. PROPERTY OF IBM.
IBM Research Licensed Internal Code
(C) Copyright IBM Corp. 2023
ALL RIGHTS RESERVED
"""
import os
import random
from copy import deepcopy
from pathlib import Path
from typing import Dict, List

import click
import numpy as np
import pandas as pd
import torch.utils.data
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from torchvision import datasets, transforms

from ..logging_configuration import setup_basic_logging_for_scripts
from ..modeling.classification import CLASSIFICATION_HEADS
from ..modeling.encoders import ENCODERS_REGISTRY

num_images = int(os.getenv("NUMBER_OF_IMAGES", 50))
num_rep = int(os.getenv("NUMBER_OF_REPEATS", 50))


@click.command()
@click.option("--task", type=str, default="red_wines", help="Dataset name identifier.")
@click.option(
    "--n_comp",
    type=int,
    default=10,
    help="Number of principal components to be used as predictors.",
)
@click.option(
    "--mix_ratio",
    type=float,
    default=0.95,
    help="Fraction of pixel intensity for image mixing and data augmentation. Needs to be between 0 and 1.",
)
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
    "--output_path",
    required=True,
    type=click.Path(path_type=Path),
    help="Path to save classification model validation results.",
)
def main(
    task: str,
    n_comp: int,
    mix_ratio: float,
    batch_size: int,
    data_path: Path,
    output_path: Path,
) -> None:
    setup_basic_logging_for_scripts()

    w_class = mix_ratio
    w_other = 1 - w_class

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

    Path(output_path).mkdir(exist_ok=True)
    result_path = Path.joinpath(output_path, task)

    model_heads = CLASSIFICATION_HEADS.keys()
    classification_heads = deepcopy(CLASSIFICATION_HEADS)

    for model_head in model_heads:
        Path.joinpath(result_path, model_head).mkdir(exist_ok=True)

    complete_results = {}
    for model_name in list(ENCODERS_REGISTRY.keys()):
        processor = ENCODERS_REGISTRY[model_name]["processor"]
        model = ENCODERS_REGISTRY[model_name]["model"]

        class_labels = np.empty(0)
        image_collection = torch.empty(
            (
                0,
                3,
                ENCODERS_REGISTRY[model_name]["size"],
                ENCODERS_REGISTRY[model_name]["size"],
            )
        )
        for images, labels in dataloader:
            images = transforms.Resize(
                (
                    ENCODERS_REGISTRY[model_name]["size"],
                    ENCODERS_REGISTRY[model_name]["size"],
                )
            )(images)
            image_collection = torch.cat((image_collection, images), 0)
            labels_batch = labels.detach().numpy()
            class_labels = np.concatenate((class_labels, class_names[labels_batch]))

        for shots in [3, 5, 7]:
            acc_val: Dict[str, List[float]] = {}
            for model_head in model_heads:
                acc_val[model_head] = []

            for rep in range(num_rep):
                train_idx = []
                for group in class_names:
                    train_idx.append(
                        random.sample(
                            [
                                idx
                                for idx, value in enumerate(class_labels)
                                if value == group
                            ],
                            k=shots,
                        )
                    )

                t_idx = [item for sublist in train_idx for item in sublist]
                v_idx = [item for item in range(len(class_labels)) if item not in t_idx]

                images_train = torch.empty(
                    (
                        0,
                        3,
                        ENCODERS_REGISTRY[model_name]["size"],
                        ENCODERS_REGISTRY[model_name]["size"],
                    )
                )
                class_labels_train = np.empty(0)
                for class_group in train_idx:
                    images_group = image_collection[class_group]
                    images_train = torch.cat((images_train, images_group), 0)
                    for k in range(num_images - shots):
                        rand_shot = random.randint(0, shots - 1)
                        aug_image = torch.mul(images_group[rand_shot], w_class)
                        idx_choice = t_idx.copy()
                        idx_choice.remove(class_group[rand_shot])
                        aug_image = torch.add(
                            aug_image,
                            torch.mul(
                                image_collection[random.choice(idx_choice)], w_other
                            ),
                        )
                        images_train = torch.cat(
                            (images_train, aug_image.expand(1, -1, -1, -1)), 0
                        )
                    class_labels_train = np.concatenate(
                        (
                            class_labels_train,
                            np.repeat(
                                (np.array(class_labels)[class_group])[0],
                                num_images,
                            ),
                        )
                    )

                images_val = image_collection[v_idx]
                class_labels_val = class_labels[v_idx]

                inputs_train = processor(images=images_train, return_tensors="pt")
                outputs_train = model(**inputs_train)
                feat_train = outputs_train.pooler_output.detach().numpy()

                inputs_val = processor(images=images_val, return_tensors="pt")
                outputs_val = model(**inputs_val)
                feat_val = outputs_val.pooler_output.detach().numpy()

                feat_comp = np.vstack((feat_train, feat_val))
                if len(feat_comp.shape) > 2:
                    for j in range(len(feat_comp.shape) - 2):
                        feat_comp = np.squeeze(feat_comp, axis=2)

                pca = PCA(n_components=n_comp)
                p_comp = pca.fit_transform(feat_comp)
                explanation = np.sum(pca.explained_variance_ratio_)

                x_train = p_comp[: feat_train.shape[0], :]
                y_train = class_labels_train

                x_val = p_comp[feat_train.shape[0] :, :]
                y_val = class_labels_val

                for model_head in model_heads:
                    classification_heads[model_head].fit(x_train, y_train)

                    y_predict = classification_heads[model_head].predict(x_val).tolist()
                    y_true = y_val.tolist()

                    acc_val[model_head].append(accuracy_score(y_val, y_predict))

                    pred_df = pd.DataFrame(y_true, columns=["True"])
                    pred_df["Predicted"] = y_predict
                    pred_df.to_csv(
                        Path.joinpath(
                            result_path,
                            model_head,
                            f"Predictions_{model_head}_{task}_{str(n_comp)}comp_{str(shots)}shots_MixedAug_{str(int(w_class * 100))}{str(int(w_other * 100))}.csv",
                        ),
                        encoding="utf-8",
                        mode="a",
                    )

            for model_head in model_heads:
                results = {
                    "NetExtractor": model_name,
                    "ModelHead": model_head,
                    "Number_Shots": shots,
                    "Accuracy": str(np.average(acc_val[model_head])),
                    "StdAccuracy": str(np.std(acc_val[model_head])),
                    "PC_components": str(n_comp),
                    "PC_explanation": str(explanation),
                }

                complete_results[
                    f"{model_name}_{model_head}_{str(n_comp)}_{str(shots)}"
                ] = results

    summary_results = pd.DataFrame.from_dict(complete_results, orient="index")
    summary_results.to_csv(
        Path.joinpath(
            result_path,
            f"Accuracy_{task}_few_shots_MixedAug_{str(int(w_class * 100))}{str(int(w_other * 100))}.csv",
        ),
        encoding="utf-8",
        mode="a",
    )


if __name__ == "__main__":
    main()
