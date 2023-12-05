"""Training and testing models with extracted features."""

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

from chemsense.vision.modeling.classification import (
    attach_classification_head_fewshots,
    attach_classification_head_kfold,
    attach_classification_head_loco,
    attach_classification_head_loco_sugars,
)

from ..logging_configuration import setup_basic_logging_for_scripts
from ..modeling.encoders import ENCODERS_REGISTRY


@click.command()
@click.option("--task", type=str, default="red_wines", help="Dataset name identifier.")
@click.option("--validation", type=str, default="kfold", help="Validation strategy. Supported types are kfold, LOCO, few_shots and Sugar_LOCO.")
@click.option("--number_of_folds", type=int, default=5, help="number of folds to be used in case of kfold validation.")
@click.option("--number_of_components", type=int, default=30, help="Max number of principal components to be used.")
@click.option(
    "--features_path", required=True, type=click.Path(path_type=Path, exists=True), help="Path to directory containing extracted features."
)
@click.option("--output_path", required=True, type=click.Path(path_type=Path), help="Path to save classification model validation results.")
def main(
    task: str,
    validation: str,
    number_of_folds: int,
    number_of_components: int,
    features_path: Path,
    output_path: Path,
) -> None:
    setup_basic_logging_for_scripts()
    encoders = sorted(ENCODERS_REGISTRY.keys())

    for encoder in encoders:
        features_file = Path.joinpath(features_path, f"Features_{encoder}_{task}.csv")
        data_import = pd.read_csv(features_file)
        features_array = np.array(data_import.iloc[:, 1:-2])
        labels = np.array(data_import["CLASS"])
        subclass = np.array(data_import["SUBCLASS"])

        if validation == "kfold":
            attach_classification_head_kfold(
                features_array,
                labels,
                encoder,
                task,
                number_of_folds,
                number_of_components,
                output_path,
            )
        elif validation == "LOCO":
            attach_classification_head_loco(
                features_array,
                labels,
                encoder,
                task,
                subclass,
                number_of_components,
                output_path,
            )
        elif validation == "few_shots":
            attach_classification_head_fewshots(
                features_array, labels, encoder, task, number_of_components, output_path
            )
        elif validation == "Sugar_LOCO":
            attach_classification_head_loco_sugars(
                features_array,
                labels,
                encoder,
                task,
                subclass,
                number_of_components,
                output_path,
            )
