"""Image encoding module."""

__copyright__ = """
LICENSED INTERNAL CODE. PROPERTY OF IBM.
IBM Research Licensed Internal Code
(C) Copyright IBM Corp. 2023
ALL RIGHTS RESERVED
"""
import csv
from pathlib import Path

import click
import matplotlib.pyplot as plt
import numpy as np
from pyts.image import GramianAngularField

from chemsense.vision.processing.raw_data_processor import RawDataProcessor

from ..logging_configuration import setup_basic_logging_for_scripts


@click.command()
@click.option("--task", type=str, default="red_wines", help= "Task name to be used as identifier.")
@click.option(
    "--data_path", required=True, type=click.Path(path_type=Path, exists=True), help="Path to directory containing sensor raw data.")
@click.option("--export_spectra_path", required=True, type=click.Path(path_type=Path), help="Path to save processed sensor data.")
@click.option("--export_images_path", required=True, type=click.Path(path_type=Path),help="Path to save generated images.")
def main(
    task: str, data_path: Path, export_spectra_path: Path, export_images_path: Path
) -> None:
    setup_basic_logging_for_scripts()
    export_spectra_path = Path.joinpath(export_spectra_path, f"traces_{task}.csv")
    export_images_path = Path.joinpath(export_spectra_path, f"{task}")
    export_images_path.mkdir(exist_ok=True)

    raw_data_processor = RawDataProcessor()
    raw_data_processor.load_and_save_unfolded_measurements(
        data_path, export_spectra_path
    )

    with open(export_spectra_path) as fp:
        csv_reader = csv.reader(fp)
        _ = next(csv_reader)
        signal_matrix = []
        classes = []
        files = []
        for signal in csv_reader:
            signal_matrix.append(list(signal[1:-2]))
            classes.append(signal[-2])
            files.append(signal[-1])

    traces = np.array(signal_matrix).astype(float)

    for i in range(traces.shape[0]):
        fig = plt.figure(constrained_layout=True)
        transformer = GramianAngularField()
        X_new = transformer.transform(traces[i, :].reshape(1, -1))
        plt.imshow(X_new[0], cmap="rainbow")

        file_dir = Path.joinpath(export_images_path, "_".join(files[i].split("_")[:-1]))
        file_dir.mkdir(exist_ok=True)
        filename = f"{files[i]}.png"
        fig.savefig(Path.joinpath(file_dir, filename), format="png")
