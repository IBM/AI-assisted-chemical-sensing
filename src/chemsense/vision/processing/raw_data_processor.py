"""Raw data loading and processing module."""

__copyright__ = """
LICENSED INTERNAL CODE. PROPERTY OF IBM.
IBM Research Licensed Internal Code
(C) Copyright IBM Corp. 2023
ALL RIGHTS RESERVED
"""
import logging
from pathlib import Path
from typing import Dict, List, Union

import numpy as np
import pandas as pd

from ..types import ExperimentaMeasurement

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class RawDataProcessor:
    def __init__(
        self,
        number_of_electrodes: int = 16,
        reference_duration: int = 20,
        test_duration: int = 60,
    ) -> None:
        """Initialize the data processor.

        Args:
            number_of_electrodes: number of electrodes in the sensor. Defaults to 16.
            reference_duration: duration of the reference liquid measurement. Defaults to 20.
            test_duration: duration of the test measurement. Defaults to 60.
        """
        self.number_of_electrodes = number_of_electrodes
        self.reference_duration = reference_duration
        self.test_duration = test_duration

    @staticmethod
    def load(data_path: Path) -> Dict[str, ExperimentaMeasurement]:
        """Load the raw data.

        Args:
            path: path to the data.

        Returns:
            loaded measurements.
        """
        # `a_1` is a key in the
        # `experimental_measurements`
        # dictionary. It represents a
        # specific measurement with the
        # category "a" and repetition "1".

        experimental_measurements: Dict[str, ExperimentaMeasurement] = {}
        for path in data_path.glob("*.txt"):
            try:
                logger.info(f"loading data from path={path}...")
                name = path.name.split(".")[0]
                class_components, rep = name.split("_")
                experimental_measurements[name] = {
                    "data": pd.read_csv(path, skiprows=0),
                    "category": "_".join(class_components),
                    "rep": rep,
                }
                logger.info(f"data loaded from path={path}.")
            except Exception:
                logger.exception(f"error loading data from path={path}!")
        return experimental_measurements

    def unfold_measurements(
        self, experimental_measurements: Dict[str, ExperimentaMeasurement]
    ) -> pd.DataFrame:
        """Unfold experimental measurements in a data frame.

        Args:
            experimental_measurements: experimental measurements.

        Returns:
            unfolded measurements in a data frame.
        """
        experiments_trace: Dict[str, List[Union[float, str]]] = {}
        for name in experimental_measurements:
            experiment_data_and_info: List[Union[float, str]] = []
            for electrode in range(1, self.number_of_electrodes):
                reference_voltage = np.mean(
                    (experimental_measurements[name]["data"]).iloc[
                        0 : self.reference_duration, electrode
                    ]
                )
                relative_signal = (
                    experimental_measurements[name]["data"].iloc[
                        self.reference_duration : (
                            self.reference_duration + self.test_duration
                        ),
                        electrode,
                    ]
                    - reference_voltage
                )
                relative_signal = list(relative_signal)
                experiment_data_and_info.extend(relative_signal)
            experiment_data_and_info.append(experimental_measurements[name]["category"])
            experiment_data_and_info.append(experimental_measurements[name]["rep"])
            experiments_trace[name] = experiment_data_and_info
        return pd.DataFrame.from_dict(experiments_trace, orient="index")

    def load_and_save_unfolded_measurements(
        self, data_path: Path, output_path: Path
    ) -> None:
        """Load and save unfolded measurements.

        Args:
            data_path: path to the data.
            output_path: path to file for saving the unfolded signals.
        """
        experimental_measurements = RawDataProcessor.load(data_path=data_path)
        unfolded_measurements_df = self.unfold_measurements(
            experimental_measurements=experimental_measurements
        )
        unfolded_measurements_df.to_csv(output_path, encoding="utf-8", mode="w")
