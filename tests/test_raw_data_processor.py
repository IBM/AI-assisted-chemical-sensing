"""Tests for raw data processor."""

__copyright__ = """
LICENSED INTERNAL CODE. PROPERTY OF IBM.
IBM Research Licensed Internal Code
(C) Copyright IBM Corp. 2023
ALL RIGHTS RESERVED
"""
from typing import Dict, Generator

import importlib_resources
import pytest

from chemsense.vision.processing.raw_data_processor import RawDataProcessor
from chemsense.vision.types import ExperimentaMeasurement


@pytest.fixture
def raw_data_processor() -> Generator[RawDataProcessor, None, None]:
    yield RawDataProcessor()


@pytest.fixture
def experimental_measurements() -> (
    Generator[Dict[str, ExperimentaMeasurement], None, None]
):
    yield RawDataProcessor.load(
        importlib_resources.files("chemsense") / "vision/resources/tests/raw_data/"
    )


def test_load(raw_data_processor: RawDataProcessor) -> None:
    experimental_measurements = raw_data_processor.load(
        importlib_resources.files("chemsense") / "vision/resources/tests/raw_data/"
    )
    assert len(experimental_measurements) == 2
    assert "a_1" in experimental_measurements
    assert "b_2" in experimental_measurements
    assert experimental_measurements["a_1"]["category"] == "a"
    assert experimental_measurements["a_1"]["rep"] == "1"
    assert experimental_measurements["b_2"]["category"] == "b"
    assert experimental_measurements["b_2"]["rep"] == "2"


def test_unfold_measurements(
    raw_data_processor: RawDataProcessor,
    experimental_measurements: Dict[str, ExperimentaMeasurement],
) -> None:
    unfolded_measurements_df = raw_data_processor.unfold_measurements(
        experimental_measurements=experimental_measurements
    )
    columns = unfolded_measurements_df.columns
    assert isinstance(unfolded_measurements_df[columns[-1]].iloc[0], str)
    assert isinstance(unfolded_measurements_df[columns[-2]].iloc[0], str)
    assert isinstance(unfolded_measurements_df[columns[-3]].iloc[0], float)
    assert set(unfolded_measurements_df[columns[-1]].tolist()) == {"1", "2"}
    assert set(unfolded_measurements_df[columns[-2]].tolist()) == {"a", "b"}
