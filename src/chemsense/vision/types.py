"""Logging utilities."""

__copyright__ = """
LICENSED INTERNAL CODE. PROPERTY OF IBM.
IBM Research Licensed Internal Code
(C) Copyright IBM Corp. 2023
ALL RIGHTS RESERVED
"""
from typing import TypedDict

import pandas as pd


class ExperimentaMeasurement(TypedDict):
    data: pd.DataFrame
    category: str
    rep: str
