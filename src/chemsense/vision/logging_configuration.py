"""Logging utilities."""

__copyright__ = """
LICENSED INTERNAL CODE. PROPERTY OF IBM.
IBM Research Licensed Internal Code
(C) Copyright IBM Corp. 2023
ALL RIGHTS RESERVED
"""
import logging
import sys


def setup_basic_logging_for_scripts() -> None:
    """Setup basic stdout logging for scripts."""
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
