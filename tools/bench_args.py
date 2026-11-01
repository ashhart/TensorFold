"""Shared command-line validation for the Python benchmark clients."""
import math


def validate_temperatures(parser, value: str) -> None:
    """Check the entire selected list before requests while preserving existing workload parsing."""
    try:
        for item in value.split(","):
            if not math.isfinite(float(item)):
                raise ValueError
    except ValueError:
        parser.error("--temperatures must be comma-separated finite numbers")
