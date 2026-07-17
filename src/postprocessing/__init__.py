"""Postprocessing and visualization for friction simulation results.

This package provides tools for reading, analyzing, and visualizing the
output from friction simulations:
    - DataReader: Reads and parses simulation output files
    - Plotter: Generates plots and figures from processed data
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .plot_data import Plotter
    from .read_data import DataReader
    from . import pes_scan
    from . import indent


def __getattr__(name: str):
    """Lazily expose heavy symbols to reduce import-time overhead."""
    if name == 'DataReader':
        from .read_data import DataReader as _DataReader
        return _DataReader
    if name == 'Plotter':
        from .plot_data import Plotter as _Plotter
        return _Plotter
    if name == 'pes_scan':
        import importlib
        return importlib.import_module('.pes_scan', __name__)
    if name == 'indent':
        import importlib
        return importlib.import_module('.indent', __name__)
    raise AttributeError(f"module {__name__!s} has no attribute {name!r}")

__all__ = [
    'DataReader',
    'Plotter',
    'pes_scan',
    'indent',
]
