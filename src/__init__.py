"""FrictionSim2D - Framework for 2D material friction simulations.

Provides LAMMPS-based molecular dynamics setup for AFM and sheet-on-sheet
friction simulations with automatic potential generation.
"""

from __future__ import annotations

__version__ = "0.2.0"

import logging
from importlib.util import find_spec
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .aiida.submit import run_with_aiida
    from .builders.afm import AFMSimulation
    from .builders.sheetonsheet import SheetOnSheetSimulation
    from .core.config import (
        AFMSimulationConfig,
        SheetOnSheetSimulationConfig,
        load_settings,
        parse_config,
    )
    from .core.potential_manager import PotentialManager
    from .core.run import run_simulations

_AIIDA_AVAILABLE = find_spec('aiida') is not None

logger = logging.getLogger(__name__)


def _default_settings_file() -> str | None:
    """Return a local settings.yaml path when one exists in the current directory."""
    candidate = Path.cwd() / "settings.yaml"
    return str(candidate) if candidate.is_file() else None


def __getattr__(name: str):
    """Lazily expose top-level API symbols to minimize import-time cost."""
    if name == 'run_simulations':
        from .core.run import run_simulations as _run_simulations
        return _run_simulations
    if name == 'AFMSimulation':
        from .builders.afm import AFMSimulation as _AFMSimulation
        return _AFMSimulation
    if name == 'SheetOnSheetSimulation':
        from .builders.sheetonsheet import (
            SheetOnSheetSimulation as _SheetOnSheetSimulation,
        )
        return _SheetOnSheetSimulation
    if name == 'PotentialManager':
        from .core.potential_manager import PotentialManager as _PotentialManager
        return _PotentialManager
    if name == 'AFMSimulationConfig':
        from .core.config import AFMSimulationConfig as _AFMSimulationConfig
        return _AFMSimulationConfig
    if name == 'SheetOnSheetSimulationConfig':
        from .core.config import (
            SheetOnSheetSimulationConfig as _SheetOnSheetSimulationConfig,
        )
        return _SheetOnSheetSimulationConfig
    if name == 'load_settings':
        from .core.config import load_settings as _load_settings
        return _load_settings
    if name == 'parse_config':
        from .core.config import parse_config as _parse_config
        return _parse_config
    if name == 'run_with_aiida':
        if not _AIIDA_AVAILABLE:
            raise AttributeError(
                "run_with_aiida is unavailable because AiiDA is not installed",
            )
        from .aiida.submit import run_with_aiida as _run_with_aiida
        return _run_with_aiida
    raise AttributeError(f"module {__name__!s} has no attribute {name!r}")


def afm(config_file: str = "afm_config.ini", output_root: str = ".",
    generate_hpc: bool = False, settings_file: str | None = None):
    """Run AFM simulations from config file.

    Args:
        config_file: Path to .ini configuration file.
        output_root: Base directory for the simulation root folder.
        settings_file: Path to settings.yaml; falls back to ./settings.yaml if present.
    """
    _run_all(config_file, model="afm", output_root=output_root,
             generate_hpc=generate_hpc, settings_file=settings_file)


def sheetonsheet(config_file: str = "sheet_config.ini", output_root: str = ".",
                 generate_hpc: bool = False, settings_file: str | None = None):
    """Run sheet-on-sheet simulations from config file.

    Args:
        config_file: Path to .ini configuration file.
        output_root: Base directory for the simulation root folder.
        settings_file: Path to settings.yaml; falls back to ./settings.yaml if present.
    """
    _run_all(config_file, model="sheetonsheet", output_root=output_root,
             generate_hpc=generate_hpc, settings_file=settings_file)


def _run_all(config_file: str, model: str = "afm", output_root: str = ".",
             generate_hpc: bool = False, settings_file: str | None = None):
    """Run all simulations defined in config file.

    Args:
        config_file: Path to .ini config file.
        model: Simulation type ('afm' or 'sheetonsheet').
        output_root: Base directory for the simulation root folder.
        settings_file: Path to settings.yaml; falls back to ./settings.yaml if present.
    """
    from .core.run import run_simulations

    created_simulations, simulation_root, configs_to_run, _ = run_simulations(
        config_file=config_file,
        model=model,
        settings_file=settings_file or _default_settings_file(),
        output_root=output_root,
        generate_hpc=generate_hpc,
    )

    print(f"Found {len(configs_to_run)} simulation configurations to run.")
    for idx, sim in enumerate(created_simulations, start=1):
        print(f"  -> Completed [{idx}/{len(created_simulations)}]: {sim}")
    print(f"Simulation root: {simulation_root}")


__all__ = [
    "afm",
    "sheetonsheet",
    "AFMSimulation",
    "SheetOnSheetSimulation",
    "PotentialManager",
    "AFMSimulationConfig",
    "SheetOnSheetSimulationConfig",
    "load_settings",
    "parse_config",
    "run_simulations",
]

if _AIIDA_AVAILABLE:
    __all__.append("run_with_aiida")
