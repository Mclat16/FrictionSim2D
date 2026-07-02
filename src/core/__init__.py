"""Core modules for FrictionSim2D.

Exports are resolved lazily to keep import-time side effects low and avoid
builder/run/hpc circular imports (importing e.g. ``core.config`` must not pull
in ``simulation_base`` -> ``hpc`` at import time).
This package contains the foundational classes and utilities:
    - Configuration models (Pydantic)
    - Potential management
    - Utility functions for file I/O and calculations
    - Base builder class
"""

from importlib import import_module
from typing import Any

__all__ = [
    "ComponentConfig",
    "TipConfig",
    "SubstrateConfig",
    "SheetConfig",
    "GeneralConfig",
    "AFMSimulationConfig",
    "SheetOnSheetSimulationConfig",
    "GlobalSettings",
    "load_settings",
    "parse_config",
    "PotentialManager",
    "SimulationBase",
    "cifread",
    "count_atomtypes",
    "lj_params",
    "get_material_path",
    "get_potential_path",
    "read_config",
    "get_model_dimensions",
    "get_num_atom_types",
    "charge2atom",
    "atomic2molecular",
    "renumber_atom_types",
    "get_potential_element_order",
    "check_potential_cif_compatibility",
    "expand_config_sweeps",
    "collect_hpc_simulation_paths",
    "generate_hpc_scripts_for_root",
    "run_simulations",
]


_EXPORT_MAP = {
    "ComponentConfig": (".config", "ComponentConfig"),
    "TipConfig": (".config", "TipConfig"),
    "SubstrateConfig": (".config", "SubstrateConfig"),
    "SheetConfig": (".config", "SheetConfig"),
    "GeneralConfig": (".config", "GeneralConfig"),
    "AFMSimulationConfig": (".config", "AFMSimulationConfig"),
    "SheetOnSheetSimulationConfig": (".config", "SheetOnSheetSimulationConfig"),
    "GlobalSettings": (".config", "GlobalSettings"),
    "load_settings": (".config", "load_settings"),
    "parse_config": (".config", "parse_config"),
    "PotentialManager": (".potential_manager", "PotentialManager"),
    "SimulationBase": (".simulation_base", "SimulationBase"),
    "cifread": (".utils", "cifread"),
    "count_atomtypes": (".utils", "count_atomtypes"),
    "lj_params": (".utils", "lj_params"),
    "get_material_path": (".utils", "get_material_path"),
    "get_potential_path": (".utils", "get_potential_path"),
    "read_config": (".utils", "read_config"),
    "get_model_dimensions": (".utils", "get_model_dimensions"),
    "get_num_atom_types": (".utils", "get_num_atom_types"),
    "charge2atom": (".utils", "charge2atom"),
    "atomic2molecular": (".utils", "atomic2molecular"),
    "renumber_atom_types": (".utils", "renumber_atom_types"),
    "get_potential_element_order": (".utils", "get_potential_element_order"),
    "check_potential_cif_compatibility": (".utils", "check_potential_cif_compatibility"),
    "expand_config_sweeps": (".run", "expand_config_sweeps"),
    "collect_hpc_simulation_paths": (".run", "collect_hpc_simulation_paths"),
    "generate_hpc_scripts_for_root": (".run", "generate_hpc_scripts_for_root"),
    "run_simulations": (".run", "run_simulations"),
}


def __getattr__(name: str) -> Any:
    if name not in _EXPORT_MAP:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _EXPORT_MAP[name]
    module = import_module(module_name, __name__)
    return getattr(module, attr_name)
