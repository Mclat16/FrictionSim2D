"""Core modules for FrictionSim2D.

This package contains the foundational classes and utilities:
    - Configuration models (Pydantic)
    - Potential management
    - Utility functions for file I/O and calculations
    - Base builder class
"""

from .config import (
    ComponentConfig,
    TipConfig,
    SubstrateConfig,
    SheetConfig,
    GeneralConfig,
    AFMSimulationConfig,
    SheetOnSheetSimulationConfig,
    GlobalSettings,
    load_settings,
    parse_config,
)
from .potential_manager import PotentialManager
from .simulation_base import SimulationBase
from .utils import (
    cifread,
    count_atomtypes,
    lj_params,
    get_material_path,
    get_potential_path,
    read_config,
    get_model_dimensions,
    get_num_atom_types,
    charge2atom,
    atomic2molecular,
    renumber_atom_types,
    get_potential_element_order,
    check_potential_cif_compatibility,
)
from .run import (
    expand_config_sweeps,
    collect_hpc_simulation_paths,
    generate_hpc_scripts_for_root,
    run_simulations,
)

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
