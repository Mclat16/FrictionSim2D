"""Sheet-on-Sheet Simulation Builder.

This module orchestrates the setup of a friction simulation between two
2D material sheets. The model is an N-layer stack (minimum 3):
    - Layer 1: Fixed (bottom)
    - Layers 2 to N-1: Mobile with Langevin thermostat (friction interface)
    - Layer N: Rigid, driven by virtual atom (top)

Constraint modes control interlayer bonding and ghost interactions:
    - atom_bonds: Harmonic bonds between top 2 layers only,
        ghost LJ for non-adjacent layers.
    - com_spring: COM spring between top 2 layers, ghost LJ for non-adjacent
        layers, no constraints on bottom layers.
    - none: No bonds or springs, real LJ for all layer pairs.
"""

import logging
from pathlib import Path
from typing import Dict, Optional, List, Union

from ..core.simulation_base import SimulationBase
from ..core.config import SheetOnSheetSimulationConfig
from ..core.potential_manager import PotentialManager, POTENTIALS_WITH_INTERNAL_LJ
from ..core.utils import atomic2molecular, format_numeric_token, normalize_potential_type
from ..data.models import EV_A_TO_NN, EV_A3_TO_GPA, NM_TO_EV_A2
from . import components

logger = logging.getLogger(__name__)

MIN_LAYERS = 3


class SheetOnSheetSimulation(SimulationBase):
    """Builder for Sheet-on-Sheet friction simulations.

    Creates an N-layer stack of the same 2D material:
        - Layer 1: Fixed bottom layer
        - Layers 2 to N-1: Mobile (Langevin thermostat)
        - Layer N: Driven top layer (rigid body)
    """

    def __init__(self, config: SheetOnSheetSimulationConfig, output_dir: str,
                 config_path: Optional[str] = None):
        super().__init__(config, output_dir, config_path=config_path)
        self.config: SheetOnSheetSimulationConfig = config
        self.structure_paths: Dict[str, Path] = {}
        self.z_positions: Dict[str, float] = {}
        self.groups: Dict[str, str] = {}
        self.pm: Optional[PotentialManager] = None
        self.lat_c: Optional[float] = None
        self.sheet_dims: Optional[Dict] = None

    @property
    def n_layers(self) -> int:
        """Number of layers derived from sheet configuration.

        Sheet-on-sheet currently supports a single explicit layer count
        (not a layer sweep list).
        """
        layers = self.config.sheet.layers
        if len(layers) != 1:
            raise ValueError(
                "Sheet-on-sheet currently requires exactly one value in 2D.layers "
                "(e.g., layers=[3])."
            )
        return int(layers[0])

    @staticmethod
    def _to_list(value: Optional[Union[float, List[float]]]) -> List[float]:
        """Normalize scalar/list sweep values to a list of floats."""
        if value is None:
            return []
        if isinstance(value, list):
            return [float(v) for v in value]
        return [float(value)]


    def build(self) -> None:
        """Constructs the N-layer sheet stack."""
        n_layers = self.n_layers
        if n_layers < MIN_LAYERS:
            raise ValueError(
                f"Sheet-on-sheet requires at least {MIN_LAYERS} layers, "
                f"got {n_layers}"
            )

        logger.info("Starting Sheet-vs-Sheet Build (%d-layer model)...", n_layers)
        self._create_directories()

        build_dir = self.output_dir / "build"
        build_dir.mkdir(parents=True, exist_ok=True)

        self._init_provenance()

        constraint_mode = self.config.settings.simulation.constraint_mode
        pot_type_lower = normalize_potential_type(self.config.sheet.pot_type)

        if pot_type_lower in POTENTIALS_WITH_INTERNAL_LJ and constraint_mode != 'none':
            raise ValueError(
                f"Potential '{pot_type_lower}' has internal interlayer interactions "
                f"and does not support external LJ. "
                f"Set constraint_mode to 'none' in settings.yaml."
            )

        use_pair_bonding = constraint_mode == 'atom_bonds'

        logger.info("Building %d-layer sheet stack...", n_layers)
        stacking_type = getattr(self.config.sheet, 'stack_type', 'AB')
        sheet_path, dims, lat_c = components.build_sheet(
            self.config.sheet, build_dir,
            stack_if_multi=True, settings=self.config.settings,
            n_layers_override=n_layers, use_pair_bonding=use_pair_bonding,
            stacking_type=stacking_type,
            edge_mode=self.config.settings.geometry.finite_sheet_sheetonsheet_edge_mode,
            edge_width=self.config.settings.geometry.finite_sheet_edge_width,
        )
        self.structure_paths['sheet'] = sheet_path

        self.pm = self._generate_potentials()
        assert lat_c is not None
        for i in range(n_layers):
            self.z_positions[f'layer_{i + 1}'] = i * lat_c

        self.lat_c = lat_c
        self.sheet_dims = dims

        self.write_inputs()
        self._generate_hpc_scripts()
        logger.info("Build complete.")

    def _get_hpc_job_name(self) -> str:
        """Get sheet-on-sheet specific job name."""
        return f"sheet_{self.config.sheet.mat}"

    def _collect_simulation_paths(self) -> List[str]:
        """Sheet-on-sheet has single simulation directory."""
        lammps_dir = self.output_dir / 'lammps'
        if lammps_dir.exists():
            return ['.']
        return []

    def _init_provenance(self) -> None:
        """Initialize provenance folder and collect input files."""
        prov_dir = self.output_dir / 'provenance'
        prov_dir.mkdir(parents=True, exist_ok=True)
        logger.debug("Provenance folder initialized at: %s", prov_dir)

        self._add_component_files_to_provenance('sheet', self.config.sheet)

        logger.info("Initialized provenance folder: %s", prov_dir)

    def _generate_potentials(self) -> PotentialManager:
        """Configure potential file for the sheet-on-sheet simulation.

        Interlayer LJ handling depends on constraint_mode and potential type:
            - Potentials with internal LJ (ReaxFF, AIREBO, etc.): No explicit LJ.
            - atom_bonds / com_spring: Ghost LJ for non-adjacent layers.
            - none: Real LJ for all layer pairs.

        Returns:
            Configured PotentialManager instance.
        """
        n_layers = self.n_layers
        constraint_mode = self.config.settings.simulation.constraint_mode

        pm = PotentialManager(
            self.config.settings,
            potentials_dir=self.output_dir / "provenance" / "potentials",
            potentials_prefix=str(
                self.relative_run_dir / "provenance" / "potentials"
            )
        )
        pm.set_lj_overrides(self.config.lj_override)

        sheet_edge_regions = [None, 'edge'] if self.config.settings.geometry.finite_sheet_sheetonsheet_edge_mode != 'none' else None
        pm.register_component('sheet', self.config.sheet, n_layers=n_layers, regions=sheet_edge_regions)

        if self.config.settings.simulation.drive_method == 'virtual_atom':
            pm.register_virtual_atom()

        pm.add_self_interaction('sheet')

        has_internal_lj = not pm.is_sheet_lj(self.config.sheet.pot_type)
        if has_internal_lj:
            pass  # ReaxFF, AIREBO, etc. handle interlayer interactions internally
        elif constraint_mode in ('atom_bonds', 'com_spring'):
            pm.add_ghost_lj('sheet', max_real_distance=1)
        else:
            pm.add_interlayer_interaction('sheet')

        settings_path = self.output_dir / "lammps" / "system.in.settings"
        pm.write_file(settings_path)

        for layer in range(n_layers):
            layer_num = layer + 1
            self.groups[f'layer_{layer_num}'] = (
                pm.types.get_layer_group_string('sheet', layer)
            )

        self.groups['center'] = ' '.join(
            [self.groups[f'layer_{i}'] for i in range(2, n_layers)]
        )
        self.groups['all_types'] = pm.types.get_group_string('sheet')

        return pm

    def _build_layer_groups(self) -> Dict[str, str]:
        """Build a dict mapping 'layer_N_types' to group type strings."""
        layer_groups = {}
        for i in range(1, self.n_layers + 1):
            layer_groups[f'layer_{i}_types'] = self.groups[f'layer_{i}']
        return layer_groups

    def write_inputs(self) -> None:
        """Generate LAMMPS scripts."""
        logger.info("Writing LAMMPS inputs...")

        assert self.pm is not None
        assert self.sheet_dims is not None
        assert self.lat_c is not None

        n_layers = self.n_layers
        total_types = len(self.pm.types)
        constraint_mode = self.config.settings.simulation.constraint_mode
        pot_type = normalize_potential_type(self.config.sheet.pot_type)

        sim = self.config.settings.simulation
        out = self.config.settings.output

        rel_run_dir_str = str(self.relative_run_dir)

        if constraint_mode == 'atom_bonds':
            n_bond_types = 1
        else:
            n_bond_types = 0

        # Auto-detect atom_style based on potential and constraint mode
        if pot_type in ('reaxff', 'reax/c'):
            atom_style = 'charge'
        elif constraint_mode == 'atom_bonds':
            atom_style = 'molecular'
        else:
            atom_style = 'atomic'

        if atom_style == 'molecular':
            atomic2molecular(self.structure_paths['sheet'])

        base_context = {
            'temp': self.config.general.temp,
            'scan_angle_config': self.config.general.scan_angle,
            'scan_angle_force': self.config.general.scan_angle_force,
            'xlo': self.sheet_dims.get('xlo', 0.0),
            'xhi': self.sheet_dims.get('xhi', 100.0),
            'ylo': self.sheet_dims.get('ylo', 0.0),
            'yhi': self.sheet_dims.get('yhi', 100.0),
            'zhi': self.sheet_dims.get('zhi', 15.0),
            'data_file': (
                f"{rel_run_dir_str}/build/"
                f"{self.structure_paths['sheet'].name}"
            ),
            'potential_file': (
                f"{rel_run_dir_str}/lammps/system.in.settings"
            ),
            'num_atom_types': total_types,
            'ngroups': total_types,
            'n_layers': n_layers,
            'constraint_mode': constraint_mode,
            'n_bond_types': n_bond_types,
            **self._build_layer_groups(),
            'center_types': self.groups['center'],
            'lat_c': self.lat_c,
            'sheet_dims': self.sheet_dims,
            'bond_spring_ev': (
                (self.config.general.bond_spring or 80.0) / NM_TO_EV_A2
            ),
            'bond_min': self.lat_c - 0.15,
            'bond_max': self.lat_c + 0.15,
            'driving_spring_ev': (
                (self.config.general.driving_spring or 50.0) / NM_TO_EV_A2
            ),
            'timestep': sim.timestep,
            'thermo': sim.thermo,
            'neighbor_list': sim.neighbor_list,
            'neigh_modify_command': sim.neigh_modify_command,
            'run_steps': sim.slide_run_steps,
            'min_style': sim.min_style,
            'minimization_command': sim.minimization_command,
            'results_freq': out.results_frequency,
            'dump_freq': out.dump_frequency.get('slide', 1000),
            'dump_enabled': out.dump.get('slide', False),
            'results_file_pattern': (
                f"{rel_run_dir_str}/results/"
                f"friction_p${{pressure}}_a${{a}}_s${{speed}}"
            ),
            'dump_file_pattern': (
                f"{rel_run_dir_str}/visuals/"
                f"slide_p${{pressure}}_a${{a}}_s${{speed}}.lammpstrj"
            ),
            'drive_method': sim.drive_method,
            'thermostat_type': self.config.settings.thermostat.type,
            'atom_style': atom_style,
            'pot_type': pot_type,
            'has_internal_lj': not self.pm.is_sheet_lj(pot_type),
            'ev_a_to_nn': EV_A_TO_NN,
            'ev_a3_to_gpa': EV_A3_TO_GPA,
        }

        outer_loop = getattr(self.config.general, 'outer_loop', None)

        if outer_loop not in ('pressure', 'scan_speed'):
            context = dict(base_context)
            context['pressures'] = self.config.general.pressure
            context['scan_speed_config'] = self.config.general.scan_speed
            script = self.render_template("sheetonsheet/slide.lmp", context)
            self.write_file("lammps/slide.in", script)
            logger.info("Using single-script mode (slide.in).")
            logger.info("Inputs written to %s/lammps/", self.output_dir)
            return

        speeds = self._to_list(self.config.general.scan_speed)
        pressures = self._to_list(self.config.general.pressure)

        if outer_loop == 'pressure':
            outer_values = pressures or [0.0]
            inner_speeds: Union[float, List[float]] = (
                speeds if len(speeds) > 1
                else (speeds[0] if speeds else 0.0)
            )
            for pressure in outer_values:
                context = dict(base_context)
                context['pressures'] = pressure
                context['scan_speed_config'] = inner_speeds
                script_name = (
                    f"slide_p{format_numeric_token(float(pressure))}gpa.in"
                )
                script = self.render_template(
                    "sheetonsheet/slide.lmp", context
                )
                self.write_file(f"lammps/{script_name}", script)
        else:
            outer_values = speeds or [0.0]
            inner_pressures: Union[float, List[float]] = (
                pressures if len(pressures) > 1
                else (pressures[0] if pressures else 0.0)
            )
            for speed in outer_values:
                context = dict(base_context)
                context['pressures'] = inner_pressures
                context['scan_speed_config'] = speed
                script_name = (
                    f"slide_{format_numeric_token(float(speed))}ms.in"
                )
                script = self.render_template(
                    "sheetonsheet/slide.lmp", context
                )
                self.write_file(f"lammps/{script_name}", script)

        logger.info("Inputs written to %s/lammps/", self.output_dir)
