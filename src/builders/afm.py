"""AFM Simulation Builder.

This module orchestrates the setup of a complete Atomic Force Microscopy (AFM)
simulation. It coordinates the construction of the Tip, Substrate, and Sheet,
generates the necessary potentials, and writes the LAMMPS input scripts.
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

from ..core.simulation_base import SimulationBase
from ..core.config import AFMSimulationConfig
from ..core.potential_manager import PotentialManager
from ..core.utils import normalize_potential_type
from ..data.models import EV_A_TO_NN
from . import components

logger = logging.getLogger(__name__)


class AFMSimulation(SimulationBase):
    """Builder for AFM simulations (Tip + Sheet + Substrate).
    
    Handles layer sweeps internally - when config.sheet.layers is a list,
    builds common components once and iterates over layer counts.
    """

    def __init__(self, config: AFMSimulationConfig, output_dir: str,
                    config_path: Optional[str] = None):
        super().__init__(config, output_dir, config_path=config_path)
        self.config: AFMSimulationConfig = config
        self.sheet_paths: Dict[int, Path] = {}
        self.tip_path: Path
        self.sub_path: Path
        self.z_positions: Dict[int, Dict[str, float]] = {}
        self.groups: Dict[int, Dict[str, str]] = {}
        self.pm: Dict[int, PotentialManager] = {}
        self.lat_c: Optional[float] = None
        self.sheet_dims: Dict[str, float] = {}
        self.output_dir_layer: Dict[int, Path] = {}
        self.relative_run_dir_layer: Dict[int, Path] = {}

    def build(self) -> None:
        """Constructs the atomic systems and layout.
        
        If config.sheet.layers is a list, iterates over layer counts.
        """
        logger.info("Starting AFM Simulation Build...")

        build_dir = self.output_dir / "build"
        build_dir.mkdir(parents=True, exist_ok=True)

        self._init_provenance()

        for n_layers in self.config.sheet.layers:
            logger.info("--- Building for %s layer(s) ---", n_layers)

            self.output_dir_layer[n_layers] = self.output_dir / f"L{n_layers}"
            self.relative_run_dir_layer[n_layers] = self.relative_run_dir / f"L{n_layers}"
            self._create_directories(self.output_dir_layer[n_layers])

            stacking_type = getattr(self.config.sheet, 'stack_type', 'AB')
            sheet_path, sheet_dims, lat_c = components.build_sheet(
                self.config.sheet, build_dir,
                stack_if_multi=True, settings=self.config.settings,
                    n_layers_override=n_layers,
                    stacking_type=stacking_type
            )
            self.sheet_paths[n_layers] = sheet_path
            if self.lat_c is None:
                self.lat_c = lat_c
                self.sheet_dims = sheet_dims

        self.tip_path, tip_radius, self.sub_path = self._build_components(build_dir)

        for n_layers in self.config.sheet.layers:
            self.pm[n_layers] = self._generate_potentials(n_layers)
            self._calculate_z_positions(n_layers, tip_radius)
            self.write_inputs(n_layers)

        self._generate_hpc_scripts()
        logger.info("Build complete for all layer configurations.")

    def _get_hpc_job_name(self) -> str:
        """Get AFM-specific job name."""
        return f"afm_{self.config.sheet.mat}"

    def _init_provenance(self) -> None:
        """Initialize provenance folder and collect input files."""
        prov_dir = self.output_dir / 'provenance'
        prov_dir.mkdir(parents=True, exist_ok=True)
        logger.debug("Provenance folder initialized at: %s", prov_dir)

        for component_name, config in [
            ('sheet', self.config.sheet),
            ('tip', self.config.tip),
            ('sub', self.config.sub)
        ]:
            self._add_component_files_to_provenance(component_name, config)

        logger.info("Initialized provenance folder: %s", prov_dir)

    def _build_components(self, build_dir: Path) -> Tuple[Path, float, Path]:
        """Builds tip and substrate (shared across layer configurations).
        
        Sheet is built separately for each layer count.
        
        Returns:
            Tuple of (tip_path, tip_radius, sub_path).
        """
        tip_path, tip_radius = components.build_tip(
            self.config.tip, build_dir, self.config.settings
        )
        logger.info("Built tip: %s", tip_path.name)

        sub_path = components.build_substrate(
            self.config.sub, build_dir, self.sheet_dims,
            settings=self.config.settings
        )
        logger.info("Built substrate: %s", sub_path.name)

        if self.config.settings.thermostat.type == 'langevin':
            tip_height = tip_radius / self.config.settings.geometry.tip_reduction_factor

            components.apply_langevin_regions(
                component_name='tip',
                component_path=tip_path,
                config=self.config.tip,
                settings=self.config.settings,
                component_height=tip_height
            )
            logger.info("Applied Langevin regions to tip")

            components.apply_langevin_regions(
                component_name='sub',
                component_path=sub_path,
                config=self.config.sub,
                settings=self.config.settings,
                component_height=self.config.sub.thickness,
                substrate_thickness=self.config.sub.thickness
            )
            logger.info("Applied Langevin regions to substrate")

        return tip_path, tip_radius, sub_path

    def _calculate_z_positions(self, n_layers: int, tip_radius: float) -> None:
        """Calculates vertical positions for all components.
        
        Args:
            n_layers: Number of sheet layers.
            tip_radius: Radius of the tip.
        """
        pm = self.pm[n_layers]
        lat_c = self.lat_c

        gap_sub_sheet = pm.calculate_gap('sub', 'sheet', buffer=0.5)
        gap_sheet_tip = pm.calculate_gap('sheet', 'tip', buffer=0.5)

        logger.info("Calculated gaps: Sub-Sheet=%.2fA, Sheet-Tip=%.2fA",
                    gap_sub_sheet, gap_sheet_tip)

        sub_thickness = self.config.sub.thickness

        self.z_positions[n_layers] = {}
        self.z_positions[n_layers]['sub'] = 0.0
        sheet_base_z = sub_thickness + gap_sub_sheet
        self.z_positions[n_layers]['sheet'] = sheet_base_z

        lat_c = (self.config.sheet.lat_c or 6.0) if lat_c is None else lat_c
        sheet_stack_height = (n_layers - 1) * lat_c
        tip_z = sheet_base_z + sheet_stack_height + gap_sheet_tip + tip_radius
        self.z_positions[n_layers]['tip'] = tip_z

    def _generate_potentials(
        self,
        n_sheet_layers: int,
    ) -> PotentialManager:
        """Configures and writes the potential file using PotentialManager.
        
        Args:
            n_sheet_layers: Number of 2D material layers.
        
        Returns:
            Configured PotentialManager instance.
        """
        pm = PotentialManager(
            self.config.settings,
            potentials_dir=self.output_dir / "provenance" / "potentials",
            potentials_prefix=str(self.relative_run_dir / "provenance" / "potentials"),
        )
        pm.set_lj_overrides(self.config.lj_override)

        pm.register_component('sub', self.config.sub)
        pm.register_component('tip', self.config.tip)

        sheet_needs_layer_types = (
            n_sheet_layers > 1 and
            pm.is_sheet_lj(self.config.sheet.pot_type)
        )
        pm.register_component(
            'sheet',
            self.config.sheet,
            n_layers=n_sheet_layers if sheet_needs_layer_types else 1
        )

        if self.config.settings.simulation.drive_method == 'virtual_atom':
            pm.register_virtual_atom()

        pm.add_self_interaction('sub')
        pm.add_self_interaction('tip')
        pm.add_self_interaction('sheet')

        pm.add_cross_interaction('sub', 'tip')
        pm.add_cross_interaction('sub', 'sheet')
        pm.add_cross_interaction('tip', 'sheet')

        if sheet_needs_layer_types and n_sheet_layers > 1:
            pm.add_interlayer_interaction('sheet')

        settings_path = self.output_dir_layer[n_sheet_layers] / "lammps" / "system.in.settings"
        pm.write_file(settings_path)

        self.groups[n_sheet_layers] = {}
        self.groups[n_sheet_layers]['sub_types'] = pm.types.get_group_string('sub')
        self.groups[n_sheet_layers]['tip_types'] = pm.types.get_group_string('tip')
        self.groups[n_sheet_layers]['sheet_types'] = pm.types.get_group_string('sheet')

        if sheet_needs_layer_types:
            for layer in range(n_sheet_layers):
                layer_key = f'sheet_l{layer+1}_types'
                self.groups[n_sheet_layers][layer_key] = pm.types.get_layer_group_string('sheet', layer)

        return pm

    def write_inputs(self, n_layers: int) -> None:
        """Generates the LAMMPS input scripts.
        
        Args:
            n_layers: Number of sheet layers for this configuration.
        """
        logger.info("Writing LAMMPS inputs...")

        pm = self.pm[n_layers]
        sheet_dims = self.sheet_dims
        lat_c = self.lat_c
        z_positions = self.z_positions[n_layers]
        groups = self.groups[n_layers]
        sheet_path = self.sheet_paths[n_layers]
        tip_path = self.tip_path
        sub_path = self.sub_path

        total_types = len(pm.types)

        sim = self.config.settings.simulation
        out = self.config.settings.output

        reaxff_types = {'reaxff', 'reax/c'}
        uses_reaxff = (
            normalize_potential_type(self.config.sheet.pot_type) in reaxff_types or
            normalize_potential_type(self.config.tip.pot_type) in reaxff_types or
            normalize_potential_type(self.config.sub.pot_type) in reaxff_types
        )
        atom_style = 'charge' if uses_reaxff else 'atomic'

        if self.config.general.scan_speed is None:
            raise ValueError("scan_speed must be specified in [general] section")

        xlo, xhi = self.sheet_dims['xlo'], self.sheet_dims['xhi']
        ylo, yhi = self.sheet_dims['ylo'], self.sheet_dims['yhi']
        zhi_box = z_positions['tip'] + 50.0

        tip_x = (xlo + xhi) / 2.0
        tip_y = (ylo + yhi) / 2.0
        tip_z = z_positions['tip']

        sub_natypes = len(groups['sub_types'].split())
        tip_natypes = len(groups['tip_types'].split())
        offset_2d = sub_natypes + tip_natypes

        context = {
            'temp': self.config.general.temp,
            'forces': self.config.general.force,
            'scan_angle_config': self.config.general.scan_angle,
            'scan_angle_force': self.config.general.scan_angle_force,
            'scan_speed_config': self.config.general.scan_speed,
            'xlo': xlo,
            'xhi': xhi,
            'ylo': ylo,
            'yhi': yhi,
            'zhi_box': zhi_box,
            'potential_file': f"{self.relative_run_dir_layer[n_layers]}/lammps/system.in.settings",
            'sub_file': f"{self.relative_run_dir}/build/{sub_path.name}",
            'tip_file': f"{self.relative_run_dir}/build/{tip_path.name}",
            'sheet_file': f"{self.relative_run_dir}/build/{sheet_path.name}",
            'path_sub': f"{self.relative_run_dir}/build/{sub_path.name}",
            'path_tip': f"{self.relative_run_dir}/build/{tip_path.name}",
            'path_sheet': f"{self.relative_run_dir}/build/{sheet_path.name}",
            'tip_x': tip_x,
            'tip_y': tip_y,
            'tip_z': tip_z,
            'sheet_z': z_positions['sheet'],
            'offset_2d': offset_2d,
            'results_file_pattern': (f"{self.relative_run_dir_layer[n_layers]}/results/"
                                    f"friction_f$(v_find)_a${{a}}_s${{speed}}_layer{n_layers}.txt"),
            'dump_file_pattern': (f"{self.relative_run_dir_layer[n_layers]}/visuals/"
                                    f"slide_f$(v_find)_a${{a}}_s${{speed}}.lammpstrj"),
            'dump_enabled': out.dump.get('slide', False),
            'z_sub': z_positions['sub'],
            'z_sheet': z_positions['sheet'],
            'z_tip': z_positions['tip'],
            'sub_types': groups['sub_types'],
            'tip_types': groups['tip_types'],
            'sheet_types': groups['sheet_types'],
            'ngroups': total_types,
            'sub_natypes': sub_natypes,
            'tip_natypes': tip_natypes,
            'timestep': sim.timestep,
            'thermo': sim.thermo,
            'neighbor_list': sim.neighbor_list,
            'neigh_modify_command': sim.neigh_modify_command,
            'run_steps': sim.slide_run_steps,
            'drive_method': sim.drive_method,
            'damp_ev': self.config.tip.dspring / 0.016,
            'spring_ev': (self.config.general.driving_spring or 8.0) / 16.02,
            'virtual_offset': self.config.tip.r * 1.5, 
            'results_freq': out.results_frequency,
            'dump_freq': out.dump_frequency.get('slide', 1000),
            'tip_fix_group': 'tip_all' if self.config.settings.geometry.rigid_tip else 'tip_fix',
            'layer_group': '2D_all',
            'n_sheet_layers': n_layers,
            'lat_c': lat_c,
            'tip_radius': self.config.tip.r,
            'sheet_dims': sheet_dims,
            'thermostat_type': self.config.settings.thermostat.type,
            'use_langevin': self.config.settings.thermostat.type == 'langevin',
            'min_style': sim.min_style,
            'minimization_command': sim.minimization_command,
            'output_dir': f"{self.relative_run_dir_layer[n_layers]}/data",
            'dump_file': f"{self.relative_run_dir_layer[n_layers]}/visuals/system.lammpstrj",
            'atom_style': atom_style,
            'ev_a_to_nn': EV_A_TO_NN,
        }

        init_script = self.render_template("afm/system_init.lmp", context)
        self.write_file("lammps/system.in", init_script, self.output_dir_layer[n_layers])

        slide_script = self.render_template("afm/slide.lmp", context)
        self.write_file("lammps/slide.in", slide_script, self.output_dir_layer[n_layers])

        logger.info("Inputs written to %s/lammps/", self.output_dir_layer[n_layers])
