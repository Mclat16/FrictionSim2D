"""Pydantic-based configuration models for FrictionSim2D.

This module provides robust, type-safe data schemas for all simulation
parameters. By leveraging Pydantic, it ensures that all configurations—from
low-level engine settings to high-level experimental parameters—are validated
for correctness before a simulation is executed.
"""
import json
import os
from pathlib import Path
from typing import List, Optional, Union, Dict, Any, Literal, cast
import yaml
from pydantic import BaseModel, Field, ValidationInfo, field_validator, model_validator

from .utils import read_config, get_potential_path, get_material_path

# --- Internal Settings ---

class GeometrySettings(BaseModel):
    """Geometry settings for tip and substrate positioning."""
    tip_reduction_factor: float = 2.25
    rigid_tip: bool = False
    tip_base_z: float = 55.0
    lat_c_default: float = 6.0
    finite_sheet_edge_width: float = Field(
        default=4.0,
        description="Edge band width for finite-sheet runs (Angstrom), typically 1-2 atoms.",
    )
    finite_sheet_afm_edge_mode: Literal['none', 'spring'] = 'none'
    finite_sheet_edge_spring_k: float = Field(
        default=10.0,
        description="Spring constant (eV/Å²) for 'spring' edge mode. Controls how tightly edge atoms are tethered to their initial positions.",
    )
    finite_sheet_sheetonsheet_edge_mode: Literal['none', 'rigid'] = 'none'

class ThermostatSettings(BaseModel):
    """Thermostat and time integration settings."""
    type: Literal['langevin', 'nose-hoover'] = 'langevin'
    time_integration: Literal['verlet', 'respa', 'nvt', 'nve'] = Field(
        default='nve', alias='time_int_type')
    langevin_boundaries: Dict[str, Dict[str, List[float]]] = Field(
        default_factory=lambda: {
            'tip': {'fix': [3.0, 0.0], 'thermo': [6.0, 3.0]},
            'sub': {'fix': [0.0, 0.3], 'thermo': [0.3, 0.6]}
        })
    model_config = {'validate_by_name': True}

class SimulationSettings(BaseModel):
    """Simulation run parameters."""
    timestep: float = Field(default=0.001, description="Timestep in picoseconds")
    thermo: int = 100000
    min_style: str = 'cg'
    minimization_command: str = 'minimize 1e-4 1e-8 1000000 1000000'
    neighbor_list: float = 0.3
    neigh_modify_command: str = 'neigh_modify every 1 delay 0 check yes'
    slide_run_steps: int = 500000
    drive_method: Literal['smd', 'fix_move', 'virtual_atom'] = 'virtual_atom'
    constraint_mode: Literal['atom_bonds', 'com_spring', 'none'] = 'none'


class QuenchSettings(BaseModel):
    """Quenching parameters for amorphous material generation."""
    run_local: bool = True
    n_procs: int = 16
    quench_slab_dims: List[int] = Field(default_factory=lambda: [200, 200, 50])
    quench_rate: float = 1e12
    melt_temp: float = Field(default=2500.0, alias='quench_melt_temp')
    quench_temp: float = Field(default=300.0, alias='quench_target_temp')
    timestep: float = 0.002
    melt_steps: int = 50000
    quench_steps: int = 100000
    equilibrate_steps: int = 20000
    model_config = {'validate_by_name': True}

class OutputSettings(BaseModel):
    """Output and dump settings."""
    dump: Dict[str, bool] = Field(
        default_factory=lambda: {'system_init': True, 'slide': True})
    dump_frequency: Dict[str, int] = Field(
        default_factory=lambda: {'system_init': 1000, 'slide': 10000})
    results_frequency: int = 1000

class PotentialSettings(BaseModel):
    """Settings for interatomic potentials."""
    lj_cutoff: float = Field(default=11.0, alias='LJ_cutoff')
    lj_type: str = Field(default='LJ_base', alias='LJ_type')
    reaxff_safezone: float = Field(
        default=1.2,
        description="Memory safety multiplier for ReaxFF neighbor lists"
    )
    reaxff_mincap: int = Field(
        default=50,
        description="Minimum array capacity for ReaxFF internals"
    )
    model_config = {'validate_by_name': True}

class HPCSettings(BaseModel):
    """HPC cluster and job submission settings."""
    scheduler_type: Literal['pbs', 'slurm'] = 'pbs'
    queue: Optional[str] = None
    partition: Optional[str] = None
    account: str = ''
    hpc_host: Optional[str] = None
    hpc_home: Optional[str] = None
    log_dir: Optional[str] = None
    scratch_dir: Optional[str] = "$TMPDIR"
    num_nodes: int = 1
    num_cpus: int = 32
    memory_gb: int = 62
    walltime_hours: int = 20
    max_array_size: int = 300
    modules: Optional[List[str]] = Field(default_factory=lambda: None)
    mpi_command: str = "mpirun"
    use_tmpdir: bool = True
    lammps_scripts: List[str] = Field(default_factory=lambda: [
        'system.in',
        'slide.in'
    ])


class AiidaSettings(BaseModel):
    """AiiDA-specific workflow settings."""
    enabled: bool = False
    lammps_code_label: str = 'lammps@my_hpc'
    postprocess_code_label: str = 'python@my_hpc'
    postprocess_script_path: str = ''
    create_provenance: bool = True
    auto_import_results: bool = False
    hpc_mode: Literal['local', 'remote', 'offline'] = 'offline'

    # AiiDA remote computer configuration (optional, only for 'aiida setup')
    computer_label: Optional[str] = 'localhost'
    transport: Literal['local', 'ssh'] = 'local'
    hostname: Optional[str] = None
    workdir: Optional[str] = None
    username: Optional[str] = None
    ssh_port: int = 22
    key_filename: Optional[str] = None


class DatabaseProfileSettings(BaseModel):
    """Connection parameters for a single database profile."""
    host: str = 'localhost'
    port: int = 5432
    dbname: str = 'frictionsim2ddb'
    user: str = ''
    password: str = ''
    api_key: str = ''


class DatabaseSettings(BaseModel):
    """Database connection and staging pipeline configuration."""
    active_profile: str = 'local'
    local: DatabaseProfileSettings = Field(default_factory=DatabaseProfileSettings)
    central: DatabaseProfileSettings = Field(
        default_factory=lambda: DatabaseProfileSettings(
            host=os.environ.get('FRICTION_CENTRAL_DB_HOST', ''),
            port=int(os.environ.get('FRICTION_CENTRAL_DB_PORT', '5432')),
            dbname=os.environ.get('FRICTION_CENTRAL_DB_NAME', 'frictionsim2ddb'),
            user=os.environ.get('FRICTION_CENTRAL_DB_USER', ''),
            password=os.environ.get('FRICTION_CENTRAL_DB_PASSWORD', ''),
            api_key=os.environ.get('FRICTION_DB_API_KEY', ''),
        )
    )
    auto_validate: bool = True
    skip_fraction: float = 0.2
    api_url: str = os.environ.get('FRICTION_CENTRAL_API_URL', 'http://localhost:8000')
    api_host: str = '0.0.0.0'
    api_port: int = 8000


class GlobalSettings(BaseModel):
    """Represents the full structure of settings.yaml with hardcoded defaults."""
    geometry: GeometrySettings = Field(default_factory=GeometrySettings)
    thermostat: ThermostatSettings = Field(default_factory=ThermostatSettings)
    simulation: SimulationSettings = Field(default_factory=SimulationSettings)
    quench: QuenchSettings = Field(default_factory=QuenchSettings)
    output: OutputSettings = Field(default_factory=OutputSettings)
    potential: PotentialSettings = Field(default_factory=PotentialSettings)
    hpc: HPCSettings = Field(default_factory=HPCSettings)
    aiida: AiidaSettings = Field(default_factory=AiidaSettings)
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)

# --- User Input Settings (From .ini files) ---

class ComponentConfig(BaseModel):
    """Base configuration for any material component.

    Attributes:
        mat: Material identifier.
        pot_type: Potential type (e.g., 'sw', 'tersoff').
        pot_path: Path to potential file.
        cif_path: Path to CIF structure file.
    """
    mat: str
    pot_type: str
    pot_path: str
    cif_path: str

    @field_validator('pot_path', 'cif_path', mode='after')
    @classmethod
    def resolve_path(cls, v: str, info: ValidationInfo) -> str:
        """Resolve filesystem paths from config values.

        Args:
            v: Path value from config.
            info: Pydantic validation info including field name.

        Returns:
            Resolved absolute path as string.
        """
        path = Path(v)
        if path.exists():
            return str(path)
        if info.field_name == 'pot_path':
            resolved = get_potential_path(v)
        else:
            resolved = get_material_path(v)

        return str(resolved) if resolved.exists() else v

class TipConfig(ComponentConfig):
    """Tip configuration parameters."""
    r: float = Field(..., description="Tip radius in Angstroms")
    amorph: Literal['c', 'a'] = Field('c', description="'c' for crystalline, 'a' for amorphous")
    dspring: Optional[float] = Field(
        0.0,
        description=(
            "Explicit viscous damping override (raw value passed to LAMMPS as damp_ev = dspring/0.016). "
            "Defaults to 0.0 (no damping). Any set value (including 0.0) overrides damping_ratio; set it "
            "to null/empty to auto-compute the viscous coefficient from damping_ratio instead."
        ),
    )
    damping_ratio: Optional[float] = Field(
        0.0072,
        description=(
            "Target damping ratio (zeta) of critical for the spring-driven tip. The viscous coefficient "
            "is computed in the LAMMPS script from the driving spring constant and the actual tip "
            "mass/atom count, so it stays at this zeta for any tip size or spring. Used only when "
            "dspring is None. zeta=1 is critical; ~0.05-0.1 is a light, low-contamination default."
        ),
    )
    tip_x: Optional[float] = Field(
        default=None,
        description="Optional absolute AFM tip x-position in box coordinates.",
    )
    tip_y: Optional[float] = Field(
        default=None,
        description="Optional absolute AFM tip y-position in box coordinates.",
    )
    tip_x_offset: float = Field(
        default=0.0,
        description="AFM tip x-offset applied only when tip_x is not explicitly set.",
    )
    tip_y_offset: float = Field(
        default=0.0,
        description="AFM tip y-offset applied only when tip_y is not explicitly set.",
    )

    @field_validator('amorph', mode='before')
    @classmethod
    def handle_empty_amorph(cls, v):
        """Convert None to default 'c' value."""
        return 'c' if v is None else v

class SubstrateConfig(ComponentConfig):
    """Substrate configuration parameters."""
    thickness: float
    x: Optional[float] = Field(
        default=None,
        description="Finite-sheet substrate size in x (same units as sheet x).",
    )
    y: Optional[float] = Field(
        default=None,
        description="Finite-sheet substrate size in y (same units as sheet y).",
    )
    amorph: Literal['c', 'a'] = Field('c', description="'c' for crystalline, 'a' for amorphous")
    @field_validator('amorph', mode='before')
    @classmethod
    def handle_empty_amorph(cls, v):
        """Convert None to default 'c' value."""
        return 'c' if v is None else v

class SheetConfig(ComponentConfig):
    """Sheet configuration parameters."""
    x: Union[float, List[float]]
    y: Union[float, List[float]]
    layers: List[int]
    stack_type: str = 'AA'
    lat_c: Optional[float] = None

class FlakeConfig(ComponentConfig):
    """Shaped-flake configuration (a finite patch placed between tip and sheet).

    The flake is cut from a supercell of its own 2D material into a triangle,
    square or hexagon. During AFM indentation its corner atoms are tethered in
    the xy-plane (free in z) so the flake keeps its shape; the restraint is
    released before sliding.

    Attributes:
        shape: Flake outline ('triangle', 'square', 'hexagon').
        edge_length: Characteristic size (Å). Triangle/square: side length;
            hexagon: circumradius (center-to-vertex distance).
        rotation_deg: In-plane rotation of the shape (degrees, CCW).
        x, y: Supercell footprint (Å) to carve the flake from. Defaults to
            roughly twice ``edge_length`` so the shape always fits.
        corner_radius: Capture radius (Å) around each detected vertex used to
            select corner atoms.
        corner_spring_k: Spring constant (eV/Å²) for the xy corner tether
            applied during indentation.
        center_x, center_y: Optional flake center override (box coords). When
            omitted the flake is centered under the tip.
    """
    shape: Literal['triangle', 'square', 'hexagon'] = 'hexagon'
    edge_length: float = Field(..., description="Characteristic flake size in Angstroms")
    rotation_deg: float = 0.0
    x: Optional[float] = None
    y: Optional[float] = None
    corner_radius: float = Field(
        default=5.0,
        description="Capture radius (Å) around each vertex for corner-atom selection.",
    )
    corner_spring_k: float = Field(
        default=10.0,
        description="xy spring constant (eV/Å²) tethering corner atoms during indentation.",
    )
    center_x: Optional[float] = None
    center_y: Optional[float] = None

    @model_validator(mode='after')
    def default_supercell_size(self) -> 'FlakeConfig':
        """Size the carving supercell to comfortably contain the shape."""
        margin = max(2.0 * self.corner_radius, 10.0)
        # Hexagon edge_length is a circumradius (half-width); triangle/square
        # edge_length is a full side. Use 2x as a safe footprint either way.
        span = 2.0 * self.edge_length + margin
        if self.x is None:
            self.x = span
        if self.y is None:
            self.y = span
        return self


class GeneralConfig(BaseModel):
    """General simulation parameters."""
    temp: float = 300.0
    force: Optional[Union[float, List[float]]] = None
    pressure: Optional[Union[float, List[float]]] = None
    scan_angle: Optional[Union[float, List[float]]] = 0.0
    scan_angle_force: Optional[Union[float, List[float]]] = Field(
        None,
        description=(
            "Optional force/pressure selector for applying the scan_angle list. "
            "Accepts a single value or a list of values. If omitted, all scan "
            "angles are applied to all force/pressure values."
        )
    )
    scan_speed: Optional[Union[float, List[float]]] = 2.0
    outer_loop: Optional[Literal['pressure', 'scan_speed', 'force']] = Field(
        None,
        description=(
            "Parameter expanded as separate LAMMPS slide input files "
            "(slide_*.in), one per value, instead of a single in-script loop. "
            "'pressure'/'scan_speed' apply to sheet-on-sheet runs; 'force' "
            "applies to AFM runs (emits slide_f<load>N.in per load, all sharing "
            "the single system.in indentation phase). If omitted, single-script "
            "behavior is used."
        )
    )
    bond_spring: Optional[float] = Field(80.0, description="Spring constant for harmonically bonded sheets")
    driving_spring: Optional[float] = Field(50, description="Driving spring constant for virtual atom method")
    finite_sheet: bool = Field(
        default=False,
        description="Enable finite-sheet AFM geometry for this run.",
    )


class PESConfig(BaseModel):
    """Potential-energy-surface (PES) static lateral-scan parameters.

    A PES scan replaces the dynamic sliding run with a cheap, static
    energy-vs-lateral-position map over one surface unit cell (see
    ``builders.pes_scan``). It is shared by both PES flavours:

        - ``pes_sheet``: a frozen bilayer whose top layer is grid-scanned
          laterally, relaxing only the out-of-plane coordinate at each point.
        - ``pes_tip``: the AFM tip is grid-scanned laterally over the sheet
          while the sheet + substrate relax; the lateral force on the tip is
          recorded alongside the energy.

    Attributes:
        grid_n: Number of grid points per lattice vector (N in the N×N scan).
        grid_n_refine: Optional finer grid for a convergence check. When set,
            a second scan script at this resolution is also emitted.
        z_relax: Sheet scan only — relax the top layer's z at each grid point
            (relaxed PES). When False the layers stay rigid (rigid PES).
        n_cells_x, n_cells_y: Optional explicit surface-cell tiling factors
            spanned by the scan. When omitted the full periodic box (one built
            supercell repeat) is scanned, which the builder normalizes per cell.
    """
    grid_n: int = Field(
        default=12,
        ge=2,
        description="N for the N×N lateral grid over one surface unit cell.",
    )
    grid_n_refine: Optional[int] = Field(
        default=None,
        ge=2,
        description="Optional finer grid resolution for a convergence check.",
    )
    z_relax: bool = Field(
        default=True,
        description="Sheet scan: relax top-layer z at each grid point (relaxed PES).",
    )
    tip_load: float = Field(
        default=30.0,
        gt=0,
        description=(
            "Tip scan only — the single normal load (nN) at which the tip is pressed "
            "into contact before scanning. The descriptor must be built at a STABLE "
            "contact load: at ~2 nN the tip barely holds contact (the real sims are "
            "near-random there), so the deployable scan uses a firm load in the "
            "5-50 nN friction regime (default 30 nN, chosen data-drivenly as the "
            "cof_std plateau / densest-sampled real load)."
        ),
    )
    tip_load_sweep: Optional[List[float]] = Field(
        default=None,
        description=(
            "Tip scan only — optional diagnostic list of loads (nN) to scan at, each "
            "emitting its own script/grid (e.g. [2, 20, 30, 50]). Produces the "
            "'descriptor predictive value climbs as it leaves the 2 nN pathology' "
            "figure; not the deployable descriptor (that is the single tip_load)."
        ),
    )


class AFMSimulationConfig(BaseModel):
    """Master configuration object for an AFM simulation run."""
    general: GeneralConfig
    tip: TipConfig
    sub: SubstrateConfig
    sheet: SheetConfig = Field(..., alias='2D')
    flake: Optional[FlakeConfig] = Field(default=None, alias='flake')
    pes: Optional[PESConfig] = Field(default=None, alias='pes')
    lj_override: Dict[str, Any] = Field(default_factory=dict, alias='lj_override')
    settings: GlobalSettings

    @model_validator(mode='after')
    def validate_non_finite_sheet_substrate(self) -> 'AFMSimulationConfig':
        """Require amorphous substrate when AFM finite-sheet mode is disabled."""
        if not self.general.finite_sheet and self.sub.amorph != 'a':
            raise ValueError(
                "Non-finite-sheet AFM requires an amorphous substrate "
                "(set [sub] amorph = a)."
            )
        return self

    @model_validator(mode='after')
    def validate_finite_sheet_geometry(self) -> 'AFMSimulationConfig':
        """Validate finite-sheet substrate sizing against the sheet footprint."""
        if not self.general.finite_sheet:
            return self

        if self.sub.x is None or self.sub.y is None:
            raise ValueError(
                "Finite-sheet AFM requires substrate x and y to be set in the [sub] section."
            )

        sheet_x = float(self.sheet.x[0] if isinstance(self.sheet.x, list) else self.sheet.x)
        sheet_y = float(self.sheet.y[0] if isinstance(self.sheet.y, list) else self.sheet.y)
        sub_x = float(self.sub.x)
        sub_y = float(self.sub.y)

        if sub_x < sheet_x:
            raise ValueError(
                f"Finite-sheet substrate x ({sub_x}) cannot be smaller than sheet x ({sheet_x})."
            )
        if sub_y < sheet_y:
            raise ValueError(
                f"Finite-sheet substrate y ({sub_y}) cannot be smaller than sheet y ({sheet_y})."
            )

        lj_cutoff = float(self.settings.potential.lj_cutoff)
        margin_x = sub_x - sheet_x
        margin_y = sub_y - sheet_y
        if margin_x <= lj_cutoff:
            raise ValueError(
                "Finite-sheet AFM requires (substrate_x - sheet_x) to be greater than LJ cutoff "
                f"({lj_cutoff}). Got {margin_x}."
            )
        if margin_y <= lj_cutoff:
            raise ValueError(
                "Finite-sheet AFM requires (substrate_y - sheet_y) to be greater than LJ cutoff "
                f"({lj_cutoff}). Got {margin_y}."
            )

        return self

class SheetOnSheetSimulationConfig(BaseModel):
    """Master configuration object for a Sheet-on-Sheet simulation run."""
    general: GeneralConfig
    sheet: SheetConfig = Field(..., alias='2D')
    pes: Optional[PESConfig] = Field(default=None, alias='pes')
    lj_override: Dict[str, Any] = Field(default_factory=dict, alias='lj_override')
    settings: GlobalSettings

# --- Helper Functions ---

def _global_settings_path() -> Path:
    """Return the semi-permanent per-user settings path.

    Respects ``$XDG_CONFIG_HOME`` on Linux; falls back to ``~/.config``.
    """
    xdg_config_home = os.environ.get('XDG_CONFIG_HOME')
    base = Path(xdg_config_home) if xdg_config_home else (Path.home() / '.config')
    return base / 'FrictionSim2D' / 'settings.yaml'


def _settings_paths_in_precedence_order(settings_file: Optional[Union[str, Path]] = None) -> List[Path]:
    """Return candidate settings files in descending precedence order.

    Order:
      1. Explicit ``settings_file`` argument (user-provided for this run)
      2. ``~/.config/FrictionSim2D/settings.yaml`` (semi-permanent global)

    If neither is present, hardcoded :class:`GlobalSettings` defaults are used.
    """
    paths: List[Path] = []

    if settings_file:
        paths.append(Path(settings_file).expanduser())

    paths.append(_global_settings_path())
    return paths


def load_settings(settings_file: Optional[Union[str, Path]] = None) -> GlobalSettings:
    """Load settings onto hardcoded defaults.

    Search order (first file found wins):
      1. Explicit ``settings_file`` argument
      2. ``~/.config/FrictionSim2D/settings.yaml`` (semi-permanent global)

    If no file is found, the hardcoded Pydantic defaults in
    :class:`GlobalSettings` are returned unchanged.

    Args:
        settings_file: Optional path to a settings YAML file for explicit,
            per-run configuration.

    Returns:
        :class:`GlobalSettings` populated from the first matching file, or
        pure defaults if no file exists.
    """
    for path in _settings_paths_in_precedence_order(settings_file=settings_file):
        try:
            if not path.is_file():
                continue
            with path.open('r', encoding='utf-8') as f:
                data = yaml.safe_load(f) or {}
            if data:
                return cast(GlobalSettings, GlobalSettings.model_validate(data))
        except (FileNotFoundError, OSError, yaml.YAMLError):
            continue

    return cast(GlobalSettings, GlobalSettings())


def settings_origin(settings_file: Optional[Union[str, Path]] = None) -> Optional[Path]:
    """Return the path of the settings file currently in effect, or ``None``.

    Useful for ``settings show --origin`` to tell the user which file is being used.
    """
    for path in _settings_paths_in_precedence_order(settings_file=settings_file):
        if path.is_file():
            return path
    return None


def parse_config(config_source: Union[str, Path, Dict[str, Any]]) -> Dict[str, Any]:
    """Parses configuration from various sources into a dictionary suitable for Pydantic.

    This function acts as a unified entry point for configuration loading. It can
    handle:
    1. File paths (str or Path) pointing to .ini, .yaml/.yml, or .json files.
    2. Dictionaries (e.g., from a CLI arg parser or UI form).

    Args:
        config_source (Union[str, Path, Dict]): The configuration source.

    Returns:
        Dict[str, Any]: A standardized dictionary ready for validation.

    Raises:
        ValueError: If the file extension is not supported.
        TypeError: If the input type is not supported.
    """
    if isinstance(config_source, (str, Path)):
        path = Path(config_source)
        ext = path.suffix.lower()

        if ext == '.ini':
            return read_config(path)

        if ext in ('.yaml', '.yml'):
            with open(path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f) or {}

        if ext == '.json':
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)

        else:
            raise ValueError(f"Unsupported configuration file format: {ext}. Supported formats: .ini, .yaml, .yml, .json")

    elif isinstance(config_source, dict):
        return config_source

    else:
        raise TypeError(f"Unsupported configuration source type: {type(config_source)}")
