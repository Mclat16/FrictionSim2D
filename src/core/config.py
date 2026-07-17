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
    tip_start_clearance: float = Field(
        default=20.0,
        description=(
            "Vertical clearance (Angstrom) placed above the contact gap so the tip "
            "starts clear of the surface during equilibration. After equilibration the "
            "indentation script lowers the tip by a measured amount so its lowest atom "
            "sits exactly the contact gap above the indented surface."
        ),
    )
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
    hetero_stacking: Literal['grouped', 'alternating'] = Field(
        default='grouped',
        description=(
            "Layer ordering for multi-material heterostructure stacks. 'grouped' "
            "stacks all layers of one material before the next; 'alternating' "
            "interleaves layers across materials."
        ),
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
    tip_load: Optional[float] = Field(
        default=5.0,
        description=(
            "Tip scan only — the target grid-mean normal load (nN) for a constant-load "
            "campaign. The scan itself is constant-height (the tip is held at a gap), "
            "so the load can only be known from a full scan; the campaign runner "
            "therefore calibrates each material's fz(gap) curve and picks the gap whose "
            "grid-mean load lands on this target, then runs the scan there. The gap→load "
            "calibration (see gap_load_calibration/) showed the bilayer reaches ~2-12 nN "
            "inside its stable gap window; ~5 nN is the clean default (strong, resolvable "
            "corrugation for all but the most compliant sheets, which cap at their max "
            "achievable load). Null/0 → the fixed contact gap (an adhesion descriptor)."
        ),
    )
    tip_load_sweep: Optional[List[float]] = Field(
        default=None,
        description=(
            "Tip scan only — optional diagnostic list of loads (nN) to scan at, each "
            "emitting its own script/grid (e.g. [2, 5, 10]). Produces the 'descriptor "
            "predictive value climbs as it leaves the 2 nN pathology' figure; not the "
            "deployable descriptor (that is the single tip_load). NB currently a load "
            "series is run as separate campaigns (one tip_load each); per-sim "
            "multi-grid emission is not yet wired."
        ),
    )
    scan_cells: int = Field(
        default=1,
        ge=1,
        description=(
            "Tip scan only — number M of surface unit cells (M×M) the lateral scan "
            "spans, at grid_n points *per cell* (so the emitted grid is M·grid_n square "
            "over M·a × M·b). M=1 (default) is the standard single-cell scan and changes "
            "nothing. M>1 is a representativeness diagnostic: on a disordered/amorphous "
            "backing a single cell is one random patch of a non-periodic PES, and only a "
            "multi-cell scan can measure the cell-to-cell descriptor spread (the error "
            "bar) and separate real periodic corrugation from substrate noise. Needs a "
            "sheet footprint large enough that the tip clears its own periodic image over "
            "the M-cell travel (the builder warns if not)."
        ),
    )
    eval_mode: Literal['minimize', 'md'] = Field(
        default='minimize',
        description=(
            "Tip freestanding scan only — how each grid point is evaluated. "
            "'minimize' = static energy minimization (0 K, cheapest). 'md' = a short "
            "NVE + Langevin run of md_steps with the layer_2 thermostat band active, "
            "time-averaging energy/force (a finite-T PES / potential of mean force)."
        ),
    )
    md_steps: int = Field(
        default=2000,
        ge=1,
        description="Steps per grid point when eval_mode = 'md' (ignored for 'minimize').",
    )


class IndentConfig(BaseModel):
    """AFM indentation ("indent") parameters — damped hold + pull-off retract.

    An indent run does the standard AFM build + indentation ramp (``system.in``)
    to each normal load in ``[general] force``, then (see :mod:`src.builders.indent`,
    ``POKE_SIM_RECIPE.md``) runs one hold+retract per load without ever sliding:

        1. holds the (rigid, laterally-fixed) tip at the target load for a short
           *damped* finite-T hold, recording the out-of-plane channels — the
           thermally-averaged penetration depth (``tip_z − sheet_com_z``) is the
           vertical-compliance descriptor the lateral PES scan cannot see; then
        2. tears down the MD fixes and **retracts the rigid tip quasi-statically**
           (T=0 minimization) in small z-steps — through zero load, past the
           pull-off instability, to full detachment — giving the force–distance
           curve for the **pull-off force**, **work of adhesion** and **contact
           stiffness** (the adhesion descriptors no sliding/hold data holds).

    The loads come from ``[general] force`` (``{10, 30}`` nN are the deployable
    2-point anchor; ``{5, 20, 50}`` add the 9-point stiffness ceiling and
    load-resolved pull-off), the layers from ``[2D] layers`` and the angle is 0 —
    matching the campaign labels. This one type subsumes the former separate poke
    (hold-only) and adhesion (retract-only) simulations.

    Attributes:
        hold_steps: Length of the damped hold at the target load, in steps
            (10 000 steps = 10 ps at the 1 fs campaign timestep). At the 1 ps
            (every-1000-step) campaign cadence this yields ~10 samples per
            condition — penetration is thermally stable over the window.
        z_step: Retract increment per step, in Å (recipe: 0.1–0.2 Å for a
            quasi-static curve that resolves the pull-off instability).
        n_steps: Number of retract steps. ``n_steps × z_step`` is the total
            retract distance — it must comfortably exceed the adhesive tail so the
            curve reaches full detachment (default 100 × 0.2 Å = 20 Å).
        seed: Base velocity seed for the campaign. Each ``(material, layer)``
            derives its own fresh, reproducible velocity-create seed from this base
            (recorded in the manifest). **A fresh seed per run is mandatory** —
            reusing a campaign seed reproduces the campaign contact trajectory and
            re-imports the same-run optimism the verification excludes. When
            omitted a fresh base seed is drawn once at build time.
    """
    hold_steps: int = Field(
        default=10000,
        ge=1000,
        description="Damped-hold length in steps at the target load (10 000 = 10 ps).",
    )
    z_step: float = Field(
        default=0.2,
        gt=0.0,
        description="Retract increment per step in Å (recipe: 0.1–0.2 Å).",
    )
    n_steps: int = Field(
        default=100,
        ge=1,
        description="Number of retract steps (n_steps × z_step = total retract distance).",
    )
    seed: Optional[int] = Field(
        default=None,
        ge=1,
        description=(
            "Base velocity seed; per-(material, layer) fresh seeds are derived "
            "from it and recorded in the manifest. Omit to draw a fresh base "
            "seed at build time. Never reuse a campaign seed."
        ),
    )


class LatticeMatchConfig(BaseModel):
    """Lattice-matching parameters for heterostructure interfaces."""
    method: Literal['zur_mcgill'] = 'zur_mcgill'
    strain_tol: float = Field(default=0.02, ge=0.0, le=1.0)
    max_supercell: int = Field(default=10, ge=1)
    area_tol: float = Field(default=0.10, ge=0.0, le=1.0)


class AFMSimulationConfig(BaseModel):
    """Master configuration object for an AFM simulation run."""
    general: GeneralConfig
    tip: TipConfig
    sub: Optional[SubstrateConfig] = Field(default=None, alias='sub')
    sheet: SheetConfig = Field(..., alias='2D')
    flake: Optional[FlakeConfig] = Field(default=None, alias='flake')
    pes: Optional[PESConfig] = Field(default=None, alias='pes')
    indent: Optional[IndentConfig] = Field(default=None, alias='indent')
    lj_override: Dict[str, Any] = Field(default_factory=dict, alias='lj_override')
    settings: GlobalSettings

    @model_validator(mode='after')
    def validate_non_finite_sheet_substrate(self) -> 'AFMSimulationConfig':
        """Require amorphous substrate for a substrate-backed non-finite-sheet AFM.

        A freestanding run (no [sub] section, self.sub is None) has no substrate to
        constrain, so the amorphous requirement does not apply.
        """
        if self.sub is not None and not self.general.finite_sheet and self.sub.amorph != 'a':
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

        if self.sub is None:
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
    sheets: List[SheetConfig] = Field(
        default_factory=list,
        description=(
            "Per-material sheet stack, folded from [2D-1], [2D-2], ... sections "
            "(or a single-element list wrapping [2D] when only one material is "
            "given). Populated by _fold_sheet_sections before validation."
        ),
    )
    pes: Optional[PESConfig] = Field(default=None, alias='pes')
    lattice_match: Optional[LatticeMatchConfig] = Field(default=None, alias='lattice_match')
    lj_override: Dict[str, Any] = Field(default_factory=dict, alias='lj_override')
    settings: GlobalSettings

    @model_validator(mode='before')
    @classmethod
    def _fold_sheet_sections(cls, data: Any) -> Any:
        """Fold [2D-1], [2D-2], ... sections into ``sheets``.

        Numbered ``2D-N`` sections are collected in numeric order into the
        ``sheets`` list. When only a single ``[2D]`` (alias) section is
        present, it is wrapped as a one-element ``sheets`` list, preserving
        the homogeneous single-material path. The ``2D``/``sheet`` field
        stays required for backward compatibility: when only numbered
        sections are given, the first is also mirrored onto ``2D`` so
        existing single-sheet code paths keep working unchanged.
        """
        if not isinstance(data, dict):
            return data

        folded = dict(data)
        two_d_keys = sorted(
            (k for k in folded if isinstance(k, str) and k.startswith('2D-')),
            key=lambda key: (
                int(key.split('-', maxsplit=1)[1])
                if key.split('-', maxsplit=1)[1].isdigit() else 10**9
            ),
        )

        if two_d_keys:
            folded['sheets'] = [folded[k] for k in two_d_keys]
            folded.setdefault('2D', folded[two_d_keys[0]])
        elif '2D' in folded:
            folded['sheets'] = [folded['2D']]

        return folded

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
