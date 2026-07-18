"""D1 hetero sheet-on-sheet slide assembly (consumes hetero.py's structure builder)."""
import logging
from typing import List, Dict, Optional

import numpy as np
from ase.io import read
from jinja2 import Environment

from ..core.config import SheetOnSheetSimulationConfig
from ..core.simulation_base import SimulationBase
from ..data.models import EV_A_TO_NN, EV_A3_TO_GPA, NM_TO_EV_A2
from ..interfaces.jinja import PackageLoader
from .hetero import _detect_atom_style, build_hetero_structure

logger = logging.getLogger(__name__)

PAD = 1.0  # Å pad above/below each layer's atom z-range

# Same Jinja environment construction as SimulationBase.__init__
# (src/core/simulation_base.py:80-84): package-resource loader, trimmed blocks.
_ENV = Environment(
    loader=PackageLoader('src.templates'),
    trim_blocks=True,
    lstrip_blocks=True,
)


def render_hetero_slide(context: Dict) -> str:
    """Render ``hetero/slide.lmp`` with the given context.

    Thin module-level helper mirroring how builders render templates via
    ``SimulationBase.render_template``, without needing a full builder
    instance.
    """
    return _ENV.get_template("hetero/slide.lmp").render(context)


def compute_layer_zbands(data_path, layers) -> List[Dict]:
    """Compute per-layer z-bands from each layer's KNOWN placement.

    Each layer in ``layers`` (a :class:`~src.builders.hetero.HeteroStackLayer`)
    records ``.source`` (the per-material supercell data file it was read
    from) and ``.z`` (the z-shift applied via
    ``read_data ... shift 0 0 z``, see templates/hetero/stack_hetero.lmp).
    So the placed atoms of a layer occupy exactly
    ``[native_zmin + z, native_zmax + z]``, where ``native_z`` comes from
    reading ``.source``. This is exact and independent of gap heuristics, so
    (unlike clustering on the (n-1) largest z-gaps) it cannot be fooled by a
    puckered layer whose internal sub-plane gap exceeds an inter-layer gap
    (e.g. black phosphorus, GeS).

    Multiple layers of the SAME material share one ``.source`` file but have
    DIFFERENT ``.z``, so their bands are still distinct.

    Returns one ``{'idx', 'zlo', 'zhi'}`` per layer (idx 1..n bottom->top).

    Raises:
        ValueError: If ``data_path``'s actual atom z-coordinates disagree with
            the bands derived from known placement (i.e. some atom falls in
            zero or more than one band). This is a self-check that makes a
            silent mis-grouping impossible -- it fails loud instead.
    """
    native_zrange: Dict[str, tuple] = {}
    placed = []
    for layer in layers:
        key = str(layer.source)
        if key not in native_zrange:
            zz = read(key, format="lammps-data").get_positions()[:, 2]
            native_zrange[key] = (float(zz.min()), float(zz.max()))
        zmin, zmax = native_zrange[key]
        zlo, zhi = zmin + layer.z - PAD, zmax + layer.z + PAD
        center = 0.5 * (zmin + zmax) + layer.z
        placed.append((center, zlo, zhi))

    placed.sort(key=lambda t: t[0])
    bands = [{"idx": i + 1, "zlo": zlo, "zhi": zhi} for i, (_, zlo, zhi) in enumerate(placed)]

    # Self-check (fail loud): every atom in the actual assembled data file
    # must fall in exactly one band. Catches any placement/write discrepancy
    # instead of silently mis-grouping.
    z = read(str(data_path), format="lammps-data").get_positions()[:, 2]
    for zi in z:
        hits = [b for b in bands if b["zlo"] <= zi <= b["zhi"]]
        if len(hits) != 1:
            raise ValueError(
                f"compute_layer_zbands: atom at z={zi:.4f} falls in {len(hits)} "
                f"bands (expected exactly 1); bands={bands}. The assembled "
                f"{data_path} disagrees with the known layer placement "
                f"(source native z-range + z-shift) -- check read_data shift "
                f"semantics or PAD={PAD}."
            )
    return bands


class HeteroSheetOnSheetSimulation(SimulationBase):
    """Builder for a periodic heterostructure sheet-on-sheet slide.

    Turns a two-material :class:`SheetOnSheetSimulationConfig` into a runnable
    periodic hetero slide by combining the Phase-B structure builder
    (:func:`~src.builders.hetero.build_hetero_structure`, which assembles
    ``hetero.data`` + ``lammps/system.in.settings`` and stages the potentials)
    with the D1 slide template (``templates/hetero/slide.lmp``).

    The assembled N-layer stack (N = sum of each material's layer count, >= 4)
    is a fixed bottom / thermostatted middle / driven top slide, exactly like the
    homogeneous :class:`~src.builders.sheetonsheet.SheetOnSheetSimulation`, but
    the layer groups are defined by z-slab region (hetero layers share atom
    types, so type-based groups are impossible) and the interlayer friction is
    read from the bottom-layer reaction force (many-body ``sw`` hybrid, no
    ``compute group/group``).

    Path convention (diverges from the homogeneous builder, intentional for D1):
    every path in the render context is RUN-DIR-RELATIVE (``hetero.data``,
    ``lammps/system.in.settings``, ``results/...``, ``visuals/...``). The stack
    is built directly into ``output_dir`` so the potential settings' run-dir-
    relative ``provenance/potentials/...`` prefix lines up when LAMMPS is run
    with ``cwd = output_dir`` (Task 5's e2e).
    """

    #: A dynamic hetero slide needs a fixed bottom, a thermostatted middle and a
    #: driven top across TWO materials -> at least 2 + 2 layers.
    MIN_LAYERS: int = 4

    def __init__(self, config: SheetOnSheetSimulationConfig, output_dir: str,
                 config_path: Optional[str] = None):
        super().__init__(config, output_dir, config_path=config_path)
        self.config: SheetOnSheetSimulationConfig = config

    @property
    def n_layers(self) -> int:
        """Total layer count = sum of each material's single layer count.

        Each material's ``layers`` is a single-element list (e.g. ``[2]``); a
        2+2 stack is 4 layers. Mirrors
        :attr:`SheetOnSheetSimulation.n_layers`' single-count requirement.
        """
        for sheet in self.config.sheets:
            if len(sheet.layers) != 1:
                raise ValueError(
                    "Hetero sheet-on-sheet currently requires exactly one value "
                    "in each material's 2D.layers (e.g., layers=[2])."
                )
        return sum(int(sheet.layers[0]) for sheet in self.config.sheets)

    def _init_provenance(self) -> None:
        """Create the provenance folder and record each material's input files.

        :class:`SimulationBase` has no ``_init_provenance`` (only
        :class:`SheetOnSheetSimulation` does, and that one is single-material),
        so the hetero builder provides its own: one provenance component per
        stacked material. ``build_hetero_structure`` separately stages the real
        potential files under ``provenance/potentials/`` for the run itself.
        """
        prov_dir = self.output_dir / "provenance"
        prov_dir.mkdir(parents=True, exist_ok=True)
        for i, sheet in enumerate(self.config.sheets):
            self._add_component_files_to_provenance(f"sheet_{i + 1}", sheet)

    def build(self) -> None:
        """Assemble the hetero stack and write ``lammps/slide.in``."""
        n = self.n_layers
        if n < self.MIN_LAYERS:
            raise ValueError(
                f"Hetero sheet-on-sheet requires at least {self.MIN_LAYERS} "
                f"layers, got {n}"
            )

        logger.info("Building %d-layer heterostructure sheet-on-sheet slide...", n)
        self._create_directories()
        self._init_provenance()

        # Build the stack INTO the run dir so the potential settings' run-dir-
        # relative prefix (provenance/potentials/...) resolves when LAMMPS runs
        # from output_dir (Task 5). This writes output_dir/hetero.data,
        # output_dir/lammps/system.in.settings and output_dir/provenance/.
        stack = build_hetero_structure(
            self.config, self.config.settings, workdir=self.output_dir
        )

        bands = compute_layer_zbands(stack.data_path, stack.layers)

        # Box from the assembled hetero.data (template adds the z-vacuum itself
        # via change_box, so zhi is just the topmost atom z).
        at = read(str(stack.data_path), format="lammps-data")
        cell = np.array(at.cell)
        xlo, xhi = 0.0, float(cell[0, 0])
        ylo, yhi = 0.0, float(cell[1, 1])
        zhi = float(at.get_positions()[:, 2].max())

        settings = self.config.settings
        sim = settings.simulation
        out = settings.output
        general = self.config.general

        atom_style = _detect_atom_style(stack.data_path)  # 'atomic' for sw
        lat_c = stack.interface_spacing or settings.geometry.lat_c_default

        # Same key set as SheetOnSheetSimulation.write_inputs' base_context (only
        # the keys the template actually references), with hetero-specific values
        # and RUN-DIR-RELATIVE paths.
        context = {
            "temp": general.temp,
            "xlo": xlo,
            "xhi": xhi,
            "ylo": ylo,
            "yhi": yhi,
            "zhi": zhi,
            "data_file": "hetero.data",
            "potential_file": "lammps/system.in.settings",
            "num_atom_types": stack.total_types,
            "ngroups": stack.total_types,
            "n_layers": n,
            "constraint_mode": "none",
            "n_bond_types": 0,
            "atom_style": atom_style,
            "pot_type": "sw",
            "has_internal_lj": True,  # hybrid sw -> reaction-force friction proxy
            # Default an unset load to 0.0 (zero-load slide). A raw None renders
            # ``variable pressure equal None`` -> LAMMPS "Invalid thermo keyword
            # 'None' in variable formula", breaking every default-config run.
            # Mirrors the homogeneous loop-path fallback ``pressures or [0.0]``
            # (src/builders/sheetonsheet.py) which never reaches the template
            # with a bare None.
            "pressures": general.pressure if general.pressure is not None else 0.0,
            "scan_speed_config": general.scan_speed,
            "scan_angle_config": general.scan_angle,
            "scan_angle_force": general.scan_angle_force,
            "drive_method": sim.drive_method,
            "thermostat_type": settings.thermostat.type,
            "timestep": sim.timestep,
            "thermo": sim.thermo,
            "neighbor_list": sim.neighbor_list,
            "neigh_modify_command": sim.neigh_modify_command,
            "run_steps": sim.slide_run_steps,
            "min_style": sim.min_style,
            "minimization_command": sim.minimization_command,
            "results_freq": out.results_frequency,
            "dump_freq": out.dump_frequency.get("slide", 1000),
            "dump_enabled": out.dump.get("slide", False),
            "results_file_pattern": (
                "results/friction_p${pressure}_a${a}_s${speed}"
            ),
            "dump_file_pattern": (
                "visuals/slide_p${pressure}_a${a}_s${speed}.lammpstrj"
            ),
            "driving_spring_ev": (general.driving_spring or 50.0) / NM_TO_EV_A2,
            "bond_spring_ev": (general.bond_spring or 80.0) / NM_TO_EV_A2,
            "lat_c": lat_c,
            "ev_a_to_nn": EV_A_TO_NN,
            "ev_a3_to_gpa": EV_A3_TO_GPA,
            "layers": bands,
        }

        script = self.render_template("hetero/slide.lmp", context)
        self.write_file("lammps/slide.in", script)
        logger.info("Wrote hetero slide script to %s/lammps/slide.in", self.output_dir)

        # Reuse the homogeneous HPC scaffolding, but do NOT let it block the
        # slide.in write. It also assumes the sim-root-relative path convention;
        # the hetero slide is run-dir-relative, so any script it emits needs a
        # path-convention follow-up (Task 5) rather than being forced here.
        try:
            self._generate_hpc_scripts()
        except Exception as exc:  # pragma: no cover - defensive, non-fatal
            logger.warning(
                "HPC script generation skipped for hetero slide (%s). The "
                "homogeneous HPC path assumes sim-root-relative paths while the "
                "hetero slide is run-dir-relative; deferring to Task 5.", exc,
            )
