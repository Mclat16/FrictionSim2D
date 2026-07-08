"""Static potential-energy-surface (PES) scan builders.

These builders reuse the existing sliding-simulation geometry but replace the
dynamic sliding run with a cheap, *static* energy-vs-lateral-position map over
one surface unit cell — the corrugation-barrier descriptor the friction
literature relies on (Prandtl–Tomlinson). See ``PES_SCAN_RECIPE.md``.

Two flavours:

    - :class:`PESSheetSimulation` — the sheet-on-sheet interlayer PES. A frozen
      bilayer whose top layer is grid-scanned laterally, relaxing only the
      out-of-plane coordinate at each grid point.
    - :class:`PESTipSimulation` — the tip-on-sheet (AFM) surface PES. The AFM
      stack is built and indented to a low load exactly as the sliding sim, then
      the rigid tip is grid-scanned laterally while the sheet + substrate relax;
      the lateral force on the tip is recorded alongside the energy.

Both emit their scan as the run-phase script ``lammps/slide.in`` (the framework
production-run slot the recipe swaps the sliding out of), so all existing HPC
array/combined job generation applies unchanged. The scan writes its grid to
``results/pes_scan.csv``; :mod:`src.postprocessing.pes_scan` reduces the grids
to per-material descriptors.
"""

import logging
from typing import Dict, List, Optional, Union

from ..core.potential_manager import PotentialManager
from ..core.utils import normalize_potential_type
from ..data.models import EV_A_TO_NN
from .afm import AFMSimulation
from .sheetonsheet import SheetOnSheetSimulation

logger = logging.getLogger(__name__)

#: Default lateral grid resolution (N in the N×N scan) when a config omits it.
DEFAULT_GRID_N = 12

#: Force-converged minimization for each PES grid point. The sliding-sim default
#: (``minimize 1e-4 1e-8 …``) stops on a *relative* energy tolerance that, on a
#: many-hundred-eV system, is far coarser than the sub-meV corrugation we are
#: resolving — so it quits after 1–2 iterations and buries the signal in noise.
#: etol=0 forces convergence on an absolute force tolerance (size-independent),
#: giving clean per-point energies and lateral forces.
PES_MINIMIZE_COMMAND = "minimize 0.0 1.0e-6 10000 100000"

#: Fixed contact gap (Å) between the lowest tip atom and the highest sheet atom for
#: the tip scan. A firm static contact in the tip/sheet interaction well: a true
#: repulsive-load contact is not well-defined here (a compliant monolayer conforms
#: to a pressed tip, so the normal force stays attractive), and the forced static
#: contact is essentially load-independent, so a fixed firm gap is the clean choice.
PES_TIP_CONTACT_GAP = 3.5

#: The static PES bilayer needs exactly two layers: a frozen bottom and a rigid,
#: laterally-scanned top.
PES_SHEET_LAYERS = 2


def _as_float_list(value: Optional[Union[float, List[float]]]) -> List[float]:
    """Normalize a scalar/list sweep value to a list of floats."""
    if value is None:
        return []
    if isinstance(value, list):
        return [float(v) for v in value]
    return [float(value)]


def _cell_period(unit_cell: Dict[str, float], axis: str) -> float:
    """Return the orthogonalized surface-unit-cell period along ``axis`` (Å)."""
    lo = unit_cell.get(f"{axis}lo")
    hi = unit_cell.get(f"{axis}hi")
    if lo is None or hi is None:
        raise ValueError(
            "Surface unit-cell dimensions unavailable; cannot size the PES grid. "
            "Ensure build_sheet was called with unit_cell_out."
        )
    period = float(hi) - float(lo)
    if period <= 0.0:
        raise ValueError(f"Non-positive surface cell period along {axis}: {period}.")
    return period


def _tile_count(full_dims: Dict[str, float], unit_cell: Dict[str, float]) -> int:
    """Number of surface unit cells tiling the built sheet footprint (x·y)."""
    nx = max(1, round(
        (float(full_dims["xhi"]) - float(full_dims["xlo"])) / _cell_period(unit_cell, "x")
    ))
    ny = max(1, round(
        (float(full_dims["yhi"]) - float(full_dims["ylo"])) / _cell_period(unit_cell, "y")
    ))
    return int(nx * ny)


class PESSheetSimulation(SheetOnSheetSimulation):
    """Sheet-on-sheet interlayer PES scan (frozen bilayer, static lateral grid).

    Reuses the sheet-on-sheet geometry/potential setup with a two-layer stack and
    plain real interlayer LJ (no bonds/springs, no driving atom), then writes a
    static grid-scan script instead of the sliding one.
    """

    #: A bilayer is enough (and cheapest) for a static PES.
    MIN_LAYERS: int = PES_SHEET_LAYERS

    def build(self) -> None:
        if self.n_layers != PES_SHEET_LAYERS:
            raise ValueError(
                "PES sheet scan requires exactly 2 layers (a frozen bottom + a "
                f"scanned top); set [2D] layers = [{PES_SHEET_LAYERS}], got {self.n_layers}."
            )
        # A static PES has no interlayer bonds/springs — just the real LJ that
        # governs the corrugation. Force 'none' regardless of the run settings.
        self.config.settings.simulation.constraint_mode = 'none'
        super().build()

    def _register_drive_atom(self, pm: PotentialManager) -> None:
        """No driving atom in a static scan (keeps types aligned with the data)."""

    def _get_hpc_job_name(self) -> str:
        return f"pes_sheet_{self.config.sheet.mat}"

    def _grid_n(self) -> int:
        pes = getattr(self.config, 'pes', None)
        return int(pes.grid_n) if pes is not None else DEFAULT_GRID_N

    def _grid_variants(self) -> List[tuple]:
        """Return (script_name, csv_name, grid_n) pairs to emit.

        Always emits the base ``slide.in`` at ``grid_n``. When ``pes.grid_n_refine``
        is set, also emits ``slide_refine.in`` at the finer resolution so a
        grid-convergence check (recipe §6) runs in the same job array.
        """
        variants = [("lammps/slide.in", "pes_scan.csv", self._grid_n())]
        pes = getattr(self.config, 'pes', None)
        if pes is not None and pes.grid_n_refine:
            variants.append(
                ("lammps/slide_refine.in", "pes_scan_refine.csv", int(pes.grid_n_refine))
            )
        return variants

    def write_inputs(self) -> None:
        """Render the static interlayer PES scan script(s)."""
        logger.info("Writing sheet PES scan inputs...")

        assert self.pm is not None
        assert self.sheet_dims is not None

        pot_type = normalize_potential_type(self.config.sheet.pot_type)
        atom_style = 'charge' if pot_type in ('reaxff', 'reax/c') else 'atomic'

        sim = self.config.settings.simulation
        rel = str(self.relative_run_dir)
        pes = getattr(self.config, 'pes', None)
        z_relax = bool(pes.z_relax) if pes is not None else True

        period_x = _cell_period(self.sheet_unit_cell, "x")
        period_y = _cell_period(self.sheet_unit_cell, "y")
        n_cells = _tile_count(self.sheet_dims, self.sheet_unit_cell)

        base_context = {
            'atom_style': atom_style,
            'pot_type': pot_type,
            'neighbor_list': sim.neighbor_list,
            'neigh_modify_command': sim.neigh_modify_command,
            'data_file': f"{rel}/build/{self.structure_paths['sheet'].name}",
            'potential_file': f"{rel}/lammps/system.in.settings",
            'min_style': sim.min_style,
            'minimization_command': PES_MINIMIZE_COMMAND,
            'timestep': sim.timestep,
            'thermo': sim.thermo,
            'period_x': period_x,
            'period_y': period_y,
            'n_cells': n_cells,
            'z_relax': z_relax,
        }

        for script_name, csv_name, grid_n in self._grid_variants():
            context = dict(base_context)
            context['grid_n'] = grid_n
            context['results_csv'] = f"{rel}/results/{csv_name}"
            script = self.render_template("sheetonsheet/pes_scan.lmp", context)
            self.write_file(script_name, script)

        logger.info(
            "Sheet PES scan written to %s/lammps/ (grid periods %.3f×%.3f Å, %d cells)",
            self.output_dir, period_x, period_y, n_cells,
        )


class PESTipSimulation(AFMSimulation):
    """Tip-on-sheet (AFM) surface PES scan (rigid tip, static lateral grid).

    Reuses the full AFM build + indentation (``system.in`` → ``load_<f>N.data``),
    then writes a static grid-scan script as the run phase: the rigid tip is
    held at each grid point while the sheet + substrate relax, recording the
    total energy and the lateral force on the tip.
    """

    def _get_hpc_job_name(self) -> str:
        return f"pes_tip_{self.config.sheet.mat}"

    def _grid_n(self) -> int:
        pes = getattr(self.config, 'pes', None)
        return int(pes.grid_n) if pes is not None else DEFAULT_GRID_N

    def _z_relax(self) -> bool:
        pes = getattr(self.config, 'pes', None)
        return bool(pes.z_relax) if pes is not None else True

    def write_inputs(self, n_layers: int) -> None:
        """Render the self-contained static tip PES scan (single ``slide.in``).

        The scan assembles the tip + sheet + substrate itself and places the rigid
        tip at a fixed firm contact gap, so no separate ``system.in`` indentation
        phase is generated. A true repulsive normal-load contact is not well-defined
        for a compliant monolayer (it conforms to a pressed tip, so the normal force
        stays attractive), and the forced static contact is essentially
        load-independent — the deployable descriptor is this fixed firm contact. The
        2 nN pathology lives in the dynamic friction *target*, not this scan.
        """
        logger.info("Writing tip PES scan inputs...")

        context = self.build_render_context(n_layers)
        rel_layer = str(self.relative_run_dir_layer[n_layers])
        context.update({
            'grid_n': self._grid_n(),
            'period_x': _cell_period(self.sheet_unit_cell, "x"),
            'period_y': _cell_period(self.sheet_unit_cell, "y"),
            'z_relax': self._z_relax(),
            'contact_gap': PES_TIP_CONTACT_GAP,
            'results_csv': f"{rel_layer}/results/pes_scan.csv",
            'ev_a_to_nn': EV_A_TO_NN,
            # Force-converged minimize so per-point energies/forces are clean; the
            # sliding default's relative energy tol is too coarse for the PES.
            'minimization_command': PES_MINIMIZE_COMMAND,
        })

        out_dir = self.output_dir_layer[n_layers]
        scan_script = self.render_template("afm/pes_scan.lmp", context)
        self.write_file("lammps/slide.in", scan_script, out_dir)

        logger.info("Tip PES scan written to %s/lammps/", out_dir)
