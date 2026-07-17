"""Freestanding (substrate-free) tip-on-slab simulation builders.

A self-supported multilayer slab of the sheet material replaces the amorphous
substrate: the bottom layer (``layer_1``) is held fixed and the upper layers
relax, so the tip-felt PES corrugation is not contaminated by a disordered
sheet-substrate registry. Selected automatically when a ``pes-tip`` config has
no ``[sub]`` section (see ``core.run``). Reuses the entire AFM build via the
substrate-optional path in :class:`~src.builders.afm.AFMSimulation`.
"""

import logging

from ..data.models import EV_A_TO_NN
from .pes_scan import PESTipSimulation, PES_TIP_CONTACT_GAP, PES_MINIMIZE_COMMAND, _cell_period

logger = logging.getLogger(__name__)


class FreestandingPESTipSimulation(PESTipSimulation):
    """Tip PES on a self-supported multilayer slab (no substrate)."""

    def _get_hpc_job_name(self) -> str:
        return f"pes_tip_free_{self.config.sheet.mat}"

    def _eval_mode(self) -> str:
        pes = getattr(self.config, 'pes', None)
        return getattr(pes, 'eval_mode', 'minimize') if pes is not None else 'minimize'

    def _md_steps(self) -> int:
        pes = getattr(self.config, 'pes', None)
        return int(getattr(pes, 'md_steps', 2000)) if pes is not None else 2000

    def write_inputs(self, n_layers: int) -> None:
        """Render the freestanding tip PES scan to ``lammps/slide.in``."""
        logger.info("Writing freestanding tip PES scan inputs (eval_mode=%s)...", self._eval_mode())

        if n_layers < 2:
            raise ValueError(
                "A freestanding tip PES needs a fixed bottom layer + at least one "
                f"relaxed layer; set [2D] layers >= 2 (got {n_layers})."
            )

        context = self.build_render_context(n_layers)
        rel_layer = str(self.relative_run_dir_layer[n_layers])
        m = self._scan_cells()
        period_x = _cell_period(self.sheet_unit_cell, "x")
        period_y = _cell_period(self.sheet_unit_cell, "y")
        self._warn_if_footprint_too_small(m, period_x, period_y)
        context.update({
            'grid_n': self._grid_n() * m,
            'period_x': period_x * m,
            'period_y': period_y * m,
            'contact_gap': PES_TIP_CONTACT_GAP,
            'results_csv': f"{rel_layer}/results/pes_scan.csv",
            'ev_a_to_nn': EV_A_TO_NN,
            'minimization_command': PES_MINIMIZE_COMMAND,
            'eval_mode': self._eval_mode(),
            'md_steps': self._md_steps(),
        })

        out_dir = self.output_dir_layer[n_layers]
        scan_script = self.render_template("afm_freestanding/pes_scan.lmp", context)
        self.write_file("lammps/slide.in", scan_script, out_dir)
        logger.info("Freestanding tip PES scan written to %s/lammps/", out_dir)
