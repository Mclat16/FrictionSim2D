"""AFM indentation ("indent") builder: damped hold (penetration) + retract (pull-off).

Merges the two vertical-contact measurements the lateral PES scan cannot see into
one run per ``(material, layer, load)``. It reuses the full AFM build and the
``system.in`` indentation ramp unchanged — pressing the tip to each normal load in
``[general] force`` and writing ``load_<f>N.data`` — then swaps the run-phase
template for ``afm/indent.lmp`` (emitted as ``slide.in``), which:

    1. re-applies the target load and holds the (rigid, laterally-fixed) tip for a
       short *damped* finite-T hold, recording the out-of-plane channels — the
       thermally-averaged penetration depth (``tip_z − sheet_com_z``) is the
       vertical-compliance descriptor (same 10-column layout as the sliding
       campaign, so :class:`~src.postprocessing.read_data.DataReader` parses it
       unchanged); then
    2. tears down the MD fixes, freezes the rigid tip and **retracts it
       quasi-statically** (T=0 minimization) in small z-steps to full detachment,
       writing a per-``(material, layer, load)`` force–distance CSV — the pull-off
       force, work of adhesion and contact stiffness no sliding/hold data holds.

The sliding phase (``afm/slide.lmp``) is never invoked. Like the former poke and
adhesion builders it subsumes, it reuses the identical AFM build + ``system.in``
indentation and injects a recorded **fresh** velocity seed per ``(material,
layer)``, so all existing two-phase / combined HPC job generation applies
unchanged. See :mod:`src.postprocessing.indent`, ``documentation/indent.md`` and
``POKE_SIM_RECIPE.md``.
"""

import hashlib
import json
import logging
import random
from typing import Dict, List, Optional

from ..core.config import AFMSimulationConfig
from ..data.models import EV_A_TO_NN
from .afm import AFMSimulation

logger = logging.getLogger(__name__)

#: Default damped-hold length (steps) when no ``[indent] hold_steps`` is given.
#: 10 000 steps = 10 ps at the 1 fs campaign timestep → ~10 samples/condition.
DEFAULT_HOLD_STEPS = 10000

#: Default retract increment (Å) and step count when no ``[indent]`` is given.
#: 100 × 0.2 Å = 20 Å total retract — comfortably past the LJ adhesive tail.
DEFAULT_Z_STEP = 0.2
DEFAULT_N_STEPS = 100

#: Force-converged minimization for each retract step. The sliding-sim default
#: (``minimize 1e-4 1e-8 …``) stops on a *relative* energy tolerance that, on a
#: many-hundred-eV system, quits after 1–2 iterations and buries the sub-nN
#: pull-off signal in noise. etol=0 forces convergence on an absolute force
#: tolerance (size-independent), giving a clean force–distance curve — the same
#: reasoning as the PES scan (see :mod:`src.builders.pes_scan`).
INDENT_MINIMIZE_COMMAND = "minimize 0.0 1.0e-6 10000 100000"

#: Modulus for deriving per-run velocity seeds (prime; keeps seeds in 1..this).
_SEED_MODULUS = 999983


def derive_seed(base_seed: int, material: str, n_layers: int) -> int:
    """Deterministically derive a fresh per-(material, layer) velocity seed.

    A single campaign base seed fans out to a distinct, reproducible seed for
    every run, so each starts from an independent fresh contact trajectory while
    the whole campaign stays reproducible and the seeds are recordable in the
    manifest. Uses SHA-256 (not the salted built-in ``hash``) for stability
    across processes/runs.
    """
    key = f"{base_seed}:{material}:{n_layers}".encode("utf-8")
    return int(hashlib.sha256(key).hexdigest(), 16) % _SEED_MODULUS + 1


class IndentSimulation(AFMSimulation):
    """AFM indentation builder: AFM build + system.in, then a hold + retract run.

    Reuses the full AFM build + ``system.in`` indentation (writing each
    ``load_<f>N.data``), then emits ``afm/indent.lmp`` as the run phase
    (``slide.in``) — a finite-T damped hold (penetration) followed by a
    displacement-controlled T=0 retract (pull-off), no sliding, writing both the
    10-column hold file and a force–distance CSV per ``(material, layer, load)``.
    """

    #: Swap the sliding run for the hold+retract; the build + system.in are reused.
    SLIDE_TEMPLATE = "afm/indent.lmp"

    def __init__(self, config: AFMSimulationConfig, output_dir: str,
                 config_path: Optional[str] = None):
        super().__init__(config, output_dir, config_path=config_path)
        ind = getattr(config, 'indent', None)
        self.hold_steps: int = int(ind.hold_steps) if ind is not None else DEFAULT_HOLD_STEPS
        self.z_step: float = float(ind.z_step) if ind is not None else DEFAULT_Z_STEP
        self.n_steps: int = int(ind.n_steps) if ind is not None else DEFAULT_N_STEPS
        # A fixed base seed (from [indent] seed) makes the whole campaign
        # reproducible and shares one seed batch across materials; when omitted a
        # fresh base seed is drawn once so re-runs stay fresh (recipe mandate).
        self.base_seed: int = (
            int(ind.seed) if ind is not None and ind.seed is not None
            else random.randint(1, 999999)
        )
        #: n_layers -> derived velocity seed (populated during build()).
        self.indent_seeds: Dict[int, int] = {}

    def _get_hpc_job_name(self) -> str:
        return f"indent_{self.config.sheet.mat}"

    def _velocity_seed(self, n_layers: int) -> int:
        """Fresh, recorded ``velocity create`` seed for this (material, layer)."""
        seed = derive_seed(self.base_seed, self.config.sheet.mat, n_layers)
        self.indent_seeds[n_layers] = seed
        return seed

    def build_render_context(self, n_layers: int) -> Dict[str, object]:
        context = super().build_render_context(n_layers)
        # Phase 1 (hold): the AFM base already supplies results_file_pattern (the
        # 10-column friction_* hold file) + damping/thermostat keys; we add the
        # hold length. Phase 2 (retract): the force–distance CSV path + retract
        # discretization; $(v_find) resolves to the peak load per (per-load) script.
        context['hold_steps'] = self.hold_steps
        rel_layer = self.relative_run_dir_layer[n_layers]
        context['results_csv'] = (
            f"{rel_layer}/results/indent_f$(v_find)N_l{n_layers}.csv"
        )
        context['z_step'] = self.z_step
        context['n_steps'] = self.n_steps
        # Force-converged minimize for the retract only; system.in keeps the
        # campaign's default minimizer so the loaded contact matches the campaign.
        context['retract_minimize'] = INDENT_MINIMIZE_COMMAND
        context['ev_a_to_nn'] = EV_A_TO_NN
        return context

    def _loads(self) -> List[float]:
        force = self.config.general.force
        if force is None:
            return []
        return [float(f) for f in force] if isinstance(force, list) else [float(force)]

    def build(self) -> None:
        super().build()
        self._write_indent_meta()

    def _write_indent_meta(self) -> None:
        """Record the campaign base seed + per-layer fresh seeds + run parameters.

        Written to ``provenance/indent_meta.json`` so
        :mod:`src.postprocessing.indent` can key the delivered manifest CSV
        (``material, layer, load, seed, hold_steps, z_step, n_steps, …``) and name
        each per-load penetration JSON by the base seed.
        """
        meta = {
            'base_seed': self.base_seed,
            'hold_steps': self.hold_steps,
            'z_step': self.z_step,
            'n_steps': self.n_steps,
            'material': self.config.sheet.mat,
            'layers': {str(n): seed for n, seed in sorted(self.indent_seeds.items())},
            'loads': self._loads(),
        }
        path = self.output_dir / 'provenance' / 'indent_meta.json'
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(meta, indent=2), encoding='utf-8')
        logger.info(
            "Wrote indent meta (base_seed=%d, hold_steps=%d, z_step=%.3f Å, "
            "n_steps=%d, %d layer seed(s)) -> %s",
            self.base_seed, self.hold_steps, self.z_step, self.n_steps,
            len(self.indent_seeds), path,
        )
