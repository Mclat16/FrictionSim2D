# Potential-Energy-Surface (PES) scans

PES scans are cheap, **static** energy-vs-lateral-position maps that measure the
**corrugation energy barrier** of an interface — the descriptor the friction
literature (Prandtl–Tomlinson: friction ∝ barrier height *U*) relies on. They
reuse the existing sliding-simulation geometry but replace the dynamic sliding
run with an N×N lateral grid scan over one surface unit cell. Each grid point is
a single energy minimization, so a whole scan is seconds–minutes per material —
orders of magnitude cheaper than a sliding run.

There are two flavours:

| Scan | Command | System | Records |
|---|---|---|---|
| **Interlayer** | `run pes-sheet` | frozen bilayer (no tip/substrate) | energy per surface cell |
| **Tip surface** | `run pes-tip` | AFM tip + sheet + substrate | energy + lateral force on tip |

## How it works

**Sheet (`run pes-sheet`).** A two-layer stack is built at its equilibrium
interlayer spacing. The bottom layer is frozen (`setforce 0 0 0`); the top layer
is held rigid in-plane and grid-scanned laterally over `[0, a) × [0, b)` (the
orthogonalized surface unit cell). At each point the out-of-plane (z) coordinate
relaxes (`z_relax = true`, the relaxed PES) or the layers stay rigid
(`z_relax = false`, the bare PES), and the total energy — normalized per surface
unit cell — is written to `results/pes_scan.csv` (`dx, dy, energy_eV`).

**Tip (`run pes-tip`).** Self-contained (no `system.in` indentation): the scan
assembles the tip + sheet + substrate, places the rigid tip at a fixed firm
contact gap above the sheet (stable — the dynamic force-ramp indentation loses
atoms), then holds it at each grid point and records the total energy and lateral
force on the tip to `results/pes_scan.csv` (`x, y, energy_eV, fx, fy`).

A note on **contact load**: the deployable descriptor is read at a *fixed firm
contact*, not a normal-load setpoint. A true repulsive-load contact isn't
well-defined here — a compliant monolayer conforms to a pressed tip, so the
normal force stays attractive and the forced static contact is essentially
load-independent. The "2 nN pathology" (where the real sliding sims are
near-random because the tip won't hold contact) lives in the **friction target**,
not this scan: at 2 nN the COF has `cof_std ≈ 0.8` and masks *any* descriptor's
predictive value; by 10–30 nN it de-noises to ~0.01 and the descriptors' Spearman
ρ with COF climbs from ~0 to 0.2–0.5 (see
`ml_pes_load_pathology_figure.py` in the ML-analysis project). `[pes] tip_load`
records the target regime (default 30 nN, chosen data-drivenly as the cof_std
plateau / densest-sampled load) for provenance.

**Tip on a bilayer.** The tip scan runs on whatever sheet `[2D] layers` builds,
so `layers = [2]` presses the tip on a **bilayer** with no code change — the AFM
stack already splits per-layer types and adds the interlayer LJ for a multi-layer
SW sheet. This directly targets the monolayer pathology above: a single flexible
sheet *conforms* to the pressed tip and flattens the felt corrugation, so
`pes_fmax` collapses toward the noise floor (in the 156-material sweep, ~11
compliant TMDs — the Co/Cr/Fe/Ni/Sc dichalcogenides — had a dead, sub-1%-of-max
tip force). A bilayer is held rigid by the interlayer LJ, so the tip retains a
resolvable signal: every one of those 11 materials comes back to life
(`pes_fmax` 0.33–1.22 nN) and the descriptor spread roughly doubles. The bilayer
descriptor is only ρ≈0.47–0.56 rank-correlated with the monolayer one, so it
carries **orthogonal** information rather than replacing it. Against the
*layer-matched* friction target (monolayer scan ↔ layer-1 sliding, bilayer scan
↔ layer-2 sliding) its univariate predictive value is a wash — the bilayer edges
ahead at 5–20 nN, the monolayer overtakes at 30–70 nN (bilayer friction adds an
interlayer-shear channel the tip PES can't see). Keep both as feature sets and
let the multivariate `+pes_scan` rung arbitrate. See
`ml_pes_bilayer_vs_monolayer.py` (ML-analysis project). Run it as its own
campaign with `[2D] layers = [2]` (e.g. `examples/pes_tip_config.ini` with
`layers = [2]`); descriptors extract identically (the `L2/` run dir is handled).

### Freestanding tip scan (substrate-free)

Omitting the `[sub]` section from a `pes-tip` config selects the **freestanding**
builder: a self-supported N-layer slab of the material itself (`[2D] layers = [4]`,
`layer_1` held fixed, upper layers relaxed) replaces the amorphous substrate,
removing the disordered sheet-substrate registry that otherwise contributes
82-99% of the tip-PES energy noise. Each grid point is evaluated by a static
`minimize` (`[pes] eval_mode = minimize`, cheapest) or a short finite-T MD run
with the `layer_2` Langevin band (`eval_mode = md`, a thermalized PES). Descriptor
extraction is unchanged. Example: `examples/afm_freestanding_pes_config.ini`.

Both write the scan as the production-run script `lammps/slide.in`, so the
existing HPC array/combined job generation applies unchanged. Each grid point
uses a **force-converged** minimization (`minimize 0.0 1e-6 …`); the sliding
default's relative energy tolerance is far too coarse to resolve a sub-meV
corrugation. The tip scan defaults to a fast **rigid** grid (`z_relax = false`).

## Configuration

Add a `[pes]` section to an otherwise ordinary sheet-on-sheet / AFM config:

```ini
[pes]
grid_n = 12          # N for the N×N lateral grid over one surface unit cell
grid_n_refine = 20   # optional finer grid (emitted as slide_refine.in) for a convergence check
z_relax = true       # sheet scan: relax the top layer's z at each grid point
```

- **Sheet:** requires `[2D] layers = [2]`; keep `x, y` larger than twice the LJ
  cutoff (~24 Å) to avoid periodic self-interaction. A few unit cells is enough.
- **Tip:** set `[general] force` to a single low load (e.g. `2` nN); the
  corrugation is clearest before ploughing sets in.

Runnable examples: [../examples/pes_sheet_config.ini](../examples/pes_sheet_config.ini),
[../examples/pes_tip_config.ini](../examples/pes_tip_config.ini),
[../examples/afm_freestanding_pes_config.ini](../examples/afm_freestanding_pes_config.ini) (freestanding tip PES).

```bash
FrictionSim2D run pes-sheet examples/pes_sheet_config.ini -o ./pes_output --hpc-scripts
FrictionSim2D run pes-tip   examples/pes_tip_config.ini   -o ./pes_output --hpc-scripts
```

## Descriptor extraction

After the scans run, reduce the grids to per-material descriptors:

```bash
FrictionSim2D postprocess pes-scan ./pes_output/simulation_YYYYMMDD_HHMMSS \
    --data-dir data --grid-dir outputs/pes_scan
```

This writes `data/pes_scan_sheet.csv` and `data/pes_scan_tip.csv` (one row per
material, merge key `material`) and, with `--grid-dir`, the tidied grids under
`outputs/pes_scan/{sheet,tip}/<material>.csv`. Descriptors:

| Descriptor | Definition |
|---|---|
| `pes_barrier_U` | `max(E) − min(E)` — the Prandtl–Tomlinson corrugation barrier |
| `pes_barrier_mep` | easiest straight-channel barrier — the easy-sliding barrier |
| `pes_corr_rms` | RMS of `E − mean(E)` — overall corrugation strength |
| `pes_barrier_aniso` | barrier along **a** vs **b** — frictional anisotropy |
| `pes_curv_min` | curvature of `E` at the minimum — lateral stiffness *K* |
| `pes_eta` | `2π²·U / (K·a²)` — dimensionless PT friction parameter |
| `pes_fmax` (tip) | `max‖(fx, fy)‖` — peak lateral (static-friction) force |

The descriptor maths lives in [`src/postprocessing/pes_scan.py`](../src/postprocessing/pes_scan.py).
The ML pipeline consumes these exactly like `sw_features.csv`; a thin
`ml_extract_pes_scan.py` (in the ML-analysis project) calls the same module and
adds a `+pes_scan` feature rung.
