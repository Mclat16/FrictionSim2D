# AFM indentation ("indent") simulations

An **indent** run measures both vertical-contact quantities the lateral
[PES scans](pes_scan.md) cannot see, from a single run per `(material, layer,
load)`. It is the standard AFM simulation (tip + sheet + substrate, indentation
ramp via `system.in`) run **only up to the load, then held and retracted** — the
sliding phase (`slide.in`) is never invoked. Per load it does two phases back to
back:

1. **Hold (penetration).** After the aveforce ramp presses the tip to the target
   load, a short **damped finite-T hold** records the out-of-plane channels; the
   thermally-averaged **penetration depth** (`tip_z − sheet_com_z`) is the
   vertical-compliance descriptor (deeper, softer contact → lower friction).
2. **Retract (adhesion).** The MD fixes are torn down, the rigid tip is frozen and
   **retracted quasi-statically** (T=0 minimization) in small z-steps to full
   detachment, tracing the force–distance curve for the **pull-off force**, **work
   of adhesion** and **contact stiffness** — the one contact quantity no sliding
   or hold data holds.

This one type subsumes the former separate **poke** (hold-only) and **adhesion**
(retract-only) simulations: because the retract starts from the held contact, both
descriptors come out of one job. See `POKE_SIM_RECIPE.md`.

## How it works

The indent reuses the **full AFM build and the `system.in` indentation unchanged**:
`system.in` builds the stack, minimises, and runs the aveforce ramp to each target
load, writing one `load_<f>N.data` per load. The run phase then swaps the sliding
script for `afm/indent.lmp` (emitted as `slide.in`), which per load:

**Phase 1 — hold:**
1. reads `load_<f>N.data` (the loaded contact state),
2. re-applies the target normal load via `aveforce` and holds the **rigid,
   laterally-fixed** tip (free only in *z* — a pure vertical hold, no slide),
3. records `nf`, `comz` (sheet COM *z*) and `tipz` (tip COM *z*) via `fix
   ave/time` at the 1 ps campaign cadence for a `hold_steps` (default 10 000 =
   10 ps) damped hold — same `results/friction_f<load>_a0_s<speed>_layer<N>.txt`
   10-column layout as the sliding campaign, so the campaign reader
   ([`read_data.py`](../src/postprocessing/read_data.py)) parses it **unchanged**.

**Phase 2 — retract:**
4. unfixes every dynamics fix from the hold (langevin, aveforce, viscous drag,
   nve, the rigid tip) so the minimization sees only conservative forces,
5. **freezes the rigid tip** (`tip_all`) with `setforce` and force-converged
   `minimize`s the held contact to a clean reference,
6. repeats `n_steps` times: `displace_atoms` the tip up by `z_step`,
   force-converged `minimize`, record `z_tip, fz_tip, fx/fy_tip, pe` — tracing the
   pull-off force–distance curve to `results/indent_f<load>N_l<layer>.csv`.

Everything downstream (two-phase / combined HPC job generation, `system.in →
slide.in` sequencing) applies with no change.

> **Two ensembles, by design.** The hold is finite-T damped MD (the penetration is
> a thermal average); the retract is T=0 minimization (a quasi-static force–distance
> curve). Phase 2 tears down the thermostat/aveforce/viscous fixes before minimising
> so langevin random forces cannot corrupt the sub-nN pull-off signal.

### Fresh velocity seeds

A **fresh velocity seed for every run is mandatory** — reusing a campaign seed
reproduces the campaign contact trajectory and silently re-imports the same-run
optimism the verification excludes. A single campaign base seed (`[indent] seed`)
fans out to a distinct, reproducible seed per `(material, layer)` (SHA-256 of
`base:material:layers`); each is injected into that run's `system.in` `velocity
create` and recorded in the manifest. Omit `[indent] seed` to draw a fresh base
seed at build time.

## Configuration

Add an `[indent]` section to an otherwise ordinary AFM config. Loads come from
`[general] force`, layers from `[2D] layers`, angle from `[general] scan_angle`
(0):

```ini
[general]
force = [5, 10, 20, 30, 50]   # {10,30} required; {5,20,50} give the 9-point ceiling
scan_angle = 0                # matches the deployable-unit labels
outer_loop = force            # one HPC job per load (all share the system.in ramp)

[tip]
dspring =                     # blank -> auto viscous damping (the tip settles in the hold)

[indent]
hold_steps = 10000            # 10 ps damped hold at the target load (~10 samples/condition)
z_step = 0.2                  # retract increment per step (Å); recipe: 0.1–0.2 Å
n_steps = 100                 # retract steps; n_steps × z_step = 20 Å total
seed = 20260710               # campaign base seed (fresh per-run seeds derived from it)
```

- Loads `{10, 30}` nN are the required deployable 2-point anchor; adding
  `{5, 20, 50}` delivers the 9-point stiffness ceiling and load-resolved
  penetration **and** pull-off for almost nothing extra (one hold+retract per load).
- Leave `[tip] dspring` blank so the tip's viscous damping is auto-set to a
  fraction of critical and it settles quasi-statically within the hold.
- `n_steps × z_step` (default 20 Å) is the total retract distance — it must
  comfortably exceed the adhesive tail so the curve reaches full detachment. The
  postprocessor warns if a curve's `fz_tip` minimum sits at the last steps.

Runnable example: [../examples/indent_config.ini](../examples/indent_config.ini).

```bash
FrictionSim2D run indent examples/indent_config.ini -o ./indent_output --hpc-scripts
```

## Delivery / postprocessing

After the runs finish, reduce them to both descriptor sets in one pass:

```bash
FrictionSim2D postprocess indent ./indent_output/simulation_YYYYMMDD_HHMMSS \
    --out-dir outputs/poke_sims --data-dir data
```

This writes:

| File | Contents |
|---|---|
| `outputs/poke_sims/output_indent_f<load>_s<seed>.json` | **penetration** — one per (load, seed batch), nested campaign schema `results → material → size → substrate → tip_mat → r25 → l{layer} → s{speed} → f{load} → a0`, each leaf `{columns, data}` (columns include `nf, comz, tipz`) |
| `data/indent_features.csv` | **adhesion** — one row per `(material, layer, load)`, merge key `material`, with `pulloff_force`, `work_of_adhesion`, `contact_stiffness` (+ `f_adh_min`, `snapoff_z`, `work_of_adhesion_nnA`, `detached`) |
| `outputs/poke_sims/indent/<material>_l<layer>.csv` | the tidied per-condition retract curve (`step, z_tip, fz_tip, fx_tip, fy_tip, pe`) |
| `outputs/poke_sims/indent_manifest.csv` | one row per condition with the fresh `seed`, `hold_steps`, `z_step`, `n_steps` and best-effort `wall_time` |

Reduced adhesion descriptors (`fz_tip` is + repulsive / − adhesive; `z_tip`
increases as the tip retracts):

| descriptor | definition |
|---|---|
| `pulloff_force` | `−min(fz_tip)` — the peak tensile (adhesive) force, nN |
| `work_of_adhesion` | `∫` over the attractive branch of `(−fz_tip) dz` — energy to separate, eV |
| `contact_stiffness` | slope of the leading repulsive wall of `fz_tip` vs `z_tip`, nN/Å |
| `snapoff_z` | tip COM z at the pull-off minimum, Å |

The penetration JSON feeds the ML side's penetration feature reducer exactly as the
sliding campaign (`pen_mean = mean(tip_z − com_z)`).

> **Snap-in distance and approach/retract hysteresis** need the *approach* branch,
> which this retract does not record (the approach is the `system.in` aveforce
> ramp, which stores no per-step curve). They are left to a future approach+retract
> variant — recipe §4's "+ branch label approach/retract if both are run".

The delivery maths lives in [`src/postprocessing/indent.py`](../src/postprocessing/indent.py);
the builder in [`src/builders/indent.py`](../src/builders/indent.py); the run-phase
template in [`src/templates/afm/indent.lmp`](../src/templates/afm/indent.lmp).

## Sanity checks

- **Feature consistency (penetration):** fresh-seed penetration at (loads 10/30,
  angle 0) must rank-correlate with the campaign-derived `ct_pen10`/`ct_pen30` at
  Spearman ~0.6–0.9. `≪ 0.6` ⇒ suspect build/contact handling; `≈ 1.0` exactly ⇒
  seeds were **not** fresh (hard fail).
- **Load monotonicity:** penetration must deepen monotonically with load per
  `(material, layer)`; a non-monotone curve signals an unconverged hold or contact
  loss.
- **Contact-stiffness anchor (acceptance):** `contact_stiffness` should
  rank-correlate with the sliding-derived 9-point stiffness
  (`data/contact_features.csv::ct_stiff9_slope`) at Spearman ρ > 0.6.
- **Detachment:** every retract curve must reach detachment (`detached = 1`); the
  postprocessor warns when `fz_tip`'s minimum is at the last steps — raise
  `[indent] n_steps` (or `z_step`) if so.
- **Physics anchor:** compliant/puckered materials penetrate deepest, show the
  deepest adhesive wells and largest work of adhesion (low-COF tail); stiff oxides
  penetrate least and stick least.
