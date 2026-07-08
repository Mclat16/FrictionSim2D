# Indentation-poke simulation recipe (for the simulation-building agent)

**Audience:** an agent with LAMMPS access and the existing AFM simulation infrastructure
(FrictionSim2D `run afm`).
**Deliverable:** one cheap **quasi-static indentation ("poke")** set (all materials, fresh
seeds) + one optional **adhesion / pull-off** set, in the campaign output schema, so the ML
side can score the deployable *contact-mechanics* number (`ml_verify_poke.py`).

This is the **cheapest sim in the ladder** — the equilibration stub of the probe recipe
(`docs/PROBE_SIM_RECIPE.md`) **minus the 50 ps slide**. It targets a different physical axis
than the PES scans: **out-of-plane compliance/adhesion**, not lateral corrugation.

---

## 1. Why (read this first)

Structure-only friction ranking is capped at Spearman ~0.574 and every *static lateral*
descriptor is a measured null — the tip PES on a monolayer, on a bilayer, and with the full
`fx,fy` force field mined (`docs/PES_SCAN_RESULTS.md`, `docs/PES_BILAYER_RESULTS.md`): all
sit inside the crystallography CI because lateral corrugation is collinear with the lattice.

The **contact-mechanics** channel is the exception. The offline study
(`docs/PROBE_RESULTS.md`, Track B) shows a **no-sliding** contact descriptor recovers most of
the post-sim signal:

- structure + 2-point contact features (penetration depth + stiffness from loads {10,30})
  score **0.679** [0.657, 0.701] — vs the 0.574 floor and the 0.706 post-sim ceiling, i.e.
  ~75 % of the entire floor→ceiling gap, with **no sliding at all**;
- **penetration depth** (tip_z relative to sheet COM) is the single strongest
  non-crystallography, non-COF feature: univariate ρ = **−0.51** (deeper, softer, more
  compliant contact → lower friction);
- the 2-point variant already matches the 9-point sweep ceiling (0.679 vs 0.678) — the signal
  is in the penetration **depth**, not the (noisy) stiffness slope.

**But that 0.679 was extracted from the early frames of the existing full campaign runs**
(`data/contact_features.csv` ← `probe_features_f100.csv`), so it shares the campaign
trajectory. It is a proven *signal*, not yet a proven *cheap sim*. This recipe produces the
independent, fresh-contact runs needed to **bank the number** — at a fraction of even the
probe's cost, because there is no slide.

Why it works where the lateral PES fails: pressing the tip in measures the **vertical
compliance/adhesion** of the contact, which is orthogonal to both the lateral corrugation the
PES sees and to crystallography ("this is exactly what the static PES could not see — no load
response").

## 2. Reuse the existing infrastructure — do NOT rebuild geometry

This is the standard AFM sim (`FrictionSim2D run afm`: tip + sheet + substrate, indentation
ramp via `system_init.lmp`), run **only up to and including the load hold, then stopped** —
`slide.lmp` is never invoked. Same materials (156 CIF/SW files), same Si r25 tip, same
substrate, same K300 spring, same aveforce indentation ramp, same output cadence. **Fresh
velocity seeds.**

## 3. Poke set A — quasi-static indentation curves (REQUIRED)

Per material: **layers {1, 2} × loads {10, 30} nN (required) × angle 0 × fresh seed**.
Because a poke is so cheap (no slide), **also run loads {5, 20, 50} if feasible** — it costs
almost nothing extra and delivers the 9-point stiffness ceiling *and* lets the ML side test a
load-resolved penetration descriptor (the open "does load-resolution help?" question).

| parameter | value | why |
|---|---|---|
| procedure | build + indent via the existing `system_init.lmp` aveforce ramp to the target normal load, **then a short damped hold** — no sliding | the hold gives the thermalized means the extractor needs |
| hold length | **10 000 steps = 10 ps** at the target load after the ramp | quasi-static settle; yields `pen_mean`/`pen_std`/`comz_mean`. Total poke ≈ 30k steps (20k ramp + 10k hold) |
| loads | **10 and 30 nN required**; {5, 20, 50} recommended | {10,30} is the deployable 2-point anchor the ML layout was validated on; the wider set gives the 9-point ceiling + load-resolution |
| angle | 0 | matches the deployable-unit labels |
| seeds | **fresh, K = 1** (state seeds in the manifest) | one poke suffices — penetration depth is thermally stable over a 10 ps hold; averaging is already inside the window |
| sampling cadence | **1 ps/sample (every 1000 steps)** — identical to campaign | the extractors run unchanged; a 10 ps hold gives ~10 samples/condition |

**MANDATORY: fresh velocity seed for every poke.** Re-using a campaign seed reproduces the
campaign contact trajectory and silently re-imports the same-run optimism the verification is
built to exclude. A fresh-seed penetration that matches the campaign value **exactly** (ρ ≈ 1)
is a red flag that seeds were not fresh.

**Output contract (exact):** same nested-JSON schema as the campaign
(`results → material → substrate → tip_mat → r25 → l{layer} → s{speed} → f{load} → a0`, each
leaf with `columns` including **`nf, comz, tipz`** — the out-of-plane channels the penetration
descriptor is built from; `lfx,lfy` welcome but unused here — and `data` = the per-ps hold
rows). Write one file per seed:
`outputs/poke_sims/output_poke_f{load}_s{seed}.json` + a manifest CSV
(`material, layer, load, seed, steps, wall_time`). The ML side then runs a small
`ml_extract_poke_features.py` that reduces each leaf **exactly as
`ml_extract_probe_features.py`** does (so `pen_mean = mean(tip_z − com_z)` etc. are identical
in definition), and reuses `ml_extract_contact_features.py::contact_row()` to form
`ct_pen10/ct_pen30/ct_stiff2_*` (and `ct_*9_*` if the wider sweep is present).

**Cost:** 4 pokes/material (K=1, loads {10,30}) × ~30k steps = 120k steps ≈ **0.23 full-run
equivalents per material ≈ 0.4 %** of that material's 59-condition campaign — roughly **half
the probe's cost** (no slide). The wider {5,10,20,30,50} sweep is ~10 pokes ≈ 1 % — still
negligible.

## 4. Poke set B — adhesion / pull-off retract curves (OPTIONAL, hypothesis test)

Identical to `docs/PROBE_SIM_RECIPE.md` §4 — the one contact quantity no existing data holds.
Per material × layer, at angle 0:

1. Build + indent exactly as `system_init.lmp` (aveforce ramp) to ~+10 nN.
2. **Retract** the rigid tip quasi-statically (0.1–0.2 Å z-steps; minimize or short relax at
   each step) through zero load, past the pull-off instability, to full detachment.
3. Record per step: `z_tip, fz_tip, pe` (+ branch label approach/retract if both are run).

Output: `outputs/poke_sims/adhesion/<material>_l{layer}.csv`. The ML side reduces it
(`ml_extract_adhesion.py`, to be written on delivery) to `pulloff_force`, `work_of_adhesion`
(retract-branch integral), `adh_hysteresis`, `contact_stiffness` (repulsive-wall slope),
`snapin_dist` → `data/adhesion_features.csv` keyed (material, layer).

Prior: **moderate** — adhesion was co-dominant in the interlayer model and nothing
adhesion-like exists for the tip, but the PES precedent shows collinearity can null a
well-motivated descriptor. Cost is minimization-scale (PES-scan class). Acceptance:
`contact_stiffness` should rank-correlate with the sliding-derived 9-point stiffness at
ρ > 0.6 (`data/contact_features.csv::ct_stiff9_slope`).

## 5. Acceptance / sanity checks (poke set A)

- **Coverage:** ≥ 150/156 materials × 2 layers × {10,30} nN, angle 0, fresh seeds in manifest.
- **Schema:** the poke JSON parses with the campaign reader unchanged; ~10 samples/condition,
  each carrying `nf, comz, tipz`.
- **Feature consistency (the key check):** fresh-seed penetration at (loads 10/30, angle 0)
  must rank-correlate with the campaign-derived `ct_pen10`/`ct_pen30`
  (`data/contact_features.csv`) within the replicate bracket — Spearman ~**0.6–0.9**. ≪ 0.6 ⇒
  suspect build/contact handling; ≈ 1.0 exactly ⇒ seeds were NOT fresh (hard fail).
- **Load monotonicity:** penetration must deepen monotonically with load per (material, layer)
  — a non-monotone curve signals an unconverged hold or contact loss.
- **Physics anchor:** compliant / puckered materials (e.g. `p_SnSe` family) penetrate deepest
  and should sit in the low-COF tail; stiff oxides penetrate least.

## 6. Hand-back

Deliver: `outputs/poke_sims/output_poke_f*_s*.json` (+ manifest), optional
`outputs/poke_sims/adhesion/*.csv`. The ML side then writes `ml_extract_poke_features.py`
(→ fresh `data/poke_features.csv`, `data/contact_features_fresh.csv`) and runs
`ml_verify_poke.py` under the locked honest protocol (`ml_eval_protocol.py`: repeated grouped
CV 5×5, nested top-20, mean per-load Spearman ± 95 % CI, angle-0 unit, 5–50 nN).

**Pre-registered gate:** success = mean per-load Spearman **≥ 0.63 with ci95_low ≥ 0.596**
(clears the crystallography floor *and* the +SW+symmetry rung), quoted against the offline
0.679 contact result, the 0.706 post-sim ceiling, and the label-reliability bracket. Offline
forecast after fresh-seed erosion: **0.65–0.68**. The final number and the combined
**sheet-PES ⊥ contact** cheap-stack rung go into a new section of `docs/PROBE_RESULTS.md`.
