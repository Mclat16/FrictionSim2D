# Large-scale campaigns (155-material set)

Two campaigns run over `examples/material_list.txt` (156 materials, each with an
in-repo `{mat}.cif` and `{mat}.sw`). Always generate with the `aiida` conda env
(`/home/matteo/miniconda3/envs/aiida/bin/python`); the base env lacks `lammps`.

## Two-step generation

1. **Build the per-material LAMMPS inputs** — `run <model> <config> -o <out>`.
   (`--hpc-scripts` on `run` only prints "use base output dir"; it does NOT emit the
   array — do step 2.)
2. **Generate the PBS array** — `hpc generate <out>/simulation_* --settings-file <settings.yaml>`.
   The settings file MUST set `hpc.modules` (e.g. `tools/prod`,
   `LAMMPS/29Aug2024-foss-2023b-kokkos`); otherwise the single-array path fails with
   `HPC modules list is empty` and the two-phase path silently omits the `module load`.
   Reuse a prior campaign's file, e.g. `scripts/run_pes_tip_bilayer/settings.yaml`
   (pbs, `$EPHEMERAL`, 32 cpu / 62 GB / 20 h, `max_array_size: 300`, the LAMMPS module).

## A. Freestanding tip-PES sweep (corrugation descriptors)

Substrate-free self-supported 4-layer slab, `layer_1` fixed, static `minimize`
scan over one surface cell (`grid_n = 12`), Si r25 tip. Config:
`examples/afm_freestanding_pes_campaign.ini` (no `[sub]` → freestanding builder).
Single-phase array (`slide.in` only): `#PBS -J 1-156`.

    PY=/home/matteo/miniconda3/envs/aiida/bin/python
    SET=scripts/run_pes_tip_bilayer/settings.yaml   # your HPC settings (modules!)
    $PY -m src.cli run pes-tip examples/afm_freestanding_pes_campaign.ini -o ./pes_campaign
    $PY -m src.cli hpc generate ./pes_campaign/simulation_* --settings-file "$SET"
    # submit: transfer ./pes_campaign to the cluster, follow hpc/submit_*.sh, then:
    $PY -m src.cli postprocess pes-scan ./pes_campaign/simulation_* \
        --data-dir data --grid-dir outputs/pes_scan
    # -> data/pes_scan_tip.csv (one row per material: pes_barrier_U, pes_fmax, ...)

Note: PES jobs are cheap (a static minimize scan, seconds); the reused 32-cpu / 20 h
resources are heavily over-provisioned — trim `num_cpus`/`walltime_hours` for this study.

## B. Adhesion (indent) sweep (pull-off / work-of-adhesion descriptors)

Substrate-based hold + quasi-static pull-off retract on a monolayer and bilayer,
Si r25 tip, load sweep `[5, 10, 20, 30]` nN. Loads loop IN-SCRIPT (one indent script
per material+layer, 312 total — do not set `outer_loop`). Two-phase `system.in`+`slide.in`
combined array: `#PBS -J 1-312`. Config: `examples/indent_campaign.ini`.

    $PY -m src.cli run indent examples/indent_campaign.ini -o ./indent_campaign
    $PY -m src.cli hpc generate ./indent_campaign/simulation_* --settings-file "$SET"
    $PY -m src.cli postprocess indent ./indent_campaign/simulation_* \
        --out-dir outputs/poke_sims --data-dir data
    # -> data/indent_features.csv (pull-off force, work of adhesion, contact
    #    stiffness, penetration, keyed material/layer/load)

## Notes

- **Per-material failures** are reported by the generator (`N/156 successful`) and do
  not abort the array. (In the first run all 156 built for both studies.)
- **Monolayer adhesion** (B, `layers = 1`) can reflect the sheet conforming/wrapping
  the pressed tip rather than a clean contact; retained as a comparison channel.
- The two studies are on different systems by design: A is substrate-free (intrinsic
  corrugation); B is on a substrate (pull-off is a normal-direction measurement, far
  less sensitive to the lateral substrate noise that motivated freestanding).
- Potentials resolve by literal path (no search): sheet `examples/potentials/sw/sw_lammps/{mat}.sw`,
  tip/sub `examples/potentials/sw/Si.sw`.
