# Large-scale campaigns (155-material set)

Two campaigns run over `examples/material_list.txt` (155 materials, each with an
in-repo `{mat}.cif` and `{mat}.sw`). Always generate with the `aiida` conda env
(`/home/matteo/miniconda3/envs/aiida/bin/python`); the base env lacks `lammps`.

## A. Freestanding tip-PES sweep (corrugation descriptors)

Substrate-free self-supported 4-layer slab, `layer_1` fixed, static `minimize`
scan over one surface cell (`grid_n = 12`), Si r25 tip. Config:
`examples/afm_freestanding_pes_campaign.ini` (no `[sub]` → freestanding builder).

    # 1) generate the array
    FrictionSim2D run pes-tip examples/afm_freestanding_pes_campaign.ini \
        -o ./pes_campaign --hpc-scripts
    # 2) submit: transfer ./pes_campaign to the cluster and follow its hpc/ scripts
    # 3) reduce results
    FrictionSim2D postprocess pes-scan ./pes_campaign/simulation_* \
        --data-dir data --grid-dir outputs/pes_scan
    # -> data/pes_scan_tip.csv (one row per material: pes_barrier_U, pes_fmax, ...)

## B. Adhesion (indent) sweep (pull-off / work-of-adhesion descriptors)

Substrate-based hold + quasi-static pull-off retract on a monolayer and bilayer,
Si r25 tip, load sweep `[5, 10, 20, 30]` nN. Loads loop IN-SCRIPT (one indent
script per material+layer, 310 total — do not set `outer_loop`). Config:
`examples/indent_campaign.ini`.

    FrictionSim2D run indent examples/indent_campaign.ini \
        -o ./indent_campaign --hpc-scripts
    FrictionSim2D postprocess indent ./indent_campaign/simulation_* \
        --out-dir outputs/poke_sims --data-dir data
    # -> data/indent_features.csv (pull-off force, work of adhesion, contact
    #    stiffness, penetration, keyed material/layer/load)

## Notes

- **Per-material failures** are reported by the generator (`N/155 successful`) and
  do not abort the array — some materials may not build as a 4-layer slab (A) or
  indent stack (B).
- **Monolayer adhesion** (B, `layers = 1`) can reflect the sheet conforming/wrapping
  the pressed tip rather than a clean contact; retained as a comparison channel.
- The two studies are on different systems by design: A is substrate-free (intrinsic
  corrugation); B is on a substrate (pull-off is a normal-direction measurement, far
  less sensitive to the lateral substrate noise that motivated freestanding).
