# Commands Reference

This document reflects the current CLI in `src/cli.py`.

## Top Level

```bash
FrictionSim2D --help
FrictionSim2D --version
```

## run

### run afm

```bash
FrictionSim2D run afm CONFIG_FILE [OPTIONS]
```

Options:

- `--settings-file PATH` — per-run settings.yaml override
- `-o, --output-dir TEXT` (default: `simulation_output`)
- `--aiida` — register generated simulations in AiiDA
- `--hpc-scripts` — generate PBS/SLURM scripts in the same step

### run sheetonsheet

```bash
FrictionSim2D run sheetonsheet CONFIG_FILE [OPTIONS]
```

Same options as `run afm`.

### run pes-sheet

```bash
FrictionSim2D run pes-sheet CONFIG_FILE [OPTIONS]
```

Generate a static **sheet-on-sheet interlayer PES scan**: a frozen bilayer whose
top layer is grid-scanned laterally over one surface unit cell (energy map,
per-cell normalized) instead of a sliding run. Requires `[2D] layers = [2]` and a
`[pes]` section (`grid_n`, optional `grid_n_refine`, `z_relax`). Same options as
`run afm`. See [pes_scan.md](pes_scan.md).

### run pes-tip

```bash
FrictionSim2D run pes-tip CONFIG_FILE [OPTIONS]
```

Generate a static **tip-on-sheet (AFM) surface PES scan**: builds and indents the
tip + sheet + substrate stack to a low load exactly as the sliding sim, then
grid-scans the rigid tip laterally, recording energy and the lateral force on the
tip. Same options as `run afm`. See [pes_scan.md](pes_scan.md).

## settings

### settings show

```bash
FrictionSim2D settings show [--settings-file PATH] [--origin]
```

Options:

- `--settings-file PATH` — inspect a specific file rather than the resolved default
- `--origin` — also print which settings file was loaded

### settings init

```bash
FrictionSim2D settings init [--global] [--force]
```

Options:

- `--global` — write to `~/.config/FrictionSim2D/settings.yaml`
- `--force` — overwrite without prompting

### settings reset

```bash
FrictionSim2D settings reset [--global]
```

Options:

- `--global` — remove the global settings file instead of the local one

## hpc

### hpc generate

```bash
FrictionSim2D hpc generate SIMULATION_DIR [OPTIONS]
```

Options:

- `--settings-file PATH`
- `-s, --scheduler {pbs|slurm}` (default: `pbs`)
- `-o, --output-dir TEXT` (default: `SIMULATION_DIR/hpc`)

## aiida

> AiiDA commands require `aiida-core`. Install with `pip install -e .[aiida]`.

### aiida status

```bash
FrictionSim2D aiida status
```

### aiida setup

```bash
FrictionSim2D aiida setup [OPTIONS]
```

Options:

- `-p, --profile TEXT`
- `--lammps-path PATH`
- `--use-remote` — configure a remote HPC computer using `settings.yaml`
- `--hpc-config PATH` *(deprecated — use `--use-remote` instead)*
- `--settings-file PATH`

### aiida import

Import a `simulation_XXXXXXXX` directory as a new simulation set:

```bash
FrictionSim2D aiida import SIMULATION_FOLDER [OPTIONS]
```

Options:

- `-l, --label TEXT` — required label (prompted if omitted)
- `-d, --description TEXT`
- `--profile TEXT`

### aiida list-sets

```bash
FrictionSim2D aiida list-sets [OPTIONS]
```

Options:

- `--profile TEXT`
- `--format {table|json|csv}` (default: `table`)

### aiida dump

Export time-series data from an AiiDA set to `output_full_*.json` files, ready for `postprocess plot`:

```bash
FrictionSim2D aiida dump SET_LABEL --output-dir OUTPUT_DIR [OPTIONS]
```

Options:

- `-o, --output-dir PATH` *(required)*
- `--profile TEXT`

Output layout: `OUTPUT_DIR/outputs/output_full_<size>.json`

### aiida rebuild

Reconstruct a full simulation directory tree from AiiDA provenance data alone:

```bash
FrictionSim2D aiida rebuild SET_LABEL [OPTIONS]
```

Options:

- `-o, --output-dir PATH` (default: current directory)
- `--hpc-scripts` — also generate PBS/SLURM scripts
- `--profile TEXT`

### aiida query

```bash
FrictionSim2D aiida query [OPTIONS]
```

Options:

- `-m, --material TEXT`
- `-l, --layers INTEGER`
- `-f, --force FLOAT`
- `-s, --set TEXT` — filter by simulation set label
- `--format {table|csv|json}` (default: `table`)
- `-o, --output PATH`
- `--profile TEXT`

### aiida delete

Delete a single simulation set and all linked nodes (irreversible):

```bash
FrictionSim2D aiida delete SET_LABEL [--profile TEXT]
```

### aiida clear

Delete **all** FrictionSim2D nodes from the database (irreversible):

```bash
FrictionSim2D aiida clear [--profile TEXT]
```

### aiida export

Export nodes to a portable AiiDA archive file:

```bash
FrictionSim2D aiida export [OPTIONS]
```

Options:

- `-o, --output PATH` (default: `friction2d.aiida`)
- `-m, --material TEXT`

### aiida import-archive

Import a previously exported AiiDA archive into the current profile:

```bash
FrictionSim2D aiida import-archive ARCHIVE_PATH
```

### aiida package

Create a `tar.gz` of simulation inputs for cluster transfer (excludes `.lammpstrj`):

```bash
FrictionSim2D aiida package SIMULATION_DIR [-o OUTPUT]
```

### aiida submit

```bash
FrictionSim2D aiida submit SIMULATION_DIR [OPTIONS]
```

Options:

- `--settings-file PATH`
- `-c, --code TEXT`
- `--scripts TEXT` (comma-separated list)
- `--array`
- `--machines INTEGER`
- `--mpiprocs INTEGER`
- `--walltime TEXT` (`HH:MM:SS` or integer seconds)
- `--queue TEXT`
- `--project TEXT`
- `--dry-run`

## postprocess

### postprocess read

Walk a simulation directory and export time-series data to JSON:

```bash
FrictionSim2D postprocess read RESULTS_DIR [--export]
```

Options:

- `--export` — write `output_full_*.json` files alongside results

### postprocess plot

Generate plots from a JSON plot configuration file:

```bash
FrictionSim2D postprocess plot PLOT_CONFIG [OPTIONS]
```

Options:

- `-o, --output-dir TEXT` (default: `plots`)
- `--settings PATH` — additional JSON plot-style settings

`PLOT_CONFIG` must define `data_dirs`, `labels`, and `plots`. See [examples.md](examples.md) and the reference config in `documentation/publication/plots_publication.json`.

### postprocess pes-scan

Reduce PES-scan energy/force grids to per-material descriptor CSVs:

```bash
FrictionSim2D postprocess pes-scan RUN_ROOT [OPTIONS]
```

Options:

- `-d, --data-dir PATH` (default: `data`) — output dir for `pes_scan_{sheet,tip}.csv`
- `--grid-dir PATH` — also copy tidied per-material grids to `<grid-dir>/{sheet,tip}/<material>.csv`
- `--kind [sheet|tip|both]` (default: `both`)

Scans `RUN_ROOT` for `pes_sheet/` and `pes_tip/` results and writes one row per
material (merge key `material`), ready for the ML feature pipeline. See
[pes_scan.md](pes_scan.md).

## db

The `db` command group manages the **central (remote) PostgreSQL database** — the shared summary-statistics store.
It is **not** the local AiiDA database; for full provenance and time-series access, use the `aiida` command group.

> **Two-database architecture**
> - `FrictionSim2D aiida ...` → local AiiDA database (full time-series, complete provenance)
> - `FrictionSim2D db ...` → central remote PostgreSQL (summary statistics: mean COF, forces, conditions)

All `db` commands accept a common set of connection options. Pass `--profile central` to target the shared remote DB; `--profile local` targets a local instance.

**Common DB options** (available on every `db` subcommand):

```
--profile TEXT        Profile from settings.yaml ('local' or 'central')
--host TEXT           $FRICTION_DB_HOST
--port INTEGER        $FRICTION_DB_PORT
--dbname TEXT         $FRICTION_DB_NAME
-u, --user TEXT       $FRICTION_DB_USER
--password TEXT       $FRICTION_DB_PASSWORD
```

### db setup

First-time setup: initializes schema, creates an API key, and saves it to settings:

```bash
FrictionSim2D db setup [--name NAME] [DB OPTIONS]
```

### db init

Initialize or verify the database schema (safe to re-run):

```bash
FrictionSim2D db init [DB OPTIONS]
```

### db create-key

Generate a new API key for write access:

```bash
FrictionSim2D db create-key --name NAME [DB OPTIONS]
```

### db upload

Upload simulation results to the shared database:

```bash
FrictionSim2D db upload RESULTS_DIR [--uploader NAME] [DB OPTIONS]
```

Each row is validated automatically; valid rows are published.

### db stage

Upload with an API key; rows are validated and published or rejected automatically:

```bash
FrictionSim2D db stage RESULTS_DIR --uploader NAME [--api-key KEY] [DB OPTIONS]
```

### db query

```bash
FrictionSim2D db query [OPTIONS] [DB OPTIONS]
```

Options:

- `-m, --material TEXT`
- `--type TEXT` (`afm` or `sheetonsheet`)
- `-l, --layers INTEGER`
- `-n, --uploader TEXT`
- `--limit INTEGER` (default: 50)
- `--csv PATH`

### db stats

```bash
FrictionSim2D db stats [DB OPTIONS]
```

### db publish

Promote a staged/validated result to published (curator action):

```bash
FrictionSim2D db publish ROW_ID [DB OPTIONS]
```

### db reject

Reject a staged or validated result:

```bash
FrictionSim2D db reject ROW_ID [--reason TEXT] [DB OPTIONS]
```

### db delete

Delete all rows belonging to a given uploader:

```bash
FrictionSim2D db delete --uploader NAME [DB OPTIONS]
```

## api

### api serve

```bash
FrictionSim2D api serve [OPTIONS]
```

Options:

- `--host TEXT`
- `-p, --port INTEGER`
- `--profile TEXT`
- `--reload` — auto-reload on code changes (development only)
- `--settings-file PATH`

## Practical Notes

- For the most up-to-date usage, run `FrictionSim2D <group> <command> --help`.
- `run` commands expose `--hpc-scripts` to generate scheduler scripts in one step.
- `postprocess plot` requires a JSON config file, not a results directory.
- `aiida dump` and `postprocess read --export` both produce `output_full_*.json` files, but `aiida dump` reads from the AiiDA database while `postprocess read` walks the raw LAMMPS output files on disk.

## Related Docs

- [configuration_guide.md](configuration_guide.md)
- [examples.md](examples.md)
- [aiida_workflows.md](aiida_workflows.md)
