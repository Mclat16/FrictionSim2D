# FrictionSim2D — Agent Instructions

## Project Overview
Python framework for generating and running LAMMPS friction simulations of 2D materials. Supports AFM and sheet-on-sheet setups, static PES (potential-energy-surface / corrugation-barrier) scans, HPC job generation, optional AiiDA provenance tracking, a REST API, and postprocessing/plotting tools.

See [documentation/mainpage.md](documentation/mainpage.md) for full project overview and [documentation/README.md](documentation/README.md) for the documentation index.

---

## Key Commands

```bash
# Install (development)
pip install -e .          # core only
pip install -e ".[all]"   # all optional deps (aiida, db, plotting, api)

# Run tests
pytest                    # all tests (verbose by default via pyproject.toml)
pytest tests/test_config.py   # single test file
pytest --cov              # with coverage

# CLI entry point
FrictionSim2D --help

# AiiDA bootstrap (one-time)
frictionsim2d-start-aiida
frictionsim2d-install-hooks
```

---

## Architecture

```
src/
  core/          Config loading (settings.yaml + .ini), LAMMPS runner, potential manager
  builders/      AFM, sheet-on-sheet, and PES-scan simulation builders; produce LAMMPS inputs
                 (pes_scan.py subclasses AFM/SheetOnSheet for static corrugation scans)
  interfaces/    External tool adapters: lammps.py, atomsk.py, jinja.py
  templates/     Jinja2 templates for LAMMPS input scripts and HPC job files
  hpc/           PBS/Slurm/Bash job script generation (scripts.py, manifest.py)
  data/          DB models, materials library, stored configs
  aiida/         AiiDA CalcJob, WorkChain, submit.py; custom data nodes
  api/           FastAPI REST server (server.py), client.py, auth.py
  postprocessing/ Data reading, COF analysis, stick-slip, plot_*.py modules
```

- **Lazy loading**: `src/__init__.py` uses `__getattr__` to defer module imports — keep startup fast.
- **Optional dependencies**: AiiDA, plotting, API, and DB extras are guarded by `_XXXX_AVAILABLE` flags. Never assume they are installed; check availability before importing.

---

## Configuration

Two config layers — see [documentation/configuration_guide.md](documentation/configuration_guide.md) and [documentation/settings_reference.md](documentation/settings_reference.md):

1. **`settings.yaml`** (project root) — global simulation parameters (thermostat, geometry, timestep, AiiDA profile, etc.). Loaded via `src.core.config.load_settings()` → `GlobalSettings` (Pydantic model).
2. **`.ini` files** — per-run AFM or sheet-on-sheet simulation configs. See [examples/afm_config.ini](examples/afm_config.ini) and [examples/sheet_config.ini](examples/sheet_config.ini).

---

## Testing Conventions

- Framework: **pytest**, configured in `pyproject.toml` (`-v --tb=short` by default).
- All tests in `tests/`, named `test_*.py`.
- **`tests/conftest.py`** provides `mock_settings` and related fixtures that supply a fully pre-configured `GlobalSettings` — use these fixtures to avoid file I/O in unit tests.
- Tests for optional-dependency code (AiiDA, API) are guarded with `pytest.importorskip` or similar.

---

## Simulation Workflow (High Level)

```
.ini config + settings.yaml
  → builders/ (AFM or SheetOnSheet) → LAMMPS input + HPC script
  → hpc/ (PBS/Slurm script)
  → Local run (core/run.py) OR AiiDA submission (aiida/submit.py)
  → output_full_*.json results
  → postprocessing/ (read_data, plot_data)
```

See [documentation/examples.md](documentation/examples.md) and [documentation/aiida_workflows.md](documentation/aiida_workflows.md).

---

## AiiDA Integration

- Custom entry points registered in `pyproject.toml`: `FrictionSimulationData`, `LammpsFrictionCalcJob`, `FrictionWorkChain`, `LammpsFrictionParser`.
- Requires PostgreSQL + RabbitMQ; bootstrap with `frictionsim2d-start-aiida`.
- See [documentation/PROVENANCE_ARCHITECTURE.md](documentation/PROVENANCE_ARCHITECTURE.md) and [documentation/HPC_TWO_PHASE_JOBS.md](documentation/HPC_TWO_PHASE_JOBS.md).

---

## Key Patterns & Pitfalls

- **Material names** follow a convention: prefix encodes structure (`h_` = hexagonal, `t_` = T-phase, `b_` = buckled, `p_` = phosphorene-like). Example: `h_MoS2`, `t_CoTe2`.
- **Potential types**: `sw` (Stillinger-Weber), `rebomos`, `reaxff` — used as marker/color keys in plots.
- **Simulation size** encoded as `100x100y` etc. — used as filter keys in postprocessing.
- **Conda environment**: active env is `aiida` for this project. Don't assume `base`.
