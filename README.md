# FrictionSim2D

FrictionSim2D generates and manages LAMMPS friction simulation workflows for 2D materials.
It supports AFM and sheet-on-sheet setups, HPC script generation, and optional AiiDA provenance/submission workflows.

## Installation

FrictionSim2D is not yet published on conda-forge. The supported installation path today is to create an environment yourself, then install the package from source.

### Requirements

- Python 3.9+
- LAMMPS available in your environment
- Atomsk available in your environment (optional, only for Atomsk-specific workflows)

### Create an environment

Using conda:

```bash
conda create -n frictionsim2d python=3.11
conda activate frictionsim2d
```

Or using venv:

```bash
python -m venv .venv
source .venv/bin/activate
```

### Install from source

From the repository root:

```bash
pip install -e .
```

Install optional extras as needed:

```bash
pip install -e .[aiida]
pip install -e .[db]
pip install -e .[plotting]
pip install -e .[api]
pip install -e .[all]
```

### Verify

```bash
FrictionSim2D --help
FrictionSim2D --version
lmp -h
```

If you plan to use Atomsk-based geometry operations, also verify:

```bash
atomsk --version
```

### Optional AiiDA setup

After installing the AiiDA extras, run:

```bash
FrictionSim2D aiida status
FrictionSim2D aiida setup
```

If your workflows target a remote machine, configure the `aiida` section in `settings.yaml` and use:

```bash
FrictionSim2D aiida setup --use-remote
```

### Notes

- RabbitMQ and PostgreSQL are still required for full AiiDA workflows.
- The conda-forge installation scheme below is planned for a future release and is intentionally hidden for now.

<!--
Future conda-forge installation draft

FrictionSim2D is distributed as a Conda package for maximum reproducibility and minimal dependency conflicts.

### Option A – Base package (recommended for most users)

Install LAMMPS, Atomsk, and core dependencies:

```bash
conda create -n frictionsim2d -c conda-forge frictionsim2d
conda activate frictionsim2d
```

Verify:
```bash
FrictionSim2D --help
lmp -h
atomsk --version
```

### Option B – With AiiDA (provenance tracking, HPC submission)

Install base package + AiiDA with bundled PostgreSQL and RabbitMQ:

```bash
conda create -n frictionsim2d-aiida -c conda-forge frictionsim2d-aiida
conda activate frictionsim2d-aiida
```

Then bootstrap AiiDA and start services (one-time setup):

```bash
frictionsim2d-start-aiida
```

This:
- Starts RabbitMQ with automatic `consumer_timeout` patching (36000000 ms default)
- Starts PostgreSQL
- Creates your first AiiDA profile (`friction2d`)

Options: `--profile NAME` to use a custom profile name, `--no-daemon` to skip daemon startup.

Override RabbitMQ timeout if needed:
```bash
export RABBITMQ_CONSUMER_TIMEOUT_MS=72000000
frictionsim2d-start-aiida
```

Optional: install conda activate/deactivate hooks for automatic service lifecycle:
```bash
frictionsim2d-install-hooks
```

After this, PostgreSQL and RabbitMQ start/stop automatically with `conda activate/deactivate`.

Verify AiiDA setup:
```bash
verdi --version
verdi profile show
verdi daemon status
```

### Option C – Community database only

```bash
conda create -n frictionsim2d-db -c conda-forge frictionsim2d-db
conda activate frictionsim2d-db
```

### Option D – Postprocessing / plotting

```bash
conda create -n frictionsim2d-plotting -c conda-forge frictionsim2d-plotting
conda activate frictionsim2d-plotting
```

### Option E – REST API server

```bash
conda create -n frictionsim2d-api -c conda-forge frictionsim2d-api
conda activate frictionsim2d-api
```

### Option F – All-in-one

```bash
conda create -n frictionsim2d-all -c conda-forge frictionsim2d-all
conda activate frictionsim2d-all
frictionsim2d-start-aiida
```
-->

## Quick start

Generate AFM simulations:

```bash
FrictionSim2D run afm examples/afm_config.ini --output-dir ./simulation_output
```

Generate sheet-on-sheet simulations:

```bash
FrictionSim2D run sheetonsheet examples/sheet_config.ini --output-dir ./simulation_output
```

Generate with AiiDA registration enabled:

```bash
FrictionSim2D run afm examples/afm_config.ini --aiida --output-dir ./simulation_output
```

Generate HPC scripts:

```bash
FrictionSim2D hpc generate ./simulation_output/simulation_YYYYMMDD_HHMMSS --scheduler pbs
```

## CLI overview

- `FrictionSim2D run afm ...`
- `FrictionSim2D run sheetonsheet ...`
- `FrictionSim2D hpc generate ...`
- `FrictionSim2D settings show|init|reset`
- `FrictionSim2D aiida status|setup|submit|import|query|export|import-archive|package`
- `FrictionSim2D postprocess read|plot`
- `FrictionSim2D db init|create-key|upload|stage|query|stats|delete|publish|reject`
- `FrictionSim2D api serve`

## Shared database

FrictionSim2D supports an API-first shared database model for publications.

- Users query and upload through a public REST API URL.
- Database credentials stay only on the server.
- Each contributor gets a personal API key.

### Publication links to put on GitHub

- Public FrictionSim2DDB API: https://YOUR-DOMAIN-OR-IP:8000
- Local mirror API (for local development only): http://localhost:8000

Important: localhost is only reachable on the same machine. It is not a public link.

### Configure connection

For server admins (machine running PostgreSQL + API server), set database credentials:

```bash
export FRICTION_DB_HOST=db.example.com
export FRICTION_DB_PORT=5432
export FRICTION_DB_NAME=frictionsim2d
export FRICTION_DB_USER=myuser
export FRICTION_DB_PASSWORD=secret
```

For package users/collaborators, set API URL and personal API key (no DB password needed):

```bash
export FRICTION_API_URL=https://YOUR-DOMAIN-OR-IP:8000
export FRICTION_API_KEY=sk_your_personal_key
```

### Start the API server (server admin)

```bash
FrictionSim2D db init --profile local
FrictionSim2D api serve --host 0.0.0.0 --port 8000
```

### Create contributor keys (server admin)

```bash
FrictionSim2D db create-key --name alice --profile local
FrictionSim2D db create-key --name bob --profile local
```

### Upload results

```bash
# Upload through authenticated REST endpoint (example)
curl -X POST "$FRICTION_API_URL/results" \
	-H "X-API-Key: $FRICTION_API_KEY" \
	-H "Content-Type: application/json" \
	-d '{"material":"h-MoS2","simulation_type":"afm","layers":1,"mean_cof":0.03}'
```

### Query results

```bash
# Print table to terminal
FrictionSim2D db query --material h-MoS2 --layers 1

# Save to CSV
FrictionSim2D db query --material h-WS2 --csv output.csv
```

### Database statistics

```bash
FrictionSim2D db stats
```

### Delete your own rows

```bash
FrictionSim2D db delete --uploader alice
```

## AiiDA notes

- Use `FrictionSim2D aiida setup` for first-time profile/computer/code setup.
- For secure HPC clusters, running AiiDA natively on the HPC environment is often more robust than local SSH transport.
- Use archive transfer when needed: `aiida export` on source environment and `aiida import-archive` on target.

## Output structure

Generated simulation roots typically contain:

- `afm/` or `sheetonsheet/` runs
- per-run `lammps/`, `data/`, `results/`, `visuals/`, `provenance/`
- optional `hpc/` submission scripts

## Documentation

- [documentation/mainpage.md](documentation/mainpage.md)
- [documentation/installation.md](documentation/installation.md)
- [documentation/essentials.md](documentation/essentials.md)
- [documentation/commands.md](documentation/commands.md)
- [documentation/examples.md](documentation/examples.md)
- [documentation/PROVENANCE_ARCHITECTURE.md](documentation/PROVENANCE_ARCHITECTURE.md)

## Development

Run tests:

```bash
pytest -v
```

Run lint (example):

```bash
pylint src/aiida
```
