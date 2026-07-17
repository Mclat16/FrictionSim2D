# FrictionSim2D Documentation

This folder contains user documentation for FrictionSim2D, a package that generates LAMMPS-ready friction simulation inputs for 2D materials.

## Start Here

1. [installation.md](installation.md): environment setup and dependency checks
2. [mainpage.md](mainpage.md): what FrictionSim2D does and quick start
3. [configuration_guide.md](configuration_guide.md): how to write valid config files
4. [commands.md](commands.md): complete CLI reference
5. [examples.md](examples.md): practical workflows

## Core User Guides

- [essentials.md](essentials.md): simulation lifecycle, run layout, and common patterns
- [pes_scan.md](pes_scan.md): static potential-energy-surface (corrugation-barrier) scans
- [indent.md](indent.md): AFM indentation ("indent") sims — penetration/compliance (hold) + adhesion/pull-off (retract) in one run
- [settings_reference.md](settings_reference.md): global settings used during generation
- [aiida_workflows.md](aiida_workflows.md): AiiDA setup, submission, and result import
- [python_api_guide.md](python_api_guide.md): programmatic usage from Python
- [HPC_TWO_PHASE_JOBS.md](HPC_TWO_PHASE_JOBS.md): scheduler script behavior and submission order

## Internal Architecture

- [PROVENANCE_ARCHITECTURE.md](PROVENANCE_ARCHITECTURE.md): provenance manifest model and file tracking internals

## Publication Dataset

The [`publication/`](publication/) folder contains the data and configuration files accompanying the published dataset (DOI: 10.82186/gs4dc-90a11):

- [publication/README.md](publication/README.md): dataset overview, import instructions, and how to reproduce all publication figures
- [publication/plots_publication.json](publication/plots_publication.json): master plot configuration used in the publication

## Which Docs Are Most Useful For Typical Users?

If your goal is to generate and run simulations quickly, prioritize:

1. [installation.md](installation.md)
2. [configuration_guide.md](configuration_guide.md)
3. [commands.md](commands.md)
4. [examples.md](examples.md)

The architecture document is useful for reproducibility and development work, but not required for day-to-day simulation setup.

## Suggested Reading Paths

CLI-first users:

1. [mainpage.md](mainpage.md)
2. [configuration_guide.md](configuration_guide.md)
3. [commands.md](commands.md)
4. [examples.md](examples.md)

Python/API-first users:

1. [python_api_guide.md](python_api_guide.md)
2. [configuration_guide.md](configuration_guide.md)
3. [settings_reference.md](settings_reference.md)
4. [aiida_workflows.md](aiida_workflows.md)
