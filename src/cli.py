"""Command Line Interface for FrictionSim2D.

Unified CLI for simulation execution, HPC script generation, and AiiDA integration.
"""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional, NoReturn
import click

from .core.config import load_settings, settings_origin, _global_settings_path
from .core.run import (
    run_simulations,
    _build_hpc_manifest_entries,
    dispatch_hpc_generation,
    layer_aware_path_sort_key,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


def _bootstrap_local_db_once() -> None:
    """Best-effort one-time local schema bootstrap for new environments.

    This is intentionally non-fatal: if PostgreSQL credentials or server are
    not available yet, FrictionSim2D commands should still run.
    """
    marker = Path.home() / '.config' / 'FrictionSim2D' / '.local_db_bootstrapped'
    if marker.exists():
        return

    try:
        from .data.database import db_from_profile  # noqa: PLC0415
        db_from_profile('local')
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(f"{datetime.utcnow().isoformat()}Z\n", encoding='utf-8')
        logger.info("Bootstrapped local FrictionSim2D DB schema.")
    except Exception as exc:  # pylint: disable=broad-except
        logger.info("Skipping local DB bootstrap: %s", exc)


def _ensure_hpc_settings(hpc_settings) -> None:
    """Ensure minimal HPC settings are set without interactive prompts."""
    # Scheduler type and scratch directory are the only required values.
    if not getattr(hpc_settings, 'scheduler_type', None):
        hpc_settings.scheduler_type = 'pbs'

    if getattr(hpc_settings, 'use_tmpdir', False) and not getattr(hpc_settings, 'scratch_dir', None):
        hpc_settings.scratch_dir = "$TMPDIR"


def _run_simulation(model: str, config_file: str, output_dir: str,
                    use_aiida: bool, generate_hpc: bool,
                    settings_file: Optional[Path] = None):
    if use_aiida:
        from .aiida import AIIDA_AVAILABLE
        if not AIIDA_AVAILABLE:
            click.echo("⚠️  AiiDA not available. Install with:", err=True)
            click.echo("   conda install -c conda-forge aiida-core", err=True)
            raise click.Abort()

    created_simulations, simulation_root, configs_to_run, _ = run_simulations(
        config_file=config_file,
        model=model,
        settings_file=settings_file,
        output_root=Path(output_dir),
        ensure_hpc_settings=_ensure_hpc_settings,
        use_aiida=use_aiida,
        generate_hpc=generate_hpc,
    )

    click.echo(f"📋 Found {len(configs_to_run)} simulation configurations")

    if created_simulations and use_aiida:
        click.echo(f"\n📦 Registering {len(created_simulations)} simulations in AiiDA...")
        _register_simulations_aiida(created_simulations, Path(config_file))

    click.echo(f"\n✅ Generation complete: {len(created_simulations)}/{len(configs_to_run)} successful")
    click.echo(f"📂 Output directory: {simulation_root.absolute()}")

# =============================================================================
# MAIN CLI GROUP
# =============================================================================

@click.group()
@click.version_option(version='0.1.0', prog_name='FrictionSim2D')
def cli():
    """FrictionSim2D - Generate LAMMPS input files for 2D material friction simulations."""


# =============================================================================
# RUN COMMANDS
# =============================================================================

@cli.group('run')
def run_group():
    """Run friction simulations."""


@run_group.command('afm')
@click.argument('config_file', type=click.Path(exists=True))
@click.option('--settings-file', type=click.Path(exists=True, dir_okay=False, path_type=Path),
              default=None,
              help='Path to a settings.yaml file for this run only.')
@click.option('--output-dir', '-o', default='simulation_output',
              help='Output directory for generated files')
@click.option('--aiida', 'use_aiida', is_flag=True,
              help='Enable AiiDA provenance tracking')
@click.option('--hpc-scripts', 'generate_hpc', is_flag=True,
                            help='Generate HPC scripts for the simulation root')
def run_afm(config_file: str, settings_file: Optional[Path], output_dir: str,
            use_aiida: bool, generate_hpc: bool):
    """Generate AFM simulation files.
    
    Creates all necessary LAMMPS input files, structures, and potentials
    for tip-on-substrate friction simulations.
    
    Example:
        FrictionSim2D run.afm afm_config.ini -o ./afm_output --aiida
    """
    _run_simulation(
        'afm', config_file, output_dir, use_aiida, generate_hpc, settings_file=settings_file
    )

@run_group.command('sheetonsheet')
@click.argument('config_file', type=click.Path(exists=True))
@click.option('--settings-file', type=click.Path(exists=True, dir_okay=False, path_type=Path),
          default=None,
          help='Path to a settings.yaml file for this run only.')
@click.option('--output-dir', '-o', default='simulation_output',
              help='Output directory for generated files')
@click.option('--aiida', 'use_aiida', is_flag=True,
              help='Enable AiiDA provenance tracking')
@click.option('--hpc-scripts', 'generate_hpc', is_flag=True,
        help='Generate HPC scripts for the simulation root')
def run_sheetonsheet(config_file: str, settings_file: Optional[Path], output_dir: str,
             use_aiida: bool, generate_hpc: bool):
    """Generate sheet-on-sheet simulation files.
    
    Creates all necessary LAMMPS input files for 4-layer sheet-on-sheet
    friction simulations.
    
    Example:
        FrictionSim2D run.sheetonsheet sheet_config.ini -o ./sheet_output
    """
    _run_simulation(
        'sheetonsheet', config_file, output_dir, use_aiida, generate_hpc, settings_file=settings_file
    )


@run_group.command('pes-sheet')
@click.argument('config_file', type=click.Path(exists=True))
@click.option('--settings-file', type=click.Path(exists=True, dir_okay=False, path_type=Path),
              default=None,
              help='Path to a settings.yaml file for this run only.')
@click.option('--output-dir', '-o', default='simulation_output',
              help='Output directory for generated files')
@click.option('--aiida', 'use_aiida', is_flag=True,
              help='Enable AiiDA provenance tracking')
@click.option('--hpc-scripts', 'generate_hpc', is_flag=True,
              help='Generate HPC scripts for the simulation root')
def run_pes_sheet(config_file: str, settings_file: Optional[Path], output_dir: str,
                  use_aiida: bool, generate_hpc: bool):
    """Generate sheet-on-sheet interlayer PES scan files.

    Builds a frozen bilayer and writes a cheap, static lateral grid-scan of the
    interlayer potential-energy surface (corrugation barrier) instead of a
    sliding run. Reuses the sheet-on-sheet geometry; needs [2D] layers = [2].

    Example:
        FrictionSim2D run pes-sheet examples/pes_sheet_config.ini -o ./pes_output
    """
    _run_simulation(
        'pes_sheet', config_file, output_dir, use_aiida, generate_hpc, settings_file=settings_file
    )


@run_group.command('pes-tip')
@click.argument('config_file', type=click.Path(exists=True))
@click.option('--settings-file', type=click.Path(exists=True, dir_okay=False, path_type=Path),
              default=None,
              help='Path to a settings.yaml file for this run only.')
@click.option('--output-dir', '-o', default='simulation_output',
              help='Output directory for generated files')
@click.option('--aiida', 'use_aiida', is_flag=True,
              help='Enable AiiDA provenance tracking')
@click.option('--hpc-scripts', 'generate_hpc', is_flag=True,
              help='Generate HPC scripts for the simulation root')
def run_pes_tip(config_file: str, settings_file: Optional[Path], output_dir: str,
                use_aiida: bool, generate_hpc: bool):
    """Generate tip-on-sheet (AFM) surface PES scan files.

    Builds and indents the AFM tip + sheet + substrate stack to a low load
    exactly as the sliding sim, then writes a static lateral grid-scan of the
    tip-felt surface PES (energy + lateral force on the tip) instead of sliding.

    Example:
        FrictionSim2D run pes-tip examples/pes_tip_config.ini -o ./pes_output
    """
    _run_simulation(
        'pes_tip', config_file, output_dir, use_aiida, generate_hpc, settings_file=settings_file
    )


@run_group.command('indent')
@click.argument('config_file', type=click.Path(exists=True))
@click.option('--settings-file', type=click.Path(exists=True, dir_okay=False, path_type=Path),
              default=None,
              help='Path to a settings.yaml file for this run only.')
@click.option('--output-dir', '-o', default='simulation_output',
              help='Output directory for generated files')
@click.option('--aiida', 'use_aiida', is_flag=True,
              help='Enable AiiDA provenance tracking')
@click.option('--hpc-scripts', 'generate_hpc', is_flag=True,
              help='Generate HPC scripts for the simulation root')
def run_indent(config_file: str, settings_file: Optional[Path], output_dir: str,
               use_aiida: bool, generate_hpc: bool):
    """Generate AFM indentation ("indent") simulation files: hold + pull-off retract.

    The standard AFM build + indentation ramp (system.in) to each load in
    [general] force, then per load a damped finite-T hold followed by a
    displacement-controlled quasi-static retract of the rigid tip to full
    detachment — the sliding phase is never invoked. Records both the penetration
    channels (tip_z − sheet_com_z, vertical compliance) and the force–distance
    retract curve (pull-off force, work of adhesion, contact stiffness). Uses a
    fresh, recorded velocity seed per (material, layer). Subsumes the former
    separate poke (hold-only) and adhesion (retract-only) commands.

    Example:
        FrictionSim2D run indent examples/indent_config.ini -o ./indent_output --hpc-scripts
    """
    _run_simulation(
        'indent', config_file, output_dir, use_aiida, generate_hpc, settings_file=settings_file
    )


def _register_simulations_aiida(simulation_dirs: List[Path], config_path: Path):
    """Register generated simulations with AiiDA."""
    try:
        from .aiida.integration import register_simulation_batch
        registered = register_simulation_batch(simulation_dirs, config_path)
        click.echo(f"   ✅ Registered {len(registered)} simulations")
    except ImportError:
        click.echo("   ⚠️  AiiDA integration module not found", err=True)
    except Exception as exc:  # pylint: disable=broad-except
        click.echo(f"   ⚠️  Registration failed: {exc}", err=True)
        logger.exception("AiiDA registration failed")


def _parse_walltime(value: Optional[str]) -> Optional[int]:
    if not value:
        return None
    if value.isdigit():
        return int(value)
    parts = value.split(':')
    if len(parts) == 3:
        hours, minutes, seconds = [int(p) for p in parts]
        return hours * 3600 + minutes * 60 + seconds
    raise ValueError("Walltime must be seconds or HH:MM:SS")


def _raise_abort(message: str, exc: Exception) -> NoReturn:
    click.echo(message, err=True)
    raise click.Abort() from exc


# =============================================================================
# SETTINGS COMMANDS
# =============================================================================

@cli.group('settings')
def settings_group():
    """Manage simulation settings."""


@settings_group.command('show')
@click.option('--settings-file', type=click.Path(exists=True, dir_okay=False, path_type=Path),
              default=None,
              help='Explicit settings.yaml path to inspect (run-specific).')
@click.option('--origin', is_flag=True,
              help='Show which settings file is currently in effect.')
def settings_show(settings_file: Optional[Path], origin: bool):
    """Display current effective settings.

    With --origin, also prints the path of the settings file that was loaded
    (or 'hardcoded defaults' if no file is found).

    Example:
        FrictionSim2D settings show
        FrictionSim2D settings show --origin
    """
    import yaml  # noqa: PLC0415
    if origin:
        src = settings_origin(settings_file=settings_file)
        if src:
            click.echo(f"# Settings loaded from: {src}")
        else:
            click.echo("# Settings loaded from: hardcoded defaults (no settings.yaml found)")
    effective = load_settings(settings_file=settings_file)
    click.echo(yaml.dump(effective.model_dump(), default_flow_style=False))


@settings_group.command('init')
@click.option('--global', 'global_', is_flag=True,
              help='Write to ~/.config/FrictionSim2D/settings.yaml (semi-permanent).')
@click.option('--force', is_flag=True,
              help='Overwrite an existing file without prompting.')
def settings_init(global_: bool, force: bool):
    """Create a settings.yaml file pre-populated with current effective settings.

    Without --global, writes settings.yaml to the current directory for
    per-simulation customisation. This file is only used when explicitly
    supplied to commands (e.g. run --settings-file). With --global, writes to
    ~/.config/FrictionSim2D/settings.yaml for machine-level configuration
    such as HPC host, queue names, and database credentials.

    Example:
        FrictionSim2D settings init                # create per-run file
        FrictionSim2D settings init --global       # first-time machine setup
    """
    import yaml  # noqa: PLC0415

    target = _global_settings_path() if global_ else Path('settings.yaml')

    if target.exists() and not force:
        click.confirm(
            f"'{target}' already exists. Overwrite?",
            abort=True,
        )

    effective = load_settings()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        yaml.safe_dump(effective.model_dump(), sort_keys=False),
        encoding='utf-8',
    )
    click.echo(f"✅ Created '{target}'")
    if global_:
        click.echo("   Edit this file to set HPC, database, and other machine-level defaults.")
    else:
        click.echo("   Edit this file and pass it via --settings-file for per-simulation settings.")


@settings_group.command('reset')
@click.option('--global', 'global_', is_flag=True,
              help='Remove ~/.config/FrictionSim2D/settings.yaml instead.')
@click.confirmation_option(prompt='This will delete the settings file. Continue?')
def settings_reset(global_: bool):
    """Delete a settings.yaml file and revert to defaults.

    Without --global, removes settings.yaml from the current directory.
    With --global, removes ~/.config/FrictionSim2D/settings.yaml.

    Example:
        FrictionSim2D settings reset
        FrictionSim2D settings reset --global
    """
    target = _global_settings_path() if global_ else Path('settings.yaml')
    if target.exists():
        target.unlink()
        click.echo(f"✅ Removed '{target}'")
    else:
        click.echo(f"ℹ️  No settings file found at '{target}'")

# =============================================================================
# HPC COMMANDS
# =============================================================================

@cli.group('hpc')
def hpc_group():
    """Generate HPC submission scripts."""


@hpc_group.command('generate')
@click.argument('simulation_dir', type=click.Path(exists=True))
@click.option('--settings-file', type=click.Path(exists=True, dir_okay=False, path_type=Path),
              default=None,
              help='Path to a settings.yaml file for this command.')
@click.option('--scheduler', '-s', type=click.Choice(['pbs', 'slurm']), default='pbs',
              help='HPC scheduler type')
@click.option('--output-dir', '-o', default=None,
              help='Output directory for scripts (default: simulation_dir/hpc)')
def hpc_generate(simulation_dir: str, settings_file: Optional[Path],
                 scheduler: str, output_dir: Optional[str]):
    """Generate HPC submission scripts for existing simulations.
    
    Scans simulation directory and creates PBS or SLURM job scripts.
    
    Example:
        FrictionSim2D hpc generate ./afm_output --scheduler pbs
    """
    from .hpc import HPCScriptGenerator, HPCConfig

    sim_dir = Path(simulation_dir)
    out_dir = Path(output_dir) if output_dir else sim_dir / 'hpc'
    out_dir.mkdir(parents=True, exist_ok=True)

    click.echo(f"🖥️  Generating {scheduler.upper()} scripts for: {sim_dir}")

    settings = load_settings(settings_file=settings_file)
    _ensure_hpc_settings(settings.hpc)
    hpc_config = HPCConfig.from_settings(settings.hpc)
    hpc_config.scheduler_type = scheduler  # type: ignore[assignment]

    simulation_paths = []
    script_names = hpc_config.lammps_scripts or ['system.in', 'slide.in']

    def script_exists(lammps_dir: Path, script_name: str) -> bool:
        if '*' in script_name or '?' in script_name:
            return any(lammps_dir.glob(script_name))
        if script_name == 'slide.in':
            return (
                (lammps_dir / 'slide.in').exists()
                or any(lammps_dir.glob('slide_*.in'))
            )
        return (lammps_dir / script_name).exists()

    for lammps_dir in sim_dir.rglob('lammps'):
        if not lammps_dir.is_dir():
            continue
        if not any(script_exists(lammps_dir, script) for script in script_names):
            continue
        rel_path = lammps_dir.parent.relative_to(sim_dir)
        simulation_paths.append(str(rel_path))

    simulation_paths = sorted(set(simulation_paths), key=layer_aware_path_sort_key)

    if not simulation_paths:
        click.echo("❌ No simulation directories found", err=True)
        raise click.Abort()

    click.echo(f"📋 Found {len(simulation_paths)} simulations")

    manifest_entries, resolved_scripts = _build_hpc_manifest_entries(
        sim_dir,
        simulation_paths,
        hpc_config.lammps_scripts,
    )
    hpc_config.lammps_scripts = resolved_scripts

    generator = HPCScriptGenerator(hpc_config)
    base_dir = '$PBS_O_WORKDIR' if scheduler == 'pbs' else '$SLURM_SUBMIT_DIR'
    result = dispatch_hpc_generation(
        generator,
        sim_dir,
        out_dir,
        scheduler,
        base_dir,
        manifest_entries,
    )

    if isinstance(result, dict):
        n_scripts = sum(len(v) for v in result.values())
        submit_file = 'submit_jobs.sh'
        click.echo(f"✅ Generated {n_scripts} two-phase script(s) in {out_dir}")
        click.echo("   (system → slide; slide jobs wait for system jobs to finish)")
    else:
        n_scripts = len(result)
        submit_file = 'submit_all.txt'
        click.echo(f"✅ Generated {n_scripts} script(s) in {out_dir}")
    click.echo("\n📝 Next steps:")
    click.echo(f"   1. Review scripts in {out_dir}")
    click.echo("   2. Transfer to HPC cluster")
    click.echo(f"   3. Follow {submit_file}")

# =============================================================================
# AIIDA COMMANDS (Optional)
# =============================================================================

@cli.group('aiida')
def aiida_group():
    """AiiDA workflow management (requires aiida-core)."""
    from .aiida import AIIDA_AVAILABLE
    if not AIIDA_AVAILABLE:
        click.echo("⚠️  AiiDA not available. Install with:", err=True)
        click.echo("   conda install -c conda-forge aiida-core", err=True)
        raise click.Abort()

@aiida_group.command('status')
def aiida_status():
    """Check AiiDA installation and profile status."""
    click.echo("✅ AiiDA is installed")
    
    try:
        from aiida.manage.configuration import load_profile  # type: ignore[import-untyped]
        profile = load_profile()
        click.echo(f"✅ Active profile: {profile.name}")
        click.echo(f"   Storage: {profile.storage_backend}")
    except Exception as exc:  # pylint: disable=broad-except
        click.echo(f"⚠️  No active profile: {exc}")
        click.echo("\n📝 Setup AiiDA with: verdi presto --use-postgres")

@aiida_group.command('import')
@click.argument('simulation_folder', type=click.Path(exists=True))
@click.option('--label', '-l', default=None,
              help='Required label for the simulation set (prompted if omitted)')
@click.option('--description', '-d', default='',
              help='Optional description for this simulation set')
@click.option('--profile', default=None,
              help='AiiDA profile name (default: configured default profile)')
def aiida_import(simulation_folder: str, label: Optional[str], description: str,
                 profile: Optional[str]):
    """Import a simulation set into AiiDA with full provenance.

    SIMULATION_FOLDER must be the ``simulation_XXXXXXXX`` directory produced
    by a single ``FrictionSim2D run`` invocation.  A label is required to
    identify this set in the database; you will be prompted if one is not
    supplied.

    Creates one ``FrictionSimulationSetData`` node (shared config), one
    ``FrictionProvenanceData`` + one ``FrictionSimulationData`` per material,
    and one ``FrictionResultsData`` per load/angle combination.

    Example:
        FrictionSim2D aiida import ./simulation_20260421_173546 --label "251125-sheetonsheet"
    """
    from .aiida.integration import import_simulation_set

    sim_path = Path(simulation_folder)

    if not label:
        label = click.prompt(
            'Label for this simulation set',
            default=sim_path.name,
        )
    label = (label or '').strip()
    if not label:
        _raise_abort("❌ Label cannot be empty.", ValueError("empty label"))

    click.echo(f"📥 Importing simulation set: {sim_path.name}")
    click.echo(f"   Label: {label}")
    if description:
        click.echo(f"   Description: {description}")

    try:
        set_uuid = import_simulation_set(
            simulation_folder=sim_path,
            label=label,
            description=description,
            aiida_profile=profile,
        )
        click.echo(f"✅ Import complete — set UUID: {set_uuid}")

    except Exception as exc:  # pylint: disable=broad-except
        logger.exception("Import failed")
        _raise_abort(f"❌ Import failed: {exc}", exc)


@aiida_group.command('list-sets')
@click.option('--profile', default=None,
              help='AiiDA profile name (default: configured default profile)')
@click.option('--format', 'output_format', type=click.Choice(['table', 'json', 'csv']),
              default='table', help='Output format')
def aiida_list_sets(profile: Optional[str], output_format: str):
    """List imported simulation sets stored in AiiDA.

    Example:
        FrictionSim2D aiida list-sets
        FrictionSim2D aiida list-sets --format json
    """
    from .aiida.integration import list_simulation_sets

    try:
        rows = list_simulation_sets(aiida_profile=profile)
    except Exception as exc:  # pylint: disable=broad-except
        logger.exception("Failed to list simulation sets")
        _raise_abort(f"❌ List failed: {exc}", exc)

    if not rows:
        click.echo("No simulation sets found in the current AiiDA profile.")
        return

    if output_format == 'json':
        import json
        click.echo(json.dumps(rows, indent=2))
        return

    if output_format == 'csv':
        import csv
        import io
        keys = [
            'label', 'simulation_type', 'n_materials', 'n_simulations',
            'n_results', 'run_folder', 'ctime', 'uuid', 'pk',
        ]
        buffer = io.StringIO()
        writer = csv.DictWriter(buffer, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
        click.echo(buffer.getvalue().rstrip())
        return

    headers = [
        ('Label', 'label'),
        ('Type', 'simulation_type'),
        ('Materials', 'n_materials'),
        ('Sims', 'n_simulations'),
        ('Results', 'n_results'),
        ('Run folder', 'run_folder'),
        ('UUID', 'uuid'),
    ]
    widths = []
    for title, key in headers:
        max_value_width = max(len(str(row.get(key, ''))) for row in rows)
        widths.append(max(len(title), max_value_width))

    header_line = '  '.join(title.ljust(widths[idx]) for idx, (title, _) in enumerate(headers))
    sep_line = '  '.join('-' * widths[idx] for idx in range(len(headers)))
    click.echo(header_line)
    click.echo(sep_line)
    for row in rows:
        click.echo(
            '  '.join(
                str(row.get(key, '')).ljust(widths[idx])
                for idx, (_, key) in enumerate(headers)
            )
        )
    click.echo(sep_line)
    click.echo(f"\nTotal sets: {len(rows)}")


@aiida_group.command('dump')
@click.argument('set_label')
@click.option('--output-dir', '-o', required=True, type=click.Path(),
              help='Directory to write output_full_*.json files into')
@click.option('--profile', default=None,
              help='AiiDA profile name (default: configured default profile)')
def aiida_dump(set_label: str, output_dir: str, profile: Optional[str]):
    """Export time-series data from an AiiDA set to output_full_*.json files.

    Creates <output_dir>/outputs/output_full_<size>.json in the same format
    as ``postprocess read --export``, so you can regenerate plots with
    ``postprocess plot`` against the AiiDA-exported data.

    Example:
        FrictionSim2D aiida dump 251113-afm --output-dir ~/Syncthing/aiida_dump/251113_afm
        FrictionSim2D aiida dump 260227-sevik_mos2 -o ~/Syncthing/aiida_dump/sevik_mos2
    """
    from .aiida import load_aiida_profile
    from .aiida.integration import dump_results_to_json

    load_aiida_profile(profile)

    click.echo(f"📤 Dumping set '{set_label}' → {output_dir}/outputs/")
    try:
        n = dump_results_to_json(
            set_label=set_label,
            output_dir=Path(output_dir),
            aiida_profile=profile,
        )
        click.echo(f"✅ Exported {n} simulation(s) to {output_dir}/outputs/")
    except Exception as exc:  # pylint: disable=broad-except
        logger.exception("Dump failed")
        _raise_abort(f"❌ Dump failed: {exc}", exc)


@aiida_group.command('rebuild')
@click.argument('set_label')
@click.option('--output-dir', '-o', default=None, type=click.Path(),
              help='Root directory to write the simulation tree into (default: cwd)')
@click.option('--hpc-scripts', is_flag=True, default=False,
              help='Generate HPC job-submission scripts after rebuilding')
@click.option('--profile', default=None,
              help='AiiDA profile name (default: configured default profile)')
def aiida_rebuild(set_label: str, output_dir: Optional[str], hpc_scripts: bool, profile: Optional[str]):
    """Rebuild LAMMPS input files for a set from its AiiDA provenance.

    Reads every FrictionProvenanceData node for the set, exports the stored
    CIF and potential files, and reruns the builder to produce a complete
    simulation directory tree.  This validates that the stored provenance is
    sufficient to reproduce the original simulation inputs.

    The reconstructed tree mirrors the layout produced by ``FrictionSim2D run``
    and can be submitted directly to an HPC cluster.

    Example:
        FrictionSim2D aiida rebuild 251125-sheetonsheet --output-dir ~/rebuild_test/
        FrictionSim2D aiida rebuild 251113-afm -o ~/rebuild_test/ --hpc-scripts
    """
    from .aiida import load_aiida_profile
    from .aiida.integration import rebuild_simulation_set

    load_aiida_profile(profile)

    click.echo(f"🔨 Rebuilding set '{set_label}' from AiiDA provenance…")
    try:
        created_dirs, simulation_root = rebuild_simulation_set(
            label_or_uuid=set_label,
            output_root=Path(output_dir) if output_dir else None,
            generate_hpc=hpc_scripts,
            aiida_profile=profile,
        )
        click.echo(
            f"✅ Rebuilt {len(created_dirs)} simulation(s) → {simulation_root}"
        )
    except Exception as exc:  # pylint: disable=broad-except
        logger.exception("Rebuild failed")
        _raise_abort(f"❌ Rebuild failed: {exc}", exc)


@aiida_group.command('delete')
@click.argument('set_label')
@click.option('--profile', default=None,
              help='AiiDA profile name (default: configured default profile)')
@click.confirmation_option(
    prompt='This will permanently delete the simulation set and all linked nodes. Continue?'
)
def aiida_delete(set_label: str, profile: Optional[str]):
    """Delete a single simulation set and all its linked nodes.

    SET_LABEL is the label (or UUID) of the set to delete. All linked
    SimulationData, ResultsData, and ProvenanceData nodes are also removed.

    Example:
        FrictionSim2D aiida delete 260312-force_rebomos
    """
    from .aiida.integration import delete_simulation_set

    click.echo(f"🗑️  Deleting simulation set '{set_label}'…")
    try:
        n = delete_simulation_set(label_or_uuid=set_label, aiida_profile=profile)
        click.echo(f"✅ Deleted {n} nodes for set '{set_label}'.")
    except Exception as exc:  # pylint: disable=broad-except
        logger.exception("Delete failed")
        _raise_abort(f"❌ Delete failed: {exc}", exc)


@aiida_group.command('clear')
@click.option('--profile', default=None,
              help='AiiDA profile name (default: configured default profile)')
@click.confirmation_option(
    prompt='This will permanently delete all FrictionSim2D nodes. Continue?'
)
def aiida_clear(profile: Optional[str]):
    """Delete all FrictionSim2D nodes from the AiiDA database.

    This removes every SimulationSet, Simulation, Provenance, and Results
    node.  The action is irreversible.

    Example:
        FrictionSim2D aiida clear
    """
    from .aiida.integration import clear_all_nodes

    click.echo("🗑️  Deleting all FrictionSim2D nodes…")
    try:
        n = clear_all_nodes(aiida_profile=profile)
        click.echo(f"✅ Deleted {n} nodes.")
    except Exception as exc:  # pylint: disable=broad-except
        logger.exception("Clear failed")
        _raise_abort(f"❌ Clear failed: {exc}", exc)


@aiida_group.command('query')
@click.option('--material', '-m', help='Filter by material')
@click.option('--layers', '-l', type=int, help='Filter by layer count')
@click.option('--force', '-f', type=float, help='Filter by applied force')
@click.option('--set', '-s', 'set_label', default=None, help='Filter by simulation set label (e.g. "251113-afm")')
@click.option('--format', 'output_format', type=click.Choice(['table', 'csv', 'json']),
              default='table', help='Output format')
@click.option('--output', '-o', type=click.Path(), help='Save to file')
@click.option('--profile', default=None,
                            help='AiiDA profile name (default: configured default profile)')
def aiida_query(material: Optional[str], layers: Optional[int], force: Optional[float],
                                set_label: Optional[str], output_format: str,
                                output: Optional[str], profile: Optional[str]):
    """Query simulation database.
    
    Example:
        FrictionSim2D aiida query --material h-MoS2 --layers 2 --format csv
        FrictionSim2D aiida query --set 251113-afm --format table
    """
    from .aiida import load_aiida_profile
    from .aiida.query import Friction2DDB

    load_aiida_profile(profile)
    
    db = Friction2DDB()

    if set_label:
        from .aiida.integration import list_simulation_sets
        sets = list_simulation_sets(aiida_profile=profile)
        matched = [s for s in sets if s['label'] == set_label]
        if not matched:
            available = [s['label'] for s in sets]
            raise click.ClickException(
                f"No set found with label '{set_label}'. Available: {available}"
            )
        set_uuid = matched[0]['uuid']
        click.echo(f"🔍 Querying set '{set_label}' (UUID: {set_uuid[:8]}…)")
        results = db.query_by_set(set_uuid)
        click.echo(f"📊 Found {results.total_count} results")
    else:
        query_kwargs = {
            'materials': [material] if material else None,
            'layers': layers,
            'force_range': (force, force) if force is not None else None,
        }
        # Remove None values before forwarding to Friction2DDB.query
        query_kwargs = {k: v for k, v in query_kwargs.items() if v is not None}
        click.echo(f"🔍 Querying database with filters: {query_kwargs}")
        try:
            results = db.query(**query_kwargs)
            click.echo(f"📊 Found {results.total_count} results")
        except Exception as exc:  # pylint: disable=broad-except
            logger.exception("AiiDA query failed")
            _raise_abort(f"❌ Query failed: {exc}", exc)

    try:
        if output_format == 'table':
            df = results.to_dataframe()
            click.echo("\n" + df.to_string(index=False))
        elif output_format == 'csv':
            df = results.to_dataframe()
            if output:
                df.to_csv(output, index=False)
                click.echo(f"✅ Saved to {output}")
            else:
                click.echo(df.to_csv(index=False))
        elif output_format == 'json':
            if output:
                results.export_json(Path(output))
                click.echo(f"✅ Saved to {output}")
            else:
                import json
                click.echo(json.dumps(results.query_params, indent=2))
                
    except Exception as exc:  # pylint: disable=broad-except
        logger.exception("AiiDA query failed")
        _raise_abort(f"❌ Query failed: {exc}", exc)


@aiida_group.command('setup')
@click.option('--profile', '-p', default=None,
              help='AiiDA profile name (default: auto-created)')
@click.option('--lammps-path', type=click.Path(),
              help='Explicit path to LAMMPS executable')
@click.option('--hpc-config', type=click.Path(exists=True),
              help='(Deprecated) Path to hpc.yaml for PBS/SLURM computer setup')
@click.option('--use-remote', is_flag=True,
              help='Configure remote HPC computer using settings from settings.yaml')
@click.option('--settings-file', type=click.Path(exists=True, dir_okay=False, path_type=Path),
                            default=None,
                            help='Path to a settings.yaml file used with --use-remote.')
def aiida_setup(profile: Optional[str], lammps_path: Optional[str],
                                hpc_config: Optional[str], use_remote: bool,
                                settings_file: Optional[Path]):
    """Set up AiiDA profile, computer, and LAMMPS code.

    Performs first-time AiiDA configuration:
    - Creates a profile (via verdi presto)
    - Registers localhost or remote HPC as a computer
    - Configures the LAMMPS code

    For remote HPC setup, configure the AiiDA section in settings.yaml with
    hostname, username, workdir, etc., then use --use-remote flag.

    Example::

        FrictionSim2D aiida setup
        FrictionSim2D aiida setup --lammps-path /usr/local/bin/lmp_mpi
        FrictionSim2D aiida setup --use-remote  # Uses settings.yaml AiiDA config
    """
    from .aiida.setup import full_setup

    click.echo("🔧 Running AiiDA first-time setup ...")

    # Load HPC settings if using remote
    hpc_settings = None
    aiida_settings = None
    if use_remote:
        settings = load_settings(settings_file=settings_file)
        hpc_settings = settings.hpc
        aiida_settings = settings.aiida
        click.echo(f"📡 Configuring remote HPC: {aiida_settings.hostname or 'localhost'}")

    if hpc_config:
        click.echo("⚠️  --hpc-config is deprecated. Use --use-remote with settings.yaml instead.")

    try:
        full_setup(
            profile_name=profile or 'friction2d',
            lammps_executable=lammps_path,
            hpc_settings=hpc_settings,
            aiida_settings=aiida_settings,
        )
        click.echo("✅ AiiDA setup complete")
    except Exception as exc:  # pylint: disable=broad-except
        logger.exception("AiiDA setup failed")
        _raise_abort(f"❌ Setup failed: {exc}", exc)


@aiida_group.command('export')
@click.option('--output', '-o', type=click.Path(), default='friction2d.aiida',
              help='Output archive path')
@click.option('--material', '-m', default=None,
              help='Export only simulations for a given material')
def aiida_export(output: str, material: Optional[str]):
    """Export AiiDA database to a portable archive file.

    The archive can be transferred to another machine and imported
    with ``FrictionSim2D aiida import-archive``.

    Example::

        FrictionSim2D aiida export -o results.aiida
        FrictionSim2D aiida export -m h-MoS2
    """
    from .aiida.integration import export_archive

    click.echo(f"📦 Exporting archive to {output} ...")
    try:
        export_archive(Path(output), materials=[material] if material else None)
        click.echo(f"✅ Archive saved: {output}")
    except Exception as exc:  # pylint: disable=broad-except
        logger.exception("AiiDA export failed")
        _raise_abort(f"❌ Export failed: {exc}", exc)


@aiida_group.command('import-archive')
@click.argument('archive_path', type=click.Path(exists=True))
def aiida_import_archive(archive_path: str):
    """Import an AiiDA archive into the current profile.

    Use this to set up a mirror database on a local workstation from
    results exported on an HPC cluster.

    Example::

        FrictionSim2D aiida import-archive results.aiida
    """
    from .aiida import load_aiida_profile
    from .aiida.integration import import_archive

    load_aiida_profile()

    click.echo(f"📥 Importing archive: {archive_path} ...")
    try:
        import_archive(Path(archive_path))
        click.echo("✅ Archive imported successfully")
    except Exception as exc:  # pylint: disable=broad-except
        logger.exception("AiiDA import-archive failed")
        _raise_abort(f"❌ Import failed: {exc}", exc)


@aiida_group.command('package')
@click.argument('simulation_dir', type=click.Path(exists=True))
@click.option('--output', '-o', type=click.Path(), default=None,
              help='Output archive path (.tar.gz)')
def aiida_package(simulation_dir: str, output: Optional[str]):
    """Create a tar.gz archive of simulation inputs for transfer."""
    import tarfile  # noqa: PLC0415
    sim_path = Path(simulation_dir)
    out_path = Path(output) if output else sim_path.with_suffix('.tar.gz')

    def _tar_filter(info: tarfile.TarInfo) -> Optional[tarfile.TarInfo]:
        if info.name.endswith('.lammpstrj'):
            return None
        return info

    click.echo(f"📦 Packing {sim_path} → {out_path}")
    with tarfile.open(out_path, 'w:gz') as tar:
        tar.add(sim_path, arcname=sim_path.name, filter=_tar_filter)
    click.echo(f"✅ Archive created: {out_path}")


@aiida_group.command('submit')
@click.argument('simulation_dir', type=click.Path(exists=True))
@click.option('--settings-file', type=click.Path(exists=True, dir_okay=False, path_type=Path),
              default=None,
              help='Path to a settings.yaml file for this submission.')
@click.option('--code', '-c', default=None,
              help='AiiDA code label (auto-detects if not specified)')
@click.option('--scripts', default=None,
              help='Comma-separated list of LAMMPS scripts to run in order')
@click.option('--array', 'use_array', is_flag=True,
              help='Submit a single array job for all simulations')
@click.option('--machines', type=int, default=None,
              help='Override: Number of machines per job')
@click.option('--mpiprocs', type=int, default=None,
              help='Override: MPI processes per machine')
@click.option('--walltime', type=str, default=None,
              help='Override: Walltime (HH:MM:SS or hours)')
@click.option('--queue', type=str, default=None,
              help='Override: Scheduler queue name')
@click.option('--project', type=str, default=None,
              help='Override: Scheduler project/account')
@click.option('--dry-run', is_flag=True,
              help='Preview configuration without submitting')
def aiida_submit(
    simulation_dir: str,
    settings_file: Optional[Path],
    code: Optional[str],
    scripts: Optional[str],
    use_array: bool,
    machines: Optional[int],
    mpiprocs: Optional[int],
    walltime: Optional[str],
    queue: Optional[str],
    project: Optional[str],
    dry_run: bool,
):
    """Submit simulations to AiiDA with smart defaults and prompting.

    Automatically detects codes, uses settings.yaml defaults,
    and prompts for essential missing values. Manual overrides are supported.

    Examples:

    # Minimal usage (auto-detect code, use defaults)
      FrictionSim2D aiida submit ./afm_output

      # With manual overrides
      FrictionSim2D aiida submit ./output --machines 4 --walltime 24:00:00

      # Preview before submitting
      FrictionSim2D aiida submit ./output --dry-run
    """
    from .aiida.submit import smart_submit

    sim_path = Path(simulation_dir)
    click.echo(f"🚀 Preparing submission from {sim_path}\n")

    try:
        # Collect manual overrides
        overrides = {}
        if machines is not None:
            overrides['machines'] = machines
        if mpiprocs is not None:
            overrides['mpiprocs'] = mpiprocs
        if walltime is not None:
            overrides['walltime'] = walltime
        if queue is not None:
            overrides['queue'] = queue
        if project is not None:
            overrides['project'] = project

        # Call smart submit helper
        processes = smart_submit(
            simulation_dir=sim_path,
            settings_file=settings_file,
            code_label=code,
            use_array=use_array,
            scripts=scripts,
            dry_run=dry_run,
            **overrides,
        )

        if not dry_run and processes:
            click.echo(f"\n✅ Successfully submitted {len(processes)} job(s)")

    except Exception as exc:  # pylint: disable=broad-except
        logger.exception("AiiDA submit failed")
        _raise_abort(f"❌ Submission failed: {exc}", exc)

# =============================================================================
# POSTPROCESSING COMMANDS
# =============================================================================

@cli.group('postprocess')
def postprocess_group():
    """Read, analyse and plot simulation results."""


@postprocess_group.command('read')
@click.argument('results_dir', type=click.Path(exists=True))
@click.option('--export', 'do_export', is_flag=True,
              help='Export full time-series data to JSON.')
def postprocess_read(results_dir: str, do_export: bool):
    """Read and process simulation result data."""
    from .postprocessing import DataReader  # noqa: PLC0415

    reader = DataReader(results_dir=results_dir)
    reader.export_issue_reports()
    if do_export:
        reader.export_full_data_to_json()
    click.echo("Postprocessing complete.")


@postprocess_group.command('plot')
@click.argument('plot_config', type=click.Path(exists=True))
@click.option('--output-dir', '-o', default='plots',
              help='Output directory for plots.')
@click.option('--settings', type=click.Path(exists=True),
              help='Plot settings JSON file.')
def postprocess_plot(plot_config: str, output_dir: str,
                     settings: Optional[str]):
    """Generate plots from processed simulation data."""
    import json  # noqa: PLC0415
    from .postprocessing import Plotter  # noqa: PLC0415

    with open(plot_config, 'r', encoding='utf-8') as f:
        config = json.load(f)

    plot_settings = None
    if settings:
        with open(settings, 'r', encoding='utf-8') as f:
            plot_settings = json.load(f)

    data_dirs = config.get('data_dirs', [])
    labels = config.get('labels', [])
    plots = config.get('plots', [])
    dataset_display_labels = config.get('dataset_display_labels', {})
    series_color_map = config.get('series_color_map', {})

    if not data_dirs or not labels or not plots:
        raise click.Abort(
            "'data_dirs', 'labels', and 'plots' must be defined in config."
        )

    if len(data_dirs) != len(labels):
        raise click.Abort(
            "Number of 'data_dirs' must match number of 'labels'."
        )

    plotter = Plotter(
        data_dirs,
        labels,
        output_dir,
        plot_settings,
        dataset_display_labels=dataset_display_labels,
        series_color_map=series_color_map,
    )
    for plot_cfg in plots:
        plotter.generate_plot(plot_cfg)
    click.echo("All plots generated.")


@postprocess_group.command('pes-scan')
@click.argument('run_root', type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option('--data-dir', '-d', type=click.Path(path_type=Path), default='data',
              help='Directory for the per-material descriptor CSVs (pes_scan_{sheet,tip}.csv).')
@click.option('--grid-dir', type=click.Path(path_type=Path), default=None,
              help='Optional directory for tidied per-material grids (<grid-dir>/{sheet,tip}/<material>.csv).')
@click.option('--kind', type=click.Choice(['sheet', 'tip', 'both']), default='both',
              help='Which PES scans to reduce.')
def postprocess_pes_scan(run_root: Path, data_dir: Path, grid_dir: Optional[Path], kind: str):
    """Reduce PES-scan energy/force grids to per-material descriptor CSVs.

    Scans RUN_ROOT for pes_sheet/ and pes_tip/ results and writes
    data/pes_scan_sheet.csv and data/pes_scan_tip.csv (merge key 'material').

    Example:
        FrictionSim2D postprocess pes-scan ./simulation_output/simulation_YYYYMMDD_HHMMSS \\
            --data-dir data --grid-dir outputs/pes_scan
    """
    from .postprocessing import pes_scan as pes  # noqa: PLC0415

    kinds = ['sheet', 'tip'] if kind == 'both' else [kind]
    written = pes.run(run_root, data_dir, grid_dir=grid_dir, kinds=kinds)
    if not written:
        click.echo(f"No PES-scan grids found under {run_root}.", err=True)
        return
    for k, path in written.items():
        click.echo(f"✅ {k}: wrote descriptors -> {path.resolve()}")
    if grid_dir is not None:
        click.echo(f"📂 Grids copied under {Path(grid_dir).resolve()}/{{sheet,tip}}/")


@postprocess_group.command('indent')
@click.argument('run_root', type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option('--out-dir', '-o', type=click.Path(path_type=Path), default='outputs/poke_sims',
              help='Destination for output_indent_f*_s*.json, indent/<material>_l<layer>.csv '
                   'and indent_manifest.csv.')
@click.option('--data-dir', '-d', type=click.Path(path_type=Path), default=None,
              help='Destination for indent_features.csv (default: same as --out-dir).')
def postprocess_indent(run_root: Path, out_dir: Path, data_dir: Optional[Path]):
    """Reduce an indent run to penetration JSON + adhesion features + a manifest.

    Delivers both descriptor sets from the one hold+retract run:
      - penetration: output_indent_f<load>_s<seed>.json per load (nested campaign
        schema, read by the ML side exactly like the sliding campaign);
      - adhesion: indent_features.csv (pull-off force, work of adhesion, contact
        stiffness, keyed material/layer/load) + tidied per-condition retract
        curves indent/<material>_l<layer>.csv;
      - indent_manifest.csv (material, layer, load, seed, hold_steps, z_step,
        n_steps, wall_time).

    Example:
        FrictionSim2D postprocess indent ./indent_output/simulation_YYYYMMDD_HHMMSS \\
            --out-dir outputs/poke_sims --data-dir data
    """
    from .postprocessing import indent as indent_pp  # noqa: PLC0415

    written = indent_pp.run(run_root, out_dir, data_dir=data_dir)
    if not written:
        click.echo(f"No indent results found under {run_root}.", err=True)
        return
    for name, path in written.items():
        click.echo(f"✅ {name} -> {path.resolve()}")


# =============================================================================
# DATABASE COMMANDS
# =============================================================================

@cli.group('db')
def db_group():
    """Interact with the FrictionSim2D PostgreSQL database.

    Upload your simulation results to the community database and query
    results contributed by other users.

    Connection settings are resolved in order of precedence:

    \b
      1. Explicit CLI flags (--host, --port, etc.)
      2. Environment variables (FRICTION_DB_HOST, FRICTION_DB_PORT, ...)
      3. The active database profile in settings.yaml

    Use ``--profile local`` or ``--profile central`` to switch profiles.
    """


def _make_db(
    host: Optional[str],
    port: Optional[int],
    dbname: Optional[str],
    user: Optional[str],
    password: Optional[str],
    profile: Optional[str] = None,
):
    """Instantiate a :class:`~src.data.database.FrictionDB`.

    If none of the explicit connection parameters are provided, falls back
    to the settings.yaml database profile.
    """
    has_explicit = any(v is not None for v in (host, port, dbname, user, password))
    if not has_explicit:
        try:
            from .data.database import db_from_profile  # noqa: PLC0415
            return db_from_profile(profile)
        except Exception:  # pylint: disable=broad-except  # settings not available
            pass
    from .data.database import FrictionDB  # noqa: PLC0415
    return FrictionDB(host=host, port=port, dbname=dbname, user=user, password=password)


_DB_OPTIONS = [
    click.option('--profile', '-p', default=None,
                 help="Database profile from settings.yaml ('local' or 'central')."),
    click.option('--host', default=None, envvar='FRICTION_DB_HOST',
                 help='Database hostname (default: $FRICTION_DB_HOST or localhost)'),
    click.option('--port', default=None, type=int, envvar='FRICTION_DB_PORT',
                 help='Database port (default: $FRICTION_DB_PORT or 5432)'),
    click.option('--dbname', default=None, envvar='FRICTION_DB_NAME',
                 help='Database name (default: $FRICTION_DB_NAME or frictionsim2ddb)'),
    click.option('--user', '-u', default=None, envvar='FRICTION_DB_USER',
                 help='Database username (default: $FRICTION_DB_USER)'),
    click.option('--password', default=None, envvar='FRICTION_DB_PASSWORD',
                 help='Database password (default: $FRICTION_DB_PASSWORD)'),
]


def _add_db_options(func):
    for opt in reversed(_DB_OPTIONS):
        func = opt(func)
    return func


def _iter_reader_rows(reader):
    """Yield one result row per simulation from a DataReader, with statistics."""
    import numpy as np  # noqa: PLC0415
    from .data.models import compute_friction_stats  # noqa: PLC0415
    for material, size_data in reader.full_data_nested.items():
        for _size_key, substrate_data in size_data.items():
            for _sub, tip_data in substrate_data.items():
                for _tip_mat, radius_data in tip_data.items():
                    for _radius, layer_data in radius_data.items():
                        for layer_key, speed_data in layer_data.items():
                            layers = int(layer_key.replace('l', ''))
                            for speed_key, force_data in speed_data.items():
                                speed = int(speed_key.replace('s', ''))
                                for load_key, angle_data in force_data.items():
                                    is_pressure = load_key.startswith('p')
                                    load_val = float(load_key[1:])
                                    for angle_key, df in angle_data.items():
                                        angle = int(angle_key.replace('a', ''))
                                        nf_arr = df['nf'].values if 'nf' in df.columns else np.array([])
                                        lfx_arr = df['lfx'].values if 'lfx' in df.columns else np.array([])
                                        lfy_arr = df['lfy'].values if 'lfy' in df.columns else np.array([])
                                        stats = compute_friction_stats(nf_arr, lfx_arr, lfy_arr) if nf_arr.size > 0 else {}
                                        yield material, layers, speed, is_pressure, load_val, angle, len(df), stats


@db_group.command('upload')
@click.argument('results_dir', type=click.Path(exists=True))
@click.option('--uploader', '-n', default=None,
              help='Your name or identifier (stored with each row).')
@_add_db_options
def db_upload(
    results_dir: str,
    uploader: Optional[str],
    profile: Optional[str],
    host: Optional[str],
    port: Optional[int],
    dbname: Optional[str],
    user: Optional[str],
    password: Optional[str],
):
    """Upload simulation results from RESULTS_DIR to the shared database.

    Scans RESULTS_DIR for completed simulation outputs and uploads each
    result (mean COF, forces, conditions) to the community PostgreSQL
    database.

    Example:
        FrictionSim2D db upload ./simulation_output/afm_run_1/results \\
            --uploader alice
    """
    from .postprocessing.read_data import DataReader  # noqa: PLC0415

    click.echo(f"📂 Reading results from {results_dir} ...")
    reader = DataReader(results_dir=str(results_dir))
    db = _make_db(host, port, dbname, user, password, profile=profile)
    n_published = 0
    n_rejected = 0
    for material, layers, speed, is_pressure, load_val, angle, ntimesteps, stats in _iter_reader_rows(reader):
        try:
            sim_type = 'sheetonsheet' if is_pressure else 'afm'
            row_id = db.upload_result(
                material=material.replace('_', '-'),
                simulation_type=sim_type,
                layers=layers,
                force_nN=None if is_pressure else load_val,
                pressure_gpa=load_val if is_pressure else None,
                scan_angle=float(angle),
                scan_speed=float(speed),
                uploader=uploader,
                ntimesteps=ntimesteps,
                **stats,  # type: ignore[arg-type]
            )
            vr = db.validate_staged(row_id)
            if vr.is_valid and db.publish(row_id):
                n_published += 1
            else:
                n_rejected += 1
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning("Skipped row (%s, l%d, a%d): %s", material, layers, angle, exc)

    click.echo(
        f"✅ Processed results: published={n_published}, rejected={n_rejected}."
    )


@db_group.command('query')
@click.option('--material', '-m', default=None, help='Filter by material name.')
@click.option('--type', 'sim_type', default=None,
              help='Filter by simulation type (afm or sheetonsheet).')
@click.option('--layers', '-l', type=int, default=None,
              help='Filter by number of layers.')
@click.option('--uploader', '-n', default=None, help='Filter by uploader name.')
@click.option('--limit', type=int, default=50, show_default=True,
              help='Maximum number of rows to return.')
@click.option('--csv', 'output_csv', default=None,
              help='Save results to a CSV file.')
@_add_db_options
def db_query(
    material: Optional[str],
    sim_type: Optional[str],
    layers: Optional[int],
    uploader: Optional[str],
    limit: int,
    output_csv: Optional[str],
    profile: Optional[str],
    host: Optional[str],
    port: Optional[int],
    dbname: Optional[str],
    user: Optional[str],
    password: Optional[str],
):
    """Query the shared database and print matching results.

    Example:
        FrictionSim2D db query --material h-MoS2 --layers 1 --limit 20

        FrictionSim2D db query --material h-WS2 --csv results.csv
    """
    db = _make_db(host, port, dbname, user, password, profile=profile)
    df = db.query(
        material=material,
        simulation_type=sim_type,
        layers=layers,
        uploader=uploader,
        limit=limit,
    )

    if df.empty:
        click.echo("No results found.")
        return

    click.echo(df.to_string(index=False))
    click.echo(f"\n{len(df)} row(s) returned.")

    if output_csv:
        df.to_csv(output_csv, index=False)
        click.echo(f"💾 Saved to {output_csv}")


@db_group.command('stats')
@_add_db_options
def db_stats(
    profile: Optional[str],
    host: Optional[str],
    port: Optional[int],
    dbname: Optional[str],
    user: Optional[str],
    password: Optional[str],
):
    """Show aggregate statistics for the shared database."""
    db = _make_db(host, port, dbname, user, password, profile=profile)
    stats = db.get_statistics()

    click.echo(f"Total simulations : {stats['total_rows']}")
    click.echo(f"Global mean COF   : {stats['cof_global_mean']:.4f}"
               if stats['cof_global_mean'] is not None else "Global mean COF   : n/a")
    click.echo("\nBy material:")
    for mat, count in stats['by_material'].items():
        click.echo(f"  {mat:<20} {count}")
    click.echo("\nBy simulation type:")
    for stype, count in stats['by_type'].items():
        click.echo(f"  {stype:<20} {count}")


@db_group.command('delete')
@click.option('--uploader', '-n', required=True,
              help='Delete all rows belonging to this uploader.')
@click.confirmation_option(prompt='Are you sure you want to delete these rows?')
@_add_db_options
def db_delete(
    uploader: str,
    profile: Optional[str],
    host: Optional[str],
    port: Optional[int],
    dbname: Optional[str],
    user: Optional[str],
    password: Optional[str],
):
    """Delete your uploaded results from the shared database.

    Only rows with a matching uploader name are deleted.

    Example:
        FrictionSim2D db delete --uploader alice
    """
    db = _make_db(host, port, dbname, user, password, profile=profile)
    count = db.delete_own_results(uploader)
    click.echo(f"🗑  Deleted {count} row(s) for uploader '{uploader}'.")


@db_group.command('init')
@_add_db_options
def db_init(
    profile: Optional[str],
    host: Optional[str],
    port: Optional[int],
    dbname: Optional[str],
    user: Optional[str],
    password: Optional[str],
):
    """Initialise the database schema (create/verify current tables).

    Safe to run repeatedly — existing tables and columns are not overwritten.

    Example:\n
        FrictionSim2D db init --profile local
    """
    _make_db(host, port, dbname, user, password, profile=profile)
    from .data.database import SCHEMA_VERSION  # noqa: PLC0415
    click.echo(f"✅ Database initialised at schema version {SCHEMA_VERSION}.")


@db_group.command('create-key')
@click.option('--name', '-n', required=True,
              help='User name to associate with the API key.')
@_add_db_options
def db_create_key(
    name: str,
    profile: Optional[str],
    host: Optional[str],
    port: Optional[int],
    dbname: Optional[str],
    user: Optional[str],
    password: Optional[str],
):
    """Generate a new API key for write access to the database.

    The raw API key is printed ONCE.  Store it securely — it cannot be
    retrieved again (only revoked).

    Example:\n
        FrictionSim2D db create-key --name alice
    """
    db = _make_db(host, port, dbname, user, password, profile=profile)
    raw_key = db.create_api_key(name)
    click.echo(f"🔑 API key for '{name}':")
    click.echo(f"   {raw_key}")
    click.echo("\nStore this key securely. It cannot be shown again.")


@db_group.command('stage')
@click.argument('results_dir', type=click.Path(exists=True))
@click.option('--uploader', '-n', required=True,
              help='Your name or identifier.')
@click.option('--api-key', default=None, envvar='FRICTION_DB_API_KEY',
              help='API key for write access.')
@_add_db_options
def db_stage(
    results_dir: str,
    uploader: str,
    api_key: Optional[str],
    profile: Optional[str],
    host: Optional[str],
    port: Optional[int],
    dbname: Optional[str],
    user: Optional[str],
    password: Optional[str],
):
    """Stage simulation results and apply automatic validation.

    Results are uploaded with status='staged', validated, and automatically
    published when they pass validation.

    Example:\n
        FrictionSim2D db stage ./results --uploader alice --api-key <key>
    """
    db = _make_db(host, port, dbname, user, password, profile=profile)

    if api_key:
        verified_user = db.verify_api_key(api_key)
        if not verified_user:
            raise click.ClickException("Invalid or revoked API key.")
        click.echo(f"Authenticated as: {verified_user}")

    from .postprocessing.read_data import DataReader  # noqa: PLC0415

    click.echo(f"📂 Reading results from {results_dir} ...")
    reader = DataReader(results_dir=str(results_dir))
    n_published = 0
    n_rejected = 0
    for material, layers, speed, is_pressure, load_val, angle, ntimesteps, stats in _iter_reader_rows(reader):
        try:
            sim_type = 'sheetonsheet' if is_pressure else 'afm'
            row_id = db.upload_result(
                material=material.replace('_', '-'),
                simulation_type=sim_type,
                layers=layers,
                force_nN=None if is_pressure else load_val,
                pressure_gpa=load_val if is_pressure else None,
                scan_angle=float(angle),
                scan_speed=float(speed),
                uploader=uploader,
                ntimesteps=ntimesteps,
                **stats,  # type: ignore[arg-type]
            )
            vr = db.validate_staged(row_id)
            if vr.is_valid and db.publish(row_id):
                n_published += 1
            else:
                n_rejected += 1
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning("Skipped row (%s, l%d, a%d): %s", material, layers, angle, exc)

    click.echo(
        f"✅ Processed staged results: published={n_published}, rejected={n_rejected}."
    )


@db_group.command('publish')
@click.argument('row_id', type=int)
@_add_db_options
def db_publish(
    row_id: int,
    profile: Optional[str],
    host: Optional[str],
    port: Optional[int],
    dbname: Optional[str],
    user: Optional[str],
    password: Optional[str],
):
    """Publish a validated result (curator action).

    Promotes a result from 'validated' to 'published', making it visible
    to all users.

    Example:\n
        FrictionSim2D db publish 42
    """
    db = _make_db(host, port, dbname, user, password, profile=profile)
    ok = db.publish(row_id)
    if ok:
        click.echo(f"✅ Result {row_id} published.")
    else:
        raise click.ClickException(
            f"Could not publish result {row_id}. "
            "It may not exist or may not be in 'validated' status."
        )


@db_group.command('reject')
@click.argument('row_id', type=int)
@click.option('--reason', '-r', default=None, help='Reason for rejection.')
@_add_db_options
def db_reject(
    row_id: int,
    reason: Optional[str],
    profile: Optional[str],
    host: Optional[str],
    port: Optional[int],
    dbname: Optional[str],
    user: Optional[str],
    password: Optional[str],
):
    """Reject a staged or validated result (curator action).

    Example:\n
        FrictionSim2D db reject 42 --reason "Duplicate of row 37"
    """
    db = _make_db(host, port, dbname, user, password, profile=profile)
    ok = db.reject(row_id, reason=reason)
    if ok:
        click.echo(f"❌ Result {row_id} rejected.")
    else:
        raise click.ClickException(
            f"Could not reject result {row_id}. It may not exist."
        )


@db_group.command('setup')
@click.option('--name', '-n', default=None,
              help='User/uploader name for API key (default: $USER or "user")')
@_add_db_options
def db_setup(
    name: Optional[str],
    profile: str,
    host: Optional[str],
    port: Optional[int],
    dbname: Optional[str],
    user: Optional[str],
    password: Optional[str],
):
    """First-time setup: initialize database and generate API key.

    This command:
    1. Initializes the database schema (safe to run repeatedly)
    2. Creates an API key for your user
    3. Stores the API key in ~/.config/FrictionSim2D/settings.yaml

    Example:
        FrictionSim2D db setup                              # local setup (default profile)
        FrictionSim2D db setup --name alice --profile central
    """
    import os as os_module  # noqa: PLC0415
    import yaml  # noqa: PLC0415
    from pathlib import Path as PathlibPath  # noqa: PLC0415

    if not name:
        name = os_module.environ.get('USER', os_module.environ.get('USERNAME', 'user'))

    click.echo("📊 Initializing database schema...")
    db = _make_db(host, port, dbname, user, password, profile=profile)
    from .data.database import SCHEMA_VERSION  # noqa: PLC0415
    click.echo(f"✅ Database initialised at schema version {SCHEMA_VERSION}.")

    click.echo(f"🔑 Creating API key for user '{name}'...")
    raw_key = db.create_api_key(name)
    click.echo(f"✅ API key generated: {raw_key}")

    click.echo("💾 Storing API key in settings...")
    settings_dir = PathlibPath.home() / ".config" / "FrictionSim2D"
    settings_file = settings_dir / "settings.yaml"
    settings_dir.mkdir(parents=True, exist_ok=True)

    if settings_file.exists():
        settings_data = yaml.safe_load(settings_file.read_text(encoding='utf-8'))
        if settings_data is None:
            settings_data = {}
    else:
        effective = load_settings()
        settings_data = effective.model_dump()

    if 'database' not in settings_data:
        settings_data['database'] = {}
    if profile not in settings_data['database']:
        settings_data['database'][profile] = {}

    settings_data['database'][profile]['api_key'] = raw_key

    settings_file.write_text(
        yaml.safe_dump(settings_data, sort_keys=False),
        encoding='utf-8',
    )
    click.echo(f"✅ API key saved to {settings_file}")
    click.echo()
    click.echo("📋 Next steps:")
    click.echo(f"   1. You can now upload results: FrictionSim2D db upload <dir> --uploader {name} --profile {profile}")
    click.echo(f"   2. Or stage for review: FrictionSim2D db stage <dir> --uploader {name} --profile {profile}")
    click.echo()
    click.echo("⚠️  Your API key is stored in settings.yaml. Keep it secure!")


# =============================================================================
# API SERVER COMMAND
# =============================================================================

@cli.group('api')
def api_group():
    """Run the FrictionSim2D REST API server."""


@api_group.command('serve')
@click.option('--host', default=None,
              help='Bind host (default: settings.database.api_host or 0.0.0.0)')
@click.option('--port', '-p', default=None, type=int,
              help='Port (default: settings.database.api_port or 8000)')
@click.option('--profile', default=None,
              help="Database profile to back the server ('local' or 'central').")
@click.option('--reload', is_flag=True,
              help='Auto-reload on code changes (development only).')
@click.option('--settings-file', type=click.Path(exists=True, dir_okay=False, path_type=Path),
              default=None,
              help='Path to a settings.yaml file for this command.')
def api_serve(host: Optional[str], port: Optional[int],
              profile: Optional[str], reload: bool,
              settings_file: Optional[Path]):
    """Start the REST API server.

    Runs the FastAPI application so collaborators can upload and query
    results without needing direct database credentials.  They only need
    the server URL and a personal API key.

    \b
    Server setup (run once on your machine):
        FrictionSim2D db init --profile local
        FrictionSim2D db create-key --name alice --profile local
        FrictionSim2D api serve --host 0.0.0.0 --port 8000

    \b
    Collaborator usage:
        Query via browser or scripts at http://your-server:8000/docs
        (write operations are available through authenticated API endpoints)
    """
    try:
        import uvicorn  # pylint: disable=import-outside-toplevel
    except ImportError as exc:
        _raise_abort(
            "❌ uvicorn is required to run the API server.\n"
            "   Install with: pip install uvicorn[standard]",
            exc,
        )

    settings = load_settings(settings_file=settings_file)
    db_cfg = settings.database
    _host = host or getattr(db_cfg, 'api_host', None) or '0.0.0.0'
    _port = port or getattr(db_cfg, 'api_port', None) or 8000

    from .api.server import create_app  # noqa: PLC0415
    create_app(profile=profile or db_cfg.active_profile)

    click.echo(f"🌐 FrictionSim2D API server → http://{_host}:{_port}")
    click.echo("   Press Ctrl-C to stop.\n")
    uvicorn.run("src.api.server:app", host=_host, port=_port, reload=reload)


def main():
    try:
        if not any(arg in ('--help', '-h') for arg in sys.argv[1:]):
            _bootstrap_local_db_once()
        cli()
    except Exception:  # pylint: disable=broad-except
        logger.exception("Command failed")
        sys.exit(1)

if __name__ == "__main__":
    main()
