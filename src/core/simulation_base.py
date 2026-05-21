"""Base class for simulation setup.

This module provides the abstract base class `SimulationBase`, which defines
the common interface and utility methods (template rendering, directory setup)
required by all specific simulation types (AFM, Sheet-on-Sheet, etc.).
"""
import os
import logging
from abc import ABC
from pathlib import Path
from typing import Any, Dict, List, Union, Optional
import shutil
import json
from datetime import datetime
import hashlib

from jinja2 import Environment, FileSystemLoader

from ..interfaces.atomsk import AtomskWrapper
from ..interfaces.jinja import PackageLoader
from .utils import get_material_path, get_potential_path
from ..hpc import HPCScriptGenerator, HPCConfig
from ..hpc.manifest import JobManifest

logger = logging.getLogger(__name__)

POTENTIAL_EXTENSIONS = {
    '.sw', '.tersoff', '.rebo', '.airebo', '.reaxff', '.meam',
    '.extep', '.vashishta', '.adp', '.bop', '.eam', '.edip',
    '.comb', '.comb3', '.morse', '.rebomos', '.mod'
}

def _infer_provenance_category(file_path: Path) -> str:
    """Infer provenance category from file extension."""
    ext = file_path.suffix.lower()
    if ext == '.cif':
        return 'cif'
    if ext in POTENTIAL_EXTENSIONS:
        return 'potential'
    return 'other'

class SimulationBase(ABC):
    """Abstract base class for simulation setup.

    Provides common infrastructure for directory creation, template rendering,
    and file writing. Concrete simulation classes (AFM, SheetOnSheet) inherit
    from this.

    Attributes:
        config: The validated configuration object for the simulation.
        output_dir: The root directory for simulation output.
        atomsk: Optional Atomsk interface for geometry manipulation.
        jinja_env: Template engine environment.
    """

    def __init__(self, config: Any, output_dir: Union[str, Path],
                    config_path: Optional[Union[str, Path]] = None):
        """Initialize the simulation base.

        Args:
            config: A Pydantic configuration object (specific to the simulation type).
            output_dir: The directory where files will be generated (e.g., 'afm/InO/...').
            config_path: Path to the original config .ini file (for provenance).
        """
        self.config = config
        # Store both the original relative path (for LAMMPS) and absolute path (for file I/O)
        self.relative_run_dir = Path(output_dir)
        self.output_dir = Path(output_dir).resolve()
        self.config_path = Path(config_path).resolve() if config_path else None
        self.base_output_dir: Optional[Path] = None
        try:
            self.atomsk: Optional[AtomskWrapper] = AtomskWrapper()
        except RuntimeError:
            self.atomsk = None
            logger.info(
                "Atomsk not found. Continuing with native geometry pipeline; "
                "Atomsk-only operations will be unavailable."
            )

        self.jinja_env = Environment(
            loader=PackageLoader('src.templates'),
            trim_blocks=True,
            lstrip_blocks=True
        )

    def _create_directories(self, output_dir: Optional[Path] = None,
                            subdirs: Optional[List[str]] = None) -> None:
        """Creates standard simulation subdirectories.

        Args:
            output_dir: Directory in which to create subdirectories. Defaults to self.output_dir.
            subdirs: Optional list of additional subdirectories to create.
                    Defaults to ['visuals', 'results', 'lammps', 'data'].
        """
        target_dir = output_dir if output_dir is not None else self.output_dir
        default_dirs = ['visuals', 'results', 'lammps', 'data']
        for d in default_dirs + (subdirs or []):
            (target_dir / d).mkdir(parents=True, exist_ok=True)

    def render_template(self, template_name: str,
                        context: Dict[str, Any]) -> str:
        """Render a Jinja2 template with the provided context.

        Args:
            template_name: Relative path to the template.
            context: Dictionary of variables to pass to the template.

        Returns:
            The rendered template string.
        """
        try:
            template = self.jinja_env.get_template(template_name)
            return template.render(context)
        except Exception as e:
            logger.exception("Failed to render template '%s': %s", template_name, e)
            raise

    def set_base_output_dir(self, base_output_dir: Union[str, Path]) -> None:
        """Set the base output directory and update relative run paths.

        Args:
            base_output_dir: Root directory that contains all simulations.
        """
        base_output_dir = Path(base_output_dir).resolve()
        self.base_output_dir = base_output_dir
        try:
            self.relative_run_dir = self.output_dir.relative_to(base_output_dir)
        except ValueError:
            self.relative_run_dir = Path(self.output_dir)

    def write_file(self, filename: Union[str, Path],
                    content: str, output_dir: Optional[Union[str, Path]] = None) -> Path:
        """Write string content to a file in the output directory.

        Args:
            filename: Relative path or filename (e.g., 'lammps/system.in').
            content: The string content to write.
            output_dir: Optional custom output directory. Defaults to self.output_dir.

        Returns:
            The full path to the written file.
        """
        target_dir = Path(output_dir) if output_dir is not None else self.output_dir
        full_path = target_dir / filename
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text(content, encoding='utf-8')
        return full_path

    def add_to_provenance(self, file_path: Union[str, Path], category: str = 'auto',
                            component: Optional[str] = None) -> Optional[Path]:
        """Add a file to the provenance folder for reproducibility tracking.
        
        All files (CIFs, potentials, configs) are copied into their respective
        provenance subdirectories. This ensures portability when folders are
        transferred to HPC or moved elsewhere.
        
        Also updates the provenance manifest to track component usage.
        
        Args:
            file_path: Path to the file to add
            category: 'cif', 'potential', or 'auto' (detect by extension)
            component: Name of component using this file (e.g., 'tip', 'substrate').
                        Optional but recommended for AiiDA traceability.
            
        Returns:
            Path to the file in provenance folder
        """
        file_path = Path(file_path)
        if not file_path.exists():
            logger.warning("Cannot add to provenance, file not found: %s", file_path)
            return None

        prov_dir = self.output_dir / 'provenance'

        if category == 'auto':
            category = _infer_provenance_category(file_path)

        if category == 'cif':
            dest_dir = prov_dir / 'cif'
        elif category == 'potential':
            dest_dir = prov_dir / 'potentials'
        else:
            dest_dir = prov_dir

        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_path = dest_dir / file_path.name

        if not dest_path.exists():
            shutil.copy2(file_path, dest_path)
            logger.debug("Added to provenance: %s", file_path.name)

        self._update_provenance_manifest(file_path, category, component, dest_path)

        return dest_path

    def _update_provenance_manifest(self, original_path: Path, category: str,
                                        component: Optional[str], stored_path: Path) -> None:
        """Update the provenance manifest with file metadata.
        
        Creates/updates a JSON manifest file that tracks which files are used for
        which components. This enables AiiDA to trace potentials to specific
        components (e.g., tip, substrate) via the config.ini file.
        
        Manifest structure:
        {
            "version": "1.0",
            "created": "2025-01-19T...",
            "last_updated": "2025-01-19T...",
            "files": [
                {
                    "filename": "Au.sw",
                    "original_path": "/path/to/repo/potentials/Au.sw",
                    "stored_path": "provenance/potentials/Au.sw",
                    "category": "potential",
                    "component": "tip",
                    "checksum": "sha256...",
                    "added_at": "2025-01-19T..."
                }
            ]
        }
        """
        manifest_path = self.output_dir / 'provenance' / 'manifest.json'

        if manifest_path.exists():
            manifest_data = json.loads(manifest_path.read_text(encoding='utf-8'))
        else:
            manifest_data = {
                'version': '1.0',
                'created': datetime.now().isoformat(),
                'files': []
            }

        original_path_resolved = str(original_path.resolve())
        checksum = hashlib.sha256(original_path.read_bytes()).hexdigest()

        search_criteria = (
            f for f in manifest_data['files']
            if f['category'] == category
            and (
                f.get('original_path') == original_path_resolved
                or f.get('checksum') == checksum
            )
        )
        existing = next(search_criteria, None)

        def ensure_list(val: Optional[Union[str, List[str]]]) -> List[str]:
            if val is None:
                return []
            if isinstance(val, list):
                return [entry for entry in val if isinstance(entry, str)]
            return [val]

        new_components = ensure_list(component)

        if existing:
            existing_components = ensure_list(existing.get('components'))
            for c in new_components:
                if c and c not in existing_components:
                    existing_components.append(c)
            existing['components'] = existing_components
            existing['original_path'] = original_path_resolved
            existing['stored_path'] = str(stored_path.relative_to(self.output_dir))
            existing['checksum'] = checksum
            existing['added_at'] = datetime.now().isoformat()
        else:
            manifest_data['files'].append({
                'filename': original_path.name,
                'original_path': original_path_resolved,
                'stored_path': str(stored_path.relative_to(self.output_dir)),
                'category': category,
                'components': new_components,
                'checksum': checksum,
                'added_at': datetime.now().isoformat()
            })

        manifest_data['last_updated'] = datetime.now().isoformat()

        manifest_path.write_text(json.dumps(manifest_data, indent=2), encoding='utf-8')
        logger.debug("Updated provenance manifest: %s", manifest_path)

    def _add_component_files_to_provenance(self, component_name: str, config: Any) -> None:
        """Find and add CIF and potential files for a component config to provenance.

        Uses canonical component paths from the validated schema: `cif_path` and
        `pot_path`.

        Args:
            component_name: Name of the component ('tip', 'sub', 'sheet', etc.)
            config: Component config object (TipConfig, SubstrateConfig, SheetConfig, etc.)
        """
        cif = None
        try:
            cif = getattr(config, 'cif_path', None) or (
                get_material_path(config.mat) if hasattr(config, 'mat') and config.mat else None
            )
        except (FileNotFoundError, ValueError, KeyError):
            pass

        if cif:
            try:
                self.add_to_provenance(cif, 'cif', component=component_name)
            except (FileNotFoundError, ValueError, KeyError):
                pass

        pot = None
        try:
            pot_path = getattr(config, 'pot_path', None)
            pot = get_potential_path(pot_path) if pot_path else None
        except (FileNotFoundError, ValueError, KeyError):
            pass

        if pot:
            try:
                self.add_to_provenance(pot, 'potential', component=component_name)
            except (FileNotFoundError, ValueError, KeyError):
                pass

    def _generate_hpc_scripts(self) -> None:
        """Generate HPC job submission scripts for all simulations.

        This method should be called after all simulation files are written.
        It creates a manifest of all jobs and generates appropriate HPC scripts.
        For AFM simulations with system.in files, creates two-phase jobs with dependencies.
        """

        if self.base_output_dir is not None and self.base_output_dir != self.output_dir:
            logger.info(
                "Skipping per-simulation HPC script generation; use base output dir instead."
            )
            return

        hpc_dir = self._get_hpc_output_dir()
        hpc_dir.mkdir(parents=True, exist_ok=True)

        sim_root = self.base_output_dir if self.base_output_dir is not None else self.output_dir
        manifest = JobManifest.from_simulation_directory(
            sim_root,
            name=self._get_hpc_job_name()
        )

        if manifest.n_jobs == 0:
            logger.warning("No simulation jobs found for HPC script generation")
            return

        logger.info("Created manifest with %d jobs", manifest.n_jobs)

        job_name = self._get_hpc_job_name()

        hpc_config = HPCConfig.from_settings(
            self.config.settings.hpc,
            job_name=job_name
        )

        if not hpc_config.modules:
            raise ValueError("HPC modules list is empty; set hpc.modules in settings.")
        if hpc_config.use_tmpdir and not hpc_config.scratch_dir:
            raise ValueError("HPC scratch_dir is required when use_tmpdir is true.")

        generator = HPCScriptGenerator(hpc_config)

        if hpc_config.hpc_home:
            base_dir = hpc_config.hpc_home.rstrip('/') + '/' + sim_root.name
        elif hpc_config.scheduler_type == 'pbs':
            base_dir = '$PBS_O_WORKDIR'
        else:
            base_dir = '$SLURM_SUBMIT_DIR'

        manifest_json_path = hpc_dir / 'manifest.json'
        manifest.save(manifest_json_path)
        logger.info("Saved manifest to %s", manifest_json_path)

        if manifest.has_system_jobs():
            logger.info("Detected system initialization jobs - using two-phase submission")
            scripts = generator.generate_two_phase_scripts(
                manifest=manifest,
                output_dir=hpc_dir,
                scheduler=self.config.settings.hpc.scheduler_type,
                base_dir=base_dir
            )
            n_scripts = len(scripts['system']) + len(scripts['slide'])
            logger.info("Generated %d HPC scripts (system + slide) in %s", n_scripts, hpc_dir)
        else:
            logger.info("No system jobs detected - using single-phase submission")
            slide_manifest_path = hpc_dir / 'manifest_slide.txt'
            manifest.save_script_list(slide_manifest_path)

            slide_manifest_rel = f"{base_dir}/hpc/manifest_slide.txt"

            templates_dir = Path(os.path.dirname(__file__)).parent / 'templates' / 'hpc'
            jinja_env = Environment(
                loader=FileSystemLoader(templates_dir),
                trim_blocks=True,
                lstrip_blocks=True
            )

            context = hpc_config.to_dict()
            context.update({
                'array_size': manifest.n_jobs,
                'manifest_file': slide_manifest_rel,
                'manifest_filename': 'manifest_slide.txt',
                'base_dir': base_dir,
                'log_dir': hpc_config.log_dir or f"{base_dir}/logs",
                'job_name': job_name,
            })

            template_name = 'pbs_array.j2' if hpc_config.scheduler_type == 'pbs' else 'slurm_array.j2'
            template = jinja_env.get_template(template_name)
            script_content = template.render(context)

            script_name = 'run.pbs' if hpc_config.scheduler_type == 'pbs' else 'run.sh'
            script_path = hpc_dir / script_name
            script_path.write_text(script_content, encoding='utf-8')

            logger.info("Generated HPC script in %s", hpc_dir)

    def _get_hpc_output_dir(self) -> Path:
        """Get the output directory for HPC scripts.
        
        Override this method if you need custom HPC script location.
        Default is output_dir/hpc for single simulations,
        or base_output_dir/hpc for multi-simulation setups.
        
        Returns:
            Path to HPC scripts directory
        """
        if self.base_output_dir is not None:
            return self.base_output_dir / 'hpc'
        return self.output_dir / 'hpc'

    def _get_hpc_job_name(self) -> str:
        """Get the job name for HPC scripts.
        
        Override this method to customize job names.
        
        Returns:
            Job name string
        """
        material = getattr(self.config, 'sheet', None) or getattr(self.config, '2D', None)
        if material and hasattr(material, 'mat'):
            return f"friction_{material.mat}"
        return "friction2d"

    def _collect_simulation_paths(self) -> List[str]:
        """Collect all simulation directory paths relative to HPC base.
        
        Override this method for custom simulation directory structures.
        Default implementation looks for directories with lammps/ subdirectory.
        
        Returns:
            List of relative paths to simulation directories
        """
        base_dir = self.base_output_dir if self.base_output_dir is not None else self.output_dir

        paths = []
        for item in base_dir.iterdir():
            if item.is_dir():
                lammps_dir = item / 'lammps'
                if lammps_dir.exists() and (lammps_dir / 'system.in').exists():
                    rel_path = item.relative_to(base_dir)
                    paths.append(str(rel_path))

        return sorted(paths)
