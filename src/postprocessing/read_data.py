"""Read and process friction simulation output data.

Walks through a directory of simulation results, parses filenames and
paths to extract metadata, reads time-series data, calculates derived
quantities (COF, lateral force) and stores everything in a structured
format that can be exported to JSON for downstream plotting.
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class _NpEncoder(json.JSONEncoder):
    """JSON encoder that handles NumPy and pandas types."""

    def default(self, o):  # pylint: disable=arguments-renamed
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, pd.DataFrame):
            return {'columns': o.columns.tolist(), 'data': o.values.tolist()}
        return super().default(o)


class DataReader:  # pylint: disable=too-many-instance-attributes
    """Reads and processes friction simulation data.

    Walks through a directory of simulation results, parses filenames and
    file paths to extract metadata, reads the time-series data from each
    valid file, calculates derived quantities (COF, lateral force), and
    stores it in a structured format.
    """

    # Column names matching fix fc_ave output order in afm/slide.lmp.
    # 9 LAMMPS variables + timestep = 10 file columns.
    _AFM_FILE_COLUMNS = [
        'time', 'nf', 'lfx', 'lfy', 'comx', 'comy', 'comz',
        'tipx', 'tipy', 'tipz',
    ]

    # Column names matching fix fc_ave output order in sheetonsheet/slide.lmp.
    # 16 LAMMPS variables + timestep = 17 file columns.
    _SHEET_FILE_COLUMNS = [
        'time', 'v_xfrict', 'v_yfrict', 'v_sx', 'v_sy', 'v_sz',
        'v_fx', 'v_fy', 'v_fz', 'v_comx_top', 'v_comy_top',
        'v_comx_ctop', 'v_comy_ctop', 'v_comz_ctop',
        'v_comx_cbot', 'v_comy_cbot', 'v_comz_cbot',
    ]

    _SHEET_COLUMN_RENAME = {
        'v_xfrict': 'lfx',
        'v_yfrict': 'lfy',
        'v_fz': 'nf',
    }

    # Filename patterns (compiled once at class level)
    _FILE_PATTERN_TIP = re.compile(r'fc_ave_slide_(\d+\.?\d*)nN_(\d+)angle_(\d+)ms_l(\d+)')
    # New AFM tip naming written by the current builder, e.g.
    # ``friction_f30_a0.0_s2_layer2.txt``. Groups match _FILE_PATTERN_TIP:
    # (load, angle, speed, layer). Load is always a force (nN) for tip sims.
    _FILE_PATTERN_TIP_ALT = re.compile(
        r'friction_f(\d+\.?\d*)_a(\d+\.?\d*)_s(\d+\.?\d*)_layer(\d+)'
    )
    _FILE_PATTERN_SHEET = re.compile(
        r'fc_ave_slide_(\d+\.?\d*)(GPa|nN)_(\d+\.?\d*)angle_(\d+\.?\d*)ms'
    )
    _FILE_PATTERN_SHEET_ALT = re.compile(r'friction_p(\d+\.?\d*)_a(\d+\.?\d*)_s(\d+\.?\d*)')

    # Path patterns (compiled once at class level)
    _PATH_PATTERN_TIP = re.compile(r'(\d+x_\d+y)/sub_(\w+)_tip_(\w+)_(r\d+)')
    _PATH_PATTERN_SHEET = re.compile(
        r'(?:sheetvsheet/)?([\w\d\-_]+)/(\d+x_\d+y)/([\w\d\-_]+)?/?results'
    )

    def __init__(self, results_dir: str = 'results_110725_test') -> None:
        """Initialise the DataReader.

        Args:
            results_dir: Path to the directory containing simulation results.
        """
        self.results_dir = Path(results_dir)
        self.output_dir = self.results_dir / 'outputs'
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.full_data_nested: dict = {}

        # Read the data and populate the dictionaries and metadata
        (
            self.time_series,
            self.incomplete_files,
            self.incomplete_materials,
            self.metadata,
            self.ntimestep,
        ) = self.read_data()

    def _calculate_derived_quantities(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate lateral force and coefficient of friction.

        Delegates to :func:`~src.data.models.compute_derived_columns` to
        ensure consistent calculation across the codebase.

        Args:
            df: DataFrame with at least lfx, lfy, and nf columns.

        Returns:
            DataFrame with added lateral_force and cof columns.
        """
        if 'lfx' in df.columns and 'lfy' in df.columns and 'nf' in df.columns:
            from ..data.models import compute_derived_columns  # noqa: PLC0415  # pylint: disable=import-outside-toplevel

            lateral_force, cof = compute_derived_columns(
                np.asarray(df['lfx'].values),
                np.asarray(df['lfy'].values),
                np.asarray(df['nf'].values),
            )
            df['lateral_force'] = lateral_force
            df['cof'] = cof

        return df

    def _get_output_path(self, filename: str) -> Path:
        """Construct the full path for an output file."""
        return self.output_dir / filename

    def _first_pass_ntimestep(self) -> int:
        """First directory walk to determine the expected timestep count."""
        ntimestep = 0
        logger.info("Starting first pass to determine ntimestep...")
        for root, _, files in os.walk(self.results_dir):
            is_tip = self._PATH_PATTERN_TIP.search(root)
            is_sheet = self._PATH_PATTERN_SHEET.search(root)
            if not (is_tip or is_sheet):
                continue
            file_patterns = (
                (self._FILE_PATTERN_TIP, self._FILE_PATTERN_TIP_ALT)
                if is_tip
                else (self._FILE_PATTERN_SHEET, self._FILE_PATTERN_SHEET_ALT)
            )
            for filename in files:
                if any(p.match(filename) for p in file_patterns):
                    filepath = Path(root) / filename
                    try:
                        df = pd.read_csv(
                            filepath, sep=r'\s+', header=None, usecols=[0], skiprows=2,
                        )
                        ntimestep = max(ntimestep, len(df))
                    except (pd.errors.EmptyDataError, IndexError):
                        continue
        return ntimestep

    def _resolve_path_info(
        self, root: str,
    ) -> tuple[str, str, str, str, str, str] | None:
        """Parse a directory path to extract simulation context.

        Returns:
            ``(sim_type, material, size, substrate, tip_material, tip_radius)``
            or ``None`` if the path does not match any known pattern.
        """
        path_match_tip = self._PATH_PATTERN_TIP.search(root)
        if path_match_tip:
            size, substrate_material, tip_material, tip_radius = path_match_tip.groups()
            try:
                search_path = str(self.results_dir / 'afm')
                start_dir = (
                    search_path
                    if os.path.commonpath([root, search_path]) == search_path
                    else str(self.results_dir)
                )
                material_path_full = root[:root.find(size)]
                material = str(Path(material_path_full).relative_to(start_dir)).strip('/')
            except (IndexError, ValueError):
                return None
            if not material or material == '.':
                return None
            return ('tip', material, size, substrate_material, tip_material, tip_radius)

        path_match_sheet = self._PATH_PATTERN_SHEET.search(root)
        if path_match_sheet:
            material, size, substrate_material = path_match_sheet.groups()
            substrate_material = (substrate_material or 'N/A').strip('/')
            return ('sheet', material, size, substrate_material, 'sheet', 'N/A')

        return None

    def _read_dataframe(self, filepath: Path, sim_type: str) -> pd.DataFrame:
        """Read a LAMMPS output file and return a labelled DataFrame."""
        if sim_type == 'sheet':
            sheet_col_names: list[str] | None = None
            try:
                with open(filepath, 'r', encoding='utf-8', errors='ignore') as fh:
                    header_lines = [next(fh, ''), next(fh, '')]
                if len(header_lines) > 1 and header_lines[1].startswith('#'):
                    tokens = header_lines[1].lstrip('#').strip().split()
                    if tokens:
                        sheet_col_names = [
                            'time' if t == 'TimeStep' else t for t in tokens
                        ]
            except OSError:
                pass
            col_names = sheet_col_names or self._SHEET_FILE_COLUMNS
            df = pd.read_csv(
                filepath, sep=r'\s+', header=None, names=col_names, skiprows=2,
            )
            df.rename(columns=self._SHEET_COLUMN_RENAME, inplace=True)
        else:
            df = pd.read_csv(
                filepath, sep=r'\s+', header=None,
                names=self._AFM_FILE_COLUMNS, skiprows=2,
            )
            if df.shape[1] != len(self._AFM_FILE_COLUMNS):
                logger.warning(
                    "Column count mismatch in %s: got %d, expected %d. "
                    "Template fix fc_ave order may have changed.",
                    filepath, df.shape[1], len(self._AFM_FILE_COLUMNS),
                )
        return df

    def _parse_filename_params(
        self,
        sim_type: str,
        filename: str,
        file_match: re.Match,
    ) -> tuple[float, int, int, int, bool]:
        """Extract load, angle, speed, layer and pressure flag from a regex match.

        Returns:
            ``(load_val, angle, speed, layer, is_pressure)``
        """
        if sim_type == 'sheet':
            if self._FILE_PATTERN_SHEET.match(filename):
                load_str, unit_str, angle_str, speed_str = file_match.groups()
                is_pressure = unit_str == 'GPa'
                load_val = float(load_str)
            else:
                load_str, angle_str, speed_str = file_match.groups()
                is_pressure = True
                load_val = float(load_str)
            layer = 2
        else:
            load_str, angle_str, speed_str, layer_str = file_match.groups()
            layer = int(layer_str)
            is_pressure = False
            load_val = float(load_str)
        return load_val, int(float(angle_str)), int(float(speed_str)), layer, is_pressure

    @staticmethod
    def _finalize_metadata(metadata: dict) -> dict:
        """Convert mutable sets in the metadata dict to sorted lists."""
        final: dict = {}
        for k, v in metadata.items():
            if k in ('forces_and_angles', 'pressures_and_angles'):
                final[k] = {load: sorted(list(angles)) for load, angles in v.items()}
            else:
                final[k] = sorted(list(v))
        material_types: dict = {
            'b_type': [], 'h_type': [], 't_type': [], 'p_type': [], 'other': [],
        }
        for mat in final.get('materials', []):
            prefix = mat.split('_', 1)[0]
            type_key = f"{prefix}_type"
            material_types[type_key if type_key in material_types else 'other'].append(mat)
        final['material_types'] = material_types
        return final

    def read_data(self) -> tuple:  # pylint: disable=too-many-locals
        """Walk through the results directory and read all simulation data.

        Uses a two-pass approach to dynamically determine the correct
        number of timesteps for a complete file.
        """
        ntimestep = self._first_pass_ntimestep()
        if ntimestep == 0:
            logger.warning("No valid data files found. Could not determine ntimestep.")
            return None, {}, {}, {}, 0
        logger.info("Determined ntimestep for a complete file to be: %d", ntimestep)

        time_series = None
        incomplete_files: dict = {}
        incomplete_materials: dict = {}
        metadata: dict = {
            'materials': set(), 'substrates': set(), 'tip_materials': set(),
            'tip_radii': set(), 'layers': set(), 'speeds': set(),
            'forces_and_angles': {}, 'pressures_and_angles': {},
        }

        for root, _, files in os.walk(self.results_dir):
            path_info = self._resolve_path_info(root)
            if path_info is None:
                continue
            sim_type, material, size, substrate_material, tip_material, tip_radius = path_info
            safe_material = material.replace('-', '_').replace('/', '__')
            size_key = size.replace('x_', 'x')

            for filename in files:
                if sim_type == 'sheet':
                    file_match = (
                        self._FILE_PATTERN_SHEET.match(filename)
                        or self._FILE_PATTERN_SHEET_ALT.match(filename)
                    )
                else:
                    file_match = (
                        self._FILE_PATTERN_TIP.match(filename)
                        or self._FILE_PATTERN_TIP_ALT.match(filename)
                    )
                if not file_match:
                    continue

                filepath = Path(root) / filename
                try:
                    df = self._read_dataframe(filepath, sim_type)

                    if ntimestep - len(df) > 3:
                        incomplete_files.setdefault(size_key, []).append(str(filepath))
                        incomplete_materials.setdefault(size_key, set()).add(material)
                        continue

                    if time_series is None:
                        time_series = df['time'].to_list()

                    load_val, angle, speed, layer, is_pressure = self._parse_filename_params(
                        sim_type, filename, file_match,
                    )
                    metadata['materials'].add(safe_material)
                    metadata['substrates'].add(substrate_material)
                    metadata['tip_materials'].add(tip_material)
                    metadata['tip_radii'].add(tip_radius)
                    metadata['speeds'].add(speed)
                    metadata['layers'].add(layer)
                    if is_pressure:
                        metadata['pressures_and_angles'].setdefault(load_val, set()).add(angle)
                    else:
                        metadata['forces_and_angles'].setdefault(load_val, set()).add(angle)

                    df_processed = df.drop(columns=['time'])
                    df_processed = self._calculate_derived_quantities(df_processed)

                    base_path = (
                        self.full_data_nested
                        .setdefault(safe_material, {})
                        .setdefault(size_key, {})
                        .setdefault(substrate_material, {})
                        .setdefault(tip_material, {})
                        .setdefault(tip_radius, {})
                        .setdefault(f'l{layer}', {})
                        .setdefault(f's{speed}', {})
                    )
                    load_key = f'p{load_val}' if is_pressure else f'f{load_val}'
                    base_path.setdefault(load_key, {})[f'a{angle}'] = df_processed

                except (pd.errors.EmptyDataError, IndexError, ValueError) as e:
                    logger.warning("Could not process file %s: %s", filepath, e)
                    incomplete_files.setdefault(size_key, []).append(str(filepath))
                    incomplete_materials.setdefault(size_key, set()).add(material)

        return (
            time_series,
            incomplete_files,
            incomplete_materials,
            self._finalize_metadata(metadata),
            ntimestep,
        )

    def export_full_data_to_json(self) -> None:
        """Export the full time-series data to JSON files, one per size."""
        data_by_size: dict = {}
        for material, mat_data in self.full_data_nested.items():
            for size, size_data in mat_data.items():
                data_by_size.setdefault(size, {})[material] = size_data

        for size_key, size_data in data_by_size.items():
            output_path = self._get_output_path(f'output_full_{size_key}.json')

            full_output_metadata = self.metadata.copy()
            full_output_metadata['time_series'] = self.time_series

            relevant_tip_radii: set = set()
            for mat_data in size_data.values():
                for sub_data in mat_data.values():
                    for tip_mat_data in sub_data.values():
                        relevant_tip_radii.update(tip_mat_data.keys())

            full_output_metadata['tip_radii'] = sorted(list(relevant_tip_radii))
            full_output_metadata['size'] = size_key

            output_with_metadata = {
                'metadata': full_output_metadata,
                'results': size_data,
            }

            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(output_with_metadata, f, cls=_NpEncoder)
            logger.info("Full time-series data for size %s exported to %s", size_key, output_path)

    def export_issue_reports(self) -> None:
        """Export reports on incomplete files and materials to text files."""
        if self.incomplete_files:
            for size, files in self.incomplete_files.items():
                filepath = self._get_output_path(f'incomplete_files_{size}.txt')
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write('\n'.join(sorted(files)))
                logger.info("Incomplete files for size %s saved to %s", size, filepath)

        if self.incomplete_materials:
            for size, materials in self.incomplete_materials.items():
                filepath = self._get_output_path(f'incomplete_materials_{size}.txt')
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write('\n'.join(sorted(list(materials))))
                logger.info("Incomplete materials for size %s saved to %s", size, filepath)
