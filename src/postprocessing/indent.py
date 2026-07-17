"""Indent-simulation delivery: penetration (hold) + adhesion (retract) in one pass.

Each AFM indentation ("indent") run (:mod:`src.builders.indent`) writes, per
``(material, layer, load)``, both:

    - the 10-column hold file ``results/friction_f<load>_a0_s<speed>_layer<N>.txt``
      (same layout as the sliding campaign — so the campaign reader
      :class:`~src.postprocessing.read_data.DataReader` parses it unchanged), and
    - the force–distance retract curve ``results/indent_f<load>N_l<layer>.csv``
      (columns ``step, z_tip, fz_tip, fx_tip, fy_tip, pe``).

This module reduces a run root to the recipe hand-back for **both** contact
descriptor sets (``POKE_SIM_RECIPE.md``), subsuming the former separate poke and
adhesion delivery modules:

    Penetration (from the hold files):
      - ``<out_dir>/output_indent_f<load>_s<seed>.json`` — one file per (load, seed
        batch) in the exact nested campaign schema, so the ML side reduces each
        leaf exactly as the campaign (``pen_mean = mean(tip_z − com_z)``).

    Adhesion (from the retract curves):
      - ``<data_dir>/indent_features.csv`` — one row per ``(material, layer, load)``
        with ``pulloff_force``, ``work_of_adhesion``, ``contact_stiffness`` (+
        supporting columns), merge key ``material`` (underscored id).
      - ``<out_dir>/indent/<material>_l<layer>[_f<load>].csv`` — the tidied
        per-condition retract curve.

    Combined:
      - ``<out_dir>/indent_manifest.csv`` — one row per condition with the fresh
        velocity ``seed``, the hold length (``hold_steps``), the retract
        discretization (``z_step``, ``n_steps``) and a best-effort ``wall_time``.

Reduced adhesion descriptors (``fz_tip`` is the z-force the surface exerts on the
tip: + repulsive, − adhesive; ``z_tip`` increases as the tip retracts):

    ================= ================================================================
    pulloff_force      -min(fz_tip) — the peak tensile (adhesive) force, nN
    work_of_adhesion   ∫ over the attractive branch of (-fz_tip) dz — energy to
                       separate the contact, eV (also reported in nN·Å)
    contact_stiffness  |slope of fz_tip vs z_tip| over the repulsive wall — nN/Å
    snapoff_z          tip COM z at the pull-off minimum, Å
    ================= ================================================================
"""
from __future__ import annotations

import csv
import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..data.models import EV_A_TO_NN

logger = logging.getLogger(__name__)

#: Top-level run directory name for indent runs (matches the model name).
INDENT_TOPDIR = "indent"

#: Retract CSV filename, e.g. ``indent_f10N_l2.csv``.
_CSV_RE = re.compile(r"indent_f(?P<load>[0-9.]+)N_l(?P<layer>\d+)\.csv$")

_WALL_TIME_RE = re.compile(r"Total wall time:\s*([0-9:]+)")
#: Log-like filenames worth scanning for a wall time (avoid huge trajectories).
_LOG_GLOBS = ("log.lammps", "*.log", "logs/*", "*.out", "*.o*")


def material_id(name: str) -> str:
    """Map a directory/material name to the ML merge id (``h-MoS2`` -> ``h_MoS2``)."""
    return name.replace("-", "_").replace("/", "__")


def _load_token(load_val: float) -> str:
    """Filename token for a load: ``10.0`` -> ``10``, ``10.5`` -> ``10.5``."""
    return str(int(load_val)) if float(load_val).is_integer() else str(load_val)


def _parse_wall_seconds(value: str) -> Optional[int]:
    """Parse a LAMMPS ``H:MM:SS`` (or ``S``) wall-time string to seconds."""
    try:
        nums = [int(p) for p in value.split(":")]
    except ValueError:
        return None
    seconds = 0
    for n in nums:
        seconds = seconds * 60 + n
    return seconds


def _run_wall_seconds(run_dir: Path) -> Optional[int]:
    """Best-effort: last ``Total wall time`` found in a log under ``run_dir``."""
    seen: set = set()
    for pattern in _LOG_GLOBS:
        for path in run_dir.rglob(pattern):
            if not path.is_file() or path in seen:
                continue
            seen.add(path)
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            matches = _WALL_TIME_RE.findall(text)
            if matches:
                return _parse_wall_seconds(matches[-1])
    return None


def load_indent_meta(run_root: Path) -> Dict[str, Any]:
    """Collect the per-run ``indent_meta.json`` records written by the builder.

    Returns a dict keyed by the underscored material id (matching both the
    DataReader tree and the feature-table merge id):
        - ``base_seed[material]`` — campaign base seed (names the per-load JSON),
        - ``seed[(material, layer)]`` — the fresh derived velocity seed,
        - ``hold_steps[material]`` — hold length in steps,
        - ``z_step[material]`` / ``n_steps[material]`` — retract discretization,
        - ``run_dir[material]`` — the simulation directory (for wall-time lookup).
    """
    base_seed: Dict[str, int] = {}
    seed: Dict[Tuple[str, int], int] = {}
    hold_steps: Dict[str, int] = {}
    z_step: Dict[str, float] = {}
    n_steps: Dict[str, int] = {}
    run_dir: Dict[str, Path] = {}

    for meta_path in sorted(run_root.rglob("provenance/indent_meta.json")):
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Skipping unreadable indent meta %s: %s", meta_path, exc)
            continue
        mat = material_id(str(meta.get("material", "")))
        if not mat:
            continue
        base_seed[mat] = int(meta.get("base_seed", 0))
        hold_steps[mat] = int(meta.get("hold_steps", 0))
        z_step[mat] = float(meta.get("z_step", 0.0))
        n_steps[mat] = int(meta.get("n_steps", 0))
        run_dir[mat] = meta_path.parent.parent
        for layer_str, layer_seed in (meta.get("layers") or {}).items():
            try:
                seed[(mat, int(layer_str))] = int(layer_seed)
            except (TypeError, ValueError):
                continue
    return {"base_seed": base_seed, "seed": seed, "hold_steps": hold_steps,
            "z_step": z_step, "n_steps": n_steps, "run_dir": run_dir}


# =============================================================================
# Penetration (hold) — per-load campaign-schema JSON
# =============================================================================

def _load_keys(sub_tree: Dict) -> List[str]:
    """All ``f<load>`` keys present anywhere in one material's size sub-tree."""
    keys: set = set()
    for tip_data in sub_tree.values():
        for r_data in tip_data.values():
            for l_data in r_data.values():
                for s_data in l_data.values():
                    for f_data in s_data.values():
                        keys.update(f_data.keys())
    return sorted(keys)


def _filter_to_load(sub_tree: Dict, load_key: str) -> Dict:
    """Copy a ``substrate → … → s{speed}`` sub-tree keeping only ``load_key``."""
    out: Dict = {}
    for sub, tip_data in sub_tree.items():
        for tip, r_data in tip_data.items():
            for radius, l_data in r_data.items():
                for layer, s_data in l_data.items():
                    for speed, f_data in s_data.items():
                        if load_key in f_data:
                            (out.setdefault(sub, {}).setdefault(tip, {})
                                .setdefault(radius, {}).setdefault(layer, {})
                                .setdefault(speed, {})[load_key]) = f_data[load_key]
    return out


def split_by_load(full_data_nested: Dict,
                  base_seed: Dict[str, int]) -> Dict[Tuple[str, int], Dict]:
    """Regroup the campaign tree into ``(load_key, seed) -> {material: subtree}``.

    With a shared ``[indent] seed`` every material carries the same base seed, so
    this yields one file per load; ad-hoc unseeded runs fall into per-material
    seed batches (still correct, just more files).
    """
    per_file: Dict[Tuple[str, int], Dict] = {}
    for material, size_data in full_data_nested.items():
        seed = base_seed.get(material, 0)
        for _size_key, sub_tree in size_data.items():
            for load_key in _load_keys(sub_tree):
                filtered = _filter_to_load(sub_tree, load_key)
                per_file.setdefault((load_key, seed), {})[material] = filtered
    return per_file


def iter_hold_conditions(full_data_nested: Dict):
    """Yield ``(material, layer, load_val)`` for every hold leaf.

    Walks the campaign tree depth ``material → size → substrate → tip_mat →
    radius → l{layer} → s{speed} → f{load} → a{angle}``.
    """
    for material, size_data in full_data_nested.items():
        for sub_data in size_data.values():
            for tip_data in sub_data.values():
                for r_data in tip_data.values():
                    for l_container in r_data.values():
                        for l_key, l_data in l_container.items():
                            layer = int(l_key.replace("l", ""))
                            for s_data in l_data.values():
                                for load_key in s_data:
                                    yield material, layer, float(load_key[1:])


def _deliver_penetration(top: Path, out_dir: Path,
                         meta: Dict[str, Any]) -> Tuple[Dict[str, Path], set]:
    """Write the per-load penetration JSON; return (written, hold-condition set)."""
    from .read_data import DataReader, _NpEncoder  # noqa: PLC0415

    reader = DataReader(results_dir=str(top))
    if not reader.full_data_nested:
        logger.info("No indent hold results found under %s", top)
        return {}, set()

    written: Dict[str, Path] = {}
    per_file = split_by_load(reader.full_data_nested, meta["base_seed"])
    for (load_key, seed), materials in sorted(per_file.items()):
        load_val = float(load_key[1:])
        metadata = dict(reader.metadata)
        metadata["time_series"] = reader.time_series
        metadata["indent_load_nN"] = load_val
        metadata["indent_base_seed"] = seed
        payload = {"metadata": metadata, "results": materials}
        name = f"output_indent_f{_load_token(load_val)}_s{seed}.json"
        path = out_dir / name
        with path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, cls=_NpEncoder)
        written[name] = path
        logger.info("Wrote %d material(s) at load %g nN -> %s",
                    len(materials), load_val, path)

    conditions = {(m, l, ld) for m, l, ld in iter_hold_conditions(reader.full_data_nested)}
    return written, conditions


# =============================================================================
# Adhesion (retract) — features + tidied curves
# =============================================================================

def retract_features(df: pd.DataFrame) -> Dict[str, float]:
    """Reduce one retract force–distance curve to adhesion descriptors.

    ``df`` carries ``z_tip`` (Å) and ``fz_tip`` (nN); rows are re-sorted by
    ascending ``z_tip`` (the retract direction) so the descriptors are robust to
    the row order. ``fz_tip`` is + repulsive / - adhesive.
    """
    z = df["z_tip"].to_numpy(dtype=float)
    fz = df["fz_tip"].to_numpy(dtype=float)
    order = np.argsort(z)
    z, fz = z[order], fz[order]

    out: Dict[str, float] = {"n_points": int(z.size)}
    out["z_span"] = float(z[-1] - z[0]) if z.size > 1 else 0.0

    f_min = float(np.min(fz))
    i_min = int(np.argmin(fz))
    out["f_adh_min"] = f_min                        # signed (nN); the tensile trough
    out["pulloff_force"] = float(max(0.0, -f_min))  # peak adhesive force (nN)
    out["snapoff_z"] = float(z[i_min])

    # Work of adhesion: area under the attractive (fz < 0) branch, ∫ (-fz) dz.
    tensile = np.clip(-fz, 0.0, None)
    w_nnA = float(np.trapz(tensile, z)) if z.size > 1 else 0.0
    out["work_of_adhesion_nnA"] = w_nnA
    out["work_of_adhesion"] = w_nnA / EV_A_TO_NN     # eV (1 nN·Å = 1/EV_A_TO_NN eV)

    # Contact stiffness: slope of the *leading repulsive wall* only — from the
    # loaded contact (fz > 0) up to and including the first zero crossing. Fitting
    # every fz > 0 point would fold in the near-zero tail past detachment and wash
    # the steep wall out; the wall is the near-contact segment before fz goes tensile.
    if fz[0] > 0.0:
        below = np.nonzero(fz <= 0.0)[0]
        end = max(int(below[0]) + 1 if below.size else int(fz.size), 2)
        z_wall, f_wall = z[:end], fz[:end]
        if float(np.ptp(z_wall)) > 0.0:
            out["contact_stiffness"] = -float(np.polyfit(z_wall, f_wall, 1)[0])  # nN/Å > 0
        else:
            out["contact_stiffness"] = float("nan")
    else:
        # No repulsive contact at the start (curve did not begin loaded).
        out["contact_stiffness"] = float("nan")

    # Did the curve reach detachment (fz returns toward zero past the trough)?
    reached_min_before_end = i_min < z.size - 1
    tail_small = out["pulloff_force"] == 0.0 or abs(fz[-1]) <= 0.1 * out["pulloff_force"]
    out["detached"] = float(bool(reached_min_before_end and tail_small))
    return out


def _condition_from_path(csv_path: Path, run_root: Path) -> Optional[Tuple[str, int, float]]:
    """Extract ``(material, layer, load)`` from a retract CSV path/name."""
    m = _CSV_RE.search(csv_path.name)
    if not m:
        return None
    try:
        parts = csv_path.relative_to(run_root).parts
    except ValueError:
        parts = csv_path.parts
    if INDENT_TOPDIR not in parts:
        return None
    i = parts.index(INDENT_TOPDIR)
    if i + 1 >= len(parts):
        return None
    material = material_id(parts[i + 1])
    return material, int(m.group("layer")), float(m.group("load"))


def discover_curves(run_root: Path) -> List[Tuple[str, int, float, Path]]:
    """Find ``(material, layer, load, csv_path)`` for every retract curve."""
    found: List[Tuple[str, int, float, Path]] = []
    for csv_path in sorted(run_root.rglob("results/indent_f*N_l*.csv")):
        cond = _condition_from_path(csv_path, run_root)
        if cond is not None:
            found.append((*cond, csv_path))
    return found


def _deliver_adhesion(run_root: Path, out_dir: Path, data_dir: Path,
                      curves: List[Tuple[str, int, float, Path]]) -> Dict[str, Path]:
    """Write indent_features.csv + tidied retract curves from the retract data."""
    (out_dir / INDENT_TOPDIR).mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    written: Dict[str, Path] = {}

    # Whether more than one load per (material, layer) is present decides whether the
    # tidied-curve filename needs a load token to stay unique.
    per_ml: Dict[Tuple[str, int], int] = {}
    for material, layer, _load, _p in curves:
        per_ml[(material, layer)] = per_ml.get((material, layer), 0) + 1

    feature_rows: List[Dict[str, Any]] = []
    for material, layer, load_val, csv_path in curves:
        try:
            df = pd.read_csv(csv_path)
        except (OSError, pd.errors.ParserError) as exc:
            logger.warning("Skipping unreadable retract curve %s: %s", csv_path, exc)
            continue
        if len(df) < 3 or not {"z_tip", "fz_tip"} <= set(df.columns):
            logger.warning("Skipping degenerate retract curve %s (%d rows)", csv_path, len(df))
            continue

        feats = retract_features(df)
        if not feats.get("detached"):
            logger.warning(
                "Retract curve %s did not reach detachment (min at last steps) — "
                "consider a larger [indent] n_steps/z_step.", csv_path,
            )
        feature_rows.append({"material": material, "layer": layer, "load": load_val, **feats})

        # Tidied per-condition curve (recipe hand-back).
        stem = f"{material}_l{layer}"
        if per_ml[(material, layer)] > 1:
            stem += f"_f{_load_token(load_val)}"
        (df.to_csv(out_dir / INDENT_TOPDIR / f"{stem}.csv", index=False))

    if not feature_rows:
        logger.info("No usable indent retract curves under %s", run_root)
        return {}

    feat_df = pd.DataFrame(feature_rows).sort_values(
        ["material", "layer", "load"]).reset_index(drop=True)
    feat_path = data_dir / "indent_features.csv"
    feat_df.to_csv(feat_path, index=False)
    written["indent_features.csv"] = feat_path
    logger.info("Wrote %d indent feature row(s) -> %s", len(feat_df), feat_path)
    return written


# =============================================================================
# Combined manifest + entry point
# =============================================================================

def _write_manifest(out_dir: Path, conditions: List[Tuple[str, int, float]],
                    meta: Dict[str, Any]) -> Path:
    """One row per condition with the seed + hold/retract discretization."""
    wall_cache: Dict[str, Optional[int]] = {}
    rows: List[Dict[str, Any]] = []
    for material, layer, load_val in sorted(set(conditions)):
        if material not in wall_cache:
            rdir = meta["run_dir"].get(material)
            wall_cache[material] = _run_wall_seconds(rdir) if rdir else None
        rows.append({
            "material": material,
            "layer": layer,
            "load": load_val,
            "seed": meta["seed"].get((material, layer), ""),
            "hold_steps": meta["hold_steps"].get(material, ""),
            "z_step": meta["z_step"].get(material, ""),
            "n_steps": meta["n_steps"].get(material, ""),
            "wall_time": wall_cache[material] if wall_cache[material] is not None else "",
        })

    manifest_path = out_dir / "indent_manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=["material", "layer", "load", "seed",
                            "hold_steps", "z_step", "n_steps", "wall_time"])
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Wrote indent manifest (%d rows) -> %s", len(rows), manifest_path)
    return manifest_path


def run(run_root: Path, out_dir: Path, data_dir: Optional[Path] = None) -> Dict[str, Path]:
    """Reduce an indent run to penetration JSON + adhesion features + a manifest.

    Args:
        run_root: The ``simulation_YYYYMMDD_HHMMSS`` root (or its ``indent/``
            subdirectory) containing the hold + retract outputs.
        out_dir: Destination for ``output_indent_f*_s*.json``,
            ``indent/<material>_l<layer>.csv`` and ``indent_manifest.csv``.
        data_dir: Destination for ``indent_features.csv`` (default: ``out_dir``).

    Returns:
        Mapping of output name -> written path (empty if no indent data was found).
    """
    run_root = Path(run_root)
    out_dir = Path(out_dir)
    data_dir = Path(data_dir) if data_dir is not None else out_dir
    top = run_root / INDENT_TOPDIR if (run_root / INDENT_TOPDIR).is_dir() else run_root

    meta = load_indent_meta(run_root)
    curves = discover_curves(run_root)

    out_dir.mkdir(parents=True, exist_ok=True)
    written: Dict[str, Path] = {}

    pen_written, hold_conditions = _deliver_penetration(top, out_dir, meta)
    written.update(pen_written)

    written.update(_deliver_adhesion(run_root, out_dir, data_dir, curves))

    conditions = list(hold_conditions) + [(m, l, ld) for m, l, ld, _p in curves]
    if not conditions:
        logger.info("No indent results found under %s", run_root)
        return written
    written["indent_manifest.csv"] = _write_manifest(out_dir, conditions, meta)

    return written
