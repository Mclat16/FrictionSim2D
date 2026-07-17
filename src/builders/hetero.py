"""Build matched per-material supercells for heterostructure stacking.

Given two monolayer LAMMPS data files, :func:`build_matched_supercells`
computes their Zur-McGill coincidence lattice **on the exact in-plane cells it
is about to tile** (the monolayers' own orthogonalized cells, as read back from
disk), builds each material's coincidence supercell, and strains the
smaller-area supercell onto the larger (left unstrained), so both output
supercells end up sharing exactly ONE in-plane periodic box and can be stacked
into a single heterostructure cell.

Why the match is computed internally (the correctness crux)
-----------------------------------------------------------
:func:`ase.build.make_supercell` tiles a cell as ``new = matrix @ old``. A
coincidence match ``matrix_a``/``matrix_b`` is only meaningful for the *specific*
cell it was derived from: applying a match computed on one cell (e.g. a CIF
primitive ``P``) to a *different* cell (e.g. ``build_monolayer``'s orthogonalized
output ``T @ P``) produces ``matrix @ T @ P`` instead of the intended
``matrix @ P``. The spurious factor ``matrix·T·matrix⁻¹`` is identity only when
``matrix`` is a scalar multiple of identity; for any anisotropic or sheared
match (the generic twisted / different-lattice case) the two built supercells
are no longer commensurate. To guarantee ``matrix @ (tiled cell)`` is the true
coincidence supercell, the match is always found from the very cells that are
tiled.

Why the strain is applied in the co-oriented frame
--------------------------------------------------
``find_coincidence_lattice`` measures strain between the two supercells in their
*shared* (co-oriented) frame — the frame make_supercell produces. Writing each
supercell to a LAMMPS data file independently re-canonicalizes it to a1-along-x
with a *different* rotation per material, which decoheres a sheared match: a
``change_box`` remap (which cannot rotate) would then read the rotation
mismatch as a huge spurious strain. So the smaller supercell is strained onto
the larger in the co-oriented frame via ASE ``set_cell(scale_atoms=True)``, and
only then are *both* written out (canonicalized by the same rotation because
they now share an identical cell), so their on-disk boxes coincide exactly.
"""
from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from ase import Atoms
from ase.build import make_supercell
from ase.data import atomic_masses, atomic_numbers, chemical_symbols
from ase.io import read, write
from jinja2 import Environment

from ..core.config import GlobalSettings, SheetConfig, SheetOnSheetSimulationConfig
from ..core.lattice_matching import (
    MatchResult,
    _max_strain,
    find_coincidence_lattice,
    scan_registry_index,
)
from ..core.potential_manager import PotentialManager
from ..interfaces.jinja import PackageLoader
from .components import build_monolayer, calculate_layer_shifts

PathLike = Union[str, Path]

# The one-sided strain of straining the smaller supercell fully onto the larger
# is ~= strain_a + strain_b (each measured against the symmetric mean cell), so
# it is bounded by 2 x strain_tol for any match find_coincidence_lattice would
# accept. The default fail-loud bound uses exactly that principled factor.
_ONE_SIDED_STRAIN_FACTOR = 2.0


@dataclass
class MatchedStack:
    """Two per-material coincidence supercells sharing one in-plane box.

    Attributes:
        supercell_a: Path to material A's output supercell data file.
        supercell_b: Path to material B's output supercell data file.
        reference: Which input material ('a' or 'b') supplied the
            unstrained reference box -- the one with the larger in-plane
            coincidence-supercell area.
        strain_reference: Always 0.0; the reference material is never
            strained, only rotated/canonicalized into LAMMPS form.
        strain_applied: Max absolute principal strain applied to the
            non-reference material to reach the shared (reference) box.
        match: The coincidence-lattice match found (internally) between the two
            monolayers' actual in-plane cells.
    """
    supercell_a: Path
    supercell_b: Path
    reference: str
    strain_reference: float
    strain_applied: float
    match: MatchResult = field(default=None)  # type: ignore[assignment]


@dataclass
class HeteroPotentials:
    """Registered potential spine for a stacked heterostructure.

    Every material is its own :class:`PotentialManager` component with a
    disjoint, contiguous block of global LAMMPS atom-type IDs, its own
    many-body potential inside a single ``pair_style hybrid``, and UFF-mixed
    ``lj/cut`` cross-terms for every A-B element pair at the interface -- all
    emitted into ``system.in.settings``.

    Attributes:
        pm: The configured :class:`PotentialManager` (type registry +
            interactions already written).
        settings_path: Path to the written ``system.in.settings``.
        component_names: Component name per material, in stacking order
            (``['sheet_1', 'sheet_2', ...]``).
        type_ids: Per-material list of that material's global atom-type IDs, in
            stacking order. Read straight from the PM type registry
            (``pm.types.ids_by_component``); never hand-assigned. The blocks are
            disjoint and together cover ``1..len(pm.types)`` with no gaps.
        type_offsets: Per-material starting type offset (0-based) = the number
            of types registered before that material = ``type_ids[i][0] - 1``.
            Task 4 feeds this to ``read_data ... add append offset <k>`` so the
            assembled data file's atom types line up with this PM exactly.
    """
    pm: PotentialManager
    settings_path: Path
    component_names: List[str]
    type_ids: List[List[int]]
    type_offsets: List[int]


def register_hetero_potentials(
    sheets: List[SheetConfig],
    settings: GlobalSettings,
    workdir: PathLike,
) -> HeteroPotentials:
    """Register the stacked materials as PotentialManager components.

    This is the component/atom-type spine of the heterostructure: it is
    geometry-independent (it needs only the per-material configs, not any
    assembled data file), so it runs *before* stack assembly -- assembly then
    uses the returned potential to minimize the interface spacing, and reads
    ``type_offsets`` to keep the assembled data file's atom types aligned with
    this PM.

    Each material ``sheets[i]`` is registered as a single component
    ``sheet_{i+1}`` (one component sharing types across all its layers -- for
    SW materials the short many-body cutoff prevents spurious interlayer terms,
    so per-layer type expansion is neither needed nor wanted). Every component
    gets its own many-body ``add_self_interaction`` inside a ``pair_style
    hybrid``, and every distinct material pair gets a UFF-mixed
    ``add_cross_interaction`` (``lj/cut`` for each A-B element pair). The
    settings are written to ``<workdir>/lammps/system.in.settings`` and the
    potential files are staged under ``<workdir>/provenance/potentials``.

    Args:
        sheets: Per-material sheet configs (>= 2 materials).
        settings: Global simulation settings.
        workdir: Directory to write ``system.in.settings`` and the staged
            provenance potentials into (created if missing).

    Returns:
        :class:`HeteroPotentials` exposing the configured manager, the written
        settings path, and the per-material type blocks / offsets (derived from
        the PM type registry, not hand-computed).

    Raises:
        ValueError: If fewer than two materials are supplied (a heterostructure
            needs at least two).
    """
    if len(sheets) < 2:
        raise ValueError(
            f"register_hetero_potentials needs >= 2 materials to build a "
            f"heterostructure interface, got {len(sheets)}."
        )

    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    # Mirror the AFM / sheet-on-sheet potential path exactly: stage the real
    # potential files under the run dir's provenance folder and reference them
    # by a run-dir-relative prefix in the pair_coeff lines.
    pm = PotentialManager(
        settings,
        potentials_dir=workdir / "provenance" / "potentials",
        potentials_prefix=str(Path("provenance") / "potentials"),
    )

    # 1. One component per material -> disjoint, contiguous global type blocks
    #    (the PM's TypeRegistry assigns the IDs; we never hand-roll them).
    component_names: List[str] = []
    for i, sheet in enumerate(sheets):
        name = f"sheet_{i + 1}"
        pm.register_component(name, sheet, n_layers=1)
        component_names.append(name)

    # 2. Each material's own many-body potential (hybrid entry per component).
    for name in component_names:
        pm.add_self_interaction(name)

    # 3. UFF-mixed lj/cut cross-terms for every material pair at the interface.
    for i in range(len(component_names)):
        for j in range(i + 1, len(component_names)):
            pm.add_cross_interaction(component_names[i], component_names[j])

    settings_path = workdir / "lammps" / "system.in.settings"
    pm.write_file(settings_path)

    # 4. Read the per-material type blocks back from the registry (source of
    #    truth); derive read_data offsets from where each block starts.
    type_ids = [pm.types.ids_by_component(name) for name in component_names]
    type_offsets = [ids[0] - 1 for ids in type_ids]

    return HeteroPotentials(
        pm=pm,
        settings_path=settings_path,
        component_names=component_names,
        type_ids=type_ids,
        type_offsets=type_offsets,
    )


def _embed_3x3(matrix_2x2: np.ndarray) -> np.ndarray:
    """Embed a 2x2 integer in-plane supercell matrix into a 3x3 matrix.

    ``ase.build.make_supercell`` requires a full 3x3 integer matrix; the
    z-row/column is left as identity so the out-of-plane lattice vector
    (and periodicity) is left untouched.
    """
    p = np.eye(3, dtype=int)
    p[:2, :2] = np.asarray(matrix_2x2, dtype=int)
    return p


def _in_plane_area(cell_2x2: np.ndarray) -> float:
    """Absolute area (Å²) of a 2x2 row-vector in-plane cell."""
    return abs(float(np.linalg.det(np.asarray(cell_2x2, dtype=float))))


def _detect_atom_style(path: Path) -> str:
    """Detect the LAMMPS ``atom_style`` from a data file's ``Atoms`` header."""
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            stripped = line.strip()
            if stripped.startswith("Atoms"):
                if "#" in stripped:
                    style = stripped.split("#", 1)[1].strip()
                    return style or "atomic"
                return "atomic"
    return "atomic"


def _read_monolayer(path: Path) -> Dict[str, object]:
    """Parse a LAMMPS monolayer data file, preserving all metadata.

    Reads the box (incl. tilt), per-atom types + Cartesian positions, per-atom
    charges (``atom_style charge`` only), the Masses section (type -> mass,
    element), and the detected atom_style. This lets the supercell tiling carry
    every LAMMPS type through as a *distinct* placeholder element (so sublattice
    types that share a chemical element are not collapsed by ASE), and restore
    the real masses / charges / atom_style afterwards -- i.e. no metadata is
    lost the way a bare ASE read/write would drop it.

    Returns:
        Dict with keys: ``atom_style`` (str), ``cell`` (3x3 float array, rows =
        lattice vectors, LAMMPS-canonical), ``types`` (List[int]), ``positions``
        ((N,3) float, box-origin-relative), ``charges`` (List[float] or None),
        ``masses`` ({type: (mass, element)}), ``n_types`` (int).
    """
    atom_style = _detect_atom_style(path)
    is_charge = "charge" in atom_style

    bounds: Dict[str, float] = {}
    tilt = {"xy": 0.0, "xz": 0.0, "yz": 0.0}
    masses: Dict[int, Tuple[float, str]] = {}
    types: List[int] = []
    positions: List[Tuple[float, float, float]] = []
    charges: List[float] = []

    section: Optional[str] = None
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        low = line.split("#", 1)[0].strip()
        if low.endswith(("xlo xhi", "ylo yhi", "zlo zhi")):
            parts = low.split()
            key = parts[2][0]
            bounds[f"{key}lo"] = float(parts[0])
            bounds[f"{key}hi"] = float(parts[1])
            continue
        if low.endswith("xy xz yz"):
            parts = low.split()
            tilt["xy"], tilt["xz"], tilt["yz"] = float(parts[0]), float(parts[1]), float(parts[2])
            continue
        if line == "Masses" or line == "Atoms" or line.startswith("Atoms"):
            section = "Masses" if line == "Masses" else "Atoms"
            continue
        if line in ("Velocities", "Bonds", "Angles", "Dihedrals", "Impropers"):
            section = None
            continue
        if section == "Masses":
            parts = line.split()
            if parts[0].lstrip("-").isdigit():
                elem = line.split("#")[-1].strip().split()[0] if "#" in line else "X"
                masses[int(parts[0])] = (float(parts[1]), elem)
        elif section == "Atoms":
            parts = low.split()
            if is_charge:
                if len(parts) >= 6 and parts[0].lstrip("-").isdigit():
                    types.append(int(parts[1]))
                    charges.append(float(parts[2]))
                    positions.append((float(parts[3]), float(parts[4]), float(parts[5])))
            else:
                if len(parts) >= 5 and parts[0].lstrip("-").isdigit():
                    types.append(int(parts[1]))
                    positions.append((float(parts[2]), float(parts[3]), float(parts[4])))

    lx = bounds["xhi"] - bounds["xlo"]
    ly = bounds["yhi"] - bounds["ylo"]
    lz = bounds["zhi"] - bounds["zlo"]
    cell = np.array([
        [lx, 0.0, 0.0],
        [tilt["xy"], ly, 0.0],
        [tilt["xz"], tilt["yz"], lz],
    ])
    origin = np.array([bounds["xlo"], bounds["ylo"], bounds["zlo"]])
    pos = np.array(positions, dtype=float) - origin
    n_types = max([*types, *masses.keys()], default=1)

    return {
        "atom_style": atom_style,
        "cell": cell,
        "types": types,
        "positions": pos,
        "charges": charges if is_charge else None,
        "masses": masses,
        "n_types": int(n_types),
    }


def _monolayer_to_atoms(mono: Dict[str, object]) -> Atoms:
    """Build an ASE ``Atoms`` from parsed monolayer data using placeholder elements.

    Each distinct LAMMPS type ``t`` is carried as chemical element number ``t``
    (H, He, Li, ...) so that ``make_supercell`` never collapses two LAMMPS types
    that happen to share a real element. Per-atom charges are attached when the
    source used ``atom_style charge``.
    """
    types: List[int] = mono["types"]  # type: ignore[assignment]
    placeholders = [chemical_symbols[t] for t in types]
    atoms = Atoms(
        symbols=placeholders,
        positions=mono["positions"],
        cell=mono["cell"],
        pbc=True,
    )
    if mono["charges"] is not None:
        atoms.set_initial_charges(mono["charges"])  # type: ignore[arg-type]
    return atoms


def _restore_masses_section(path: Path, masses: Dict[int, Tuple[float, str]], n_types: int) -> None:
    """Insert a real ``Masses`` section (type mass # element) into a data file.

    ``ase.io.write(format="lammps-data")`` omits masses; restoring them keeps
    the file usable by LAMMPS and by downstream element parsing. Falls back to
    the placeholder element's standard mass for any type absent from the source
    Masses section (should not happen for build_monolayer output).
    """
    if n_types < 1:
        return
    block = "Masses\n\n"
    for t in range(1, n_types + 1):
        if t in masses:
            mass, elem = masses[t]
        else:
            elem = chemical_symbols[t]
            mass = float(atomic_masses[atomic_numbers[elem]])
        block += f"{t} {mass:.6f}  # {elem}\n"
    block += "\n"

    text = path.read_text(encoding="utf-8")
    import re  # local import: only needed here

    # Insert immediately before the Atoms section header.
    new_text, count = re.subn(r"(^|\n)(Atoms)", r"\1" + block + r"\2", text, count=1)
    if count:
        path.write_text(new_text, encoding="utf-8")


def _write_supercell(
    atoms: Atoms,
    out_path: Path,
    atom_style: str,
    n_types: int,
    masses: Dict[int, Tuple[float, str]],
) -> np.ndarray:
    """Write an ASE supercell as a LAMMPS data file, preserving metadata.

    The type-ordered placeholder elements are written via ``specorder`` so the
    original integer LAMMPS type ids are reproduced exactly; ``atom_style`` and
    per-atom charges round-trip through ASE; the real Masses section is restored
    afterwards. ASE canonicalizes the box to LAMMPS form (a1 along +x) on write.

    Returns:
        The written in-plane 2x2 cell (LAMMPS-canonical), read back from disk.
    """
    if out_path.exists():
        out_path.unlink()
    specorder = [chemical_symbols[t] for t in range(1, n_types + 1)]
    write(str(out_path), atoms, format="lammps-data", atom_style=atom_style, specorder=specorder)
    _restore_masses_section(out_path, masses, n_types)
    return np.array(read(str(out_path), format="lammps-data").cell)[:2, :2]


def build_matched_supercells(
    mono_a_path: PathLike,
    mono_b_path: PathLike,
    workdir: PathLike,
    strain_tol: float = 0.02,
    max_supercell: int = 10,
    area_tol: float = 0.10,
    max_strain: Optional[float] = None,
) -> MatchedStack:
    """Build A's and B's coincidence supercells sharing one in-plane box.

    Reads each monolayer's *actual* in-plane cell from disk, finds their
    Zur-McGill coincidence lattice on those exact cells (so the integer match
    matrices are consistent with the cells that get tiled), tiles each material
    with its match matrix via :func:`ase.build.make_supercell`, picks whichever
    supercell has the larger in-plane area as an unstrained reference, and
    strains the other supercell onto the reference's box **in the co-oriented
    frame** so both output files share exactly one LAMMPS-canonical in-plane box.

    Args:
        mono_a_path: LAMMPS data file for material A's monolayer.
        mono_b_path: LAMMPS data file for material B's monolayer.
        workdir: Directory to write the two output supercell data files into
            (created if missing).
        strain_tol: Max per-material strain (vs the symmetric mean cell) allowed
            when searching for a coincidence match. Forwarded to
            :func:`~src.core.lattice_matching.find_coincidence_lattice`.
        max_supercell: Max supercell repeat along each axis for the match search.
        area_tol: Relative supercell-area agreement tolerance for the search.
        max_strain: Fail-loud bound on the *one-sided* strain actually applied
            to the smaller supercell (measured geometrically, not estimated). A
            match accepted by ``find_coincidence_lattice`` implies a one-sided
            strain <= ``2 * strain_tol``; ``max_strain`` defaults to exactly that
            bound. Straining beyond it raises ``ValueError``.

    Returns:
        :class:`MatchedStack` with both output supercell paths (ordered A then
        B, regardless of which was the reference), which material was left
        unstrained, the strain applied to the other, and the match found.

    Raises:
        ValueError: If no coincidence lattice is found within the given
            tolerances, or if the geometric strain needed to make the two
            supercells share one box exceeds ``max_strain`` (i.e. the pair is
            not actually commensurate at this match -- fail loud rather than
            silently emit an incommensurate "matched" stack).
    """
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    mono_a_path = Path(mono_a_path)
    mono_b_path = Path(mono_b_path)

    if max_strain is None:
        max_strain = _ONE_SIDED_STRAIN_FACTOR * strain_tol

    # 1. Read each monolayer (metadata-preserving) and its ACTUAL in-plane cell.
    mono = {"a": _read_monolayer(mono_a_path), "b": _read_monolayer(mono_b_path)}
    cell_mono = {label: np.asarray(m["cell"])[:2, :2] for label, m in mono.items()}

    # 2. Find the coincidence match on the very cells that will be tiled.
    match = find_coincidence_lattice(
        cell_mono["a"], cell_mono["b"],
        strain_tol=strain_tol, max_supercell=max_supercell, area_tol=area_tol,
    )
    if match is None:
        raise ValueError(
            f"No coincidence lattice found between '{mono_a_path.name}' and "
            f"'{mono_b_path.name}' within strain_tol={strain_tol:.4%}, "
            f"max_supercell={max_supercell}, area_tol={area_tol:.2%}. The two "
            f"monolayers are not commensurate under these constraints."
        )

    matrices = {"a": match.matrix_a, "b": match.matrix_b}

    # 3. Tile each monolayer into its coincidence supercell (co-oriented frame).
    atoms = {label: _monolayer_to_atoms(mono[label]) for label in ("a", "b")}
    sc_atoms = {
        label: make_supercell(atoms[label], _embed_3x3(matrices[label]))
        for label in ("a", "b")
    }
    sc_cell = {label: np.array(sc_atoms[label].cell)[:2, :2] for label in ("a", "b")}
    areas = {label: _in_plane_area(sc_cell[label]) for label in ("a", "b")}

    # 4. Larger-area supercell is the unstrained reference.
    ref_label = "a" if areas["a"] >= areas["b"] else "b"
    other_label = "b" if ref_label == "a" else "a"
    ref_cell = sc_cell[ref_label]

    # 5. Measure the real one-sided strain of straining the smaller onto the
    #    larger (both in the shared, co-oriented make_supercell frame), and fail
    #    loud if it exceeds the principled asymmetric bound.
    strain = _max_strain(ref_cell, sc_cell[other_label])
    if strain > max_strain:
        raise ValueError(
            f"Cannot build a commensurate heterostructure stack: straining "
            f"material '{other_label}' ({mono[other_label]['atom_style']}, "
            f"{(mono_a_path if other_label == 'a' else mono_b_path).name}) onto "
            f"material '{ref_label}'s reference box requires {strain:.4%} strain, "
            f"exceeding max_strain={max_strain:.4%} "
            f"(= {_ONE_SIDED_STRAIN_FACTOR:g} x strain_tol={strain_tol:.4%})."
        )

    # 6. Strain the smaller supercell onto the reference box IN THE CO-ORIENTED
    #    FRAME (set_cell with scale_atoms is a pure affine remap that includes
    #    the small aligning rotation), then write BOTH. Since both now carry an
    #    identical in-plane cell, ASE canonicalizes them by the same rotation, so
    #    their on-disk boxes coincide exactly.
    other_atoms = sc_atoms[other_label]
    lz = np.array(other_atoms.cell)[2, 2]
    aligned_cell = np.array([
        [ref_cell[0, 0], ref_cell[0, 1], 0.0],
        [ref_cell[1, 0], ref_cell[1, 1], 0.0],
        [0.0, 0.0, lz],
    ])
    other_atoms.set_cell(aligned_cell, scale_atoms=True)

    outputs: Dict[str, Path] = {}
    for label, at in ((ref_label, sc_atoms[ref_label]), (other_label, other_atoms)):
        out_path = workdir / f"{label}_supercell.lmp"
        _write_supercell(
            at, out_path,
            atom_style=str(mono[label]["atom_style"]),
            n_types=int(mono[label]["n_types"]),
            masses=mono[label]["masses"],  # type: ignore[arg-type]
        )
        outputs[label] = out_path

    return MatchedStack(
        supercell_a=outputs["a"],
        supercell_b=outputs["b"],
        reference=ref_label,
        strain_reference=0.0,
        strain_applied=strain,
        match=match,
    )


# =============================================================================
# Stack assembly: two matched supercells -> one periodic hetero cell
# =============================================================================


@dataclass
class HeteroStackLayer:
    """One placed layer in the assembled heterostructure.

    Attributes:
        material_index: 0-based material index (0 = supercell_a, 1 = supercell_b).
        layer_index: 0-based layer within that material.
        z: z-shift (Å) applied to this layer's supercell when read into the box.
        xy_offset: In-plane displacement (Å) applied to this layer = the
            material's cross-material RI anchor + this layer's intra-material
            AA/AB shift. The reference material's anchor is (0, 0); the
            non-reference material's anchor is the surviving RI offset.
        group: LAMMPS group name assigned to this layer's atoms.
        type_offset: read_data atom-type offset used for this layer (from the
            PotentialManager, so assembled global types match the spine).
        source: Path to the per-material supercell data file this layer was
            read from.
    """
    material_index: int
    layer_index: int
    z: float
    xy_offset: Tuple[float, float]
    group: str
    type_offset: int
    source: Path


@dataclass
class HeteroStack:
    """Result of assembling the matched supercells into one periodic cell.

    Attributes:
        data_path: Path to the single written ``hetero.data`` (all layers of
            both materials in one shared in-plane periodic box).
        settings_path: The real potential settings (carried through from
            :class:`HeteroPotentials`) that describe the written types/masses.
        layers: Per-layer bookkeeping (material/layer index, z, xy offset,
            group, type offset, source) for a later sim builder.
        ri_offset: The cross-material Registry-Index offset (Å) actually applied
            to the non-reference material's layers (one-sided; NOT
            centroid-cancelled). Non-zero for AB / off-registry stackings.
        reference_index: Material index left unshifted in-plane (the larger,
            unstrained reference lattice from Task 3).
        nonreference_index: Material index carrying ``ri_offset``.
        interface_spacing: Cross-material interlayer COM z-gap (Å) found by real
            minimization with the registered potential.
        stacking: The requested layer ordering ('grouped' or 'alternating').
        total_types: Number of global atom types in the assembled cell
            (== len(pm.types)).
        material_offsets: Per-material in-plane anchor (Å) applied to all of
            that material's layers.
    """
    data_path: Path
    settings_path: Path
    layers: List[HeteroStackLayer]
    ri_offset: np.ndarray
    reference_index: int
    nonreference_index: int
    interface_spacing: float
    stacking: str
    total_types: int
    material_offsets: Dict[int, np.ndarray]


def _run_lammps(workdir: Path, script_path: Path) -> str:
    """Run ``lmp_serial`` on a script with ``cwd = workdir`` and return stdout.

    Running from ``workdir`` is mandatory: the settings file's ``pair_coeff``
    lines reference the staged potentials by the run-dir-relative prefix
    ``provenance/potentials/...``, so those paths only resolve when LAMMPS'
    working directory is the run dir.

    Raises:
        RuntimeError: If LAMMPS exits non-zero (with a tail of stdout/stderr).
    """
    result = subprocess.run(
        ["lmp_serial", "-in", str(script_path), "-log", "none"],
        cwd=str(workdir),
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        tail_out = "\n".join(result.stdout.splitlines()[-30:])
        tail_err = "\n".join(result.stderr.splitlines()[-30:])
        raise RuntimeError(
            f"LAMMPS run failed (exit {result.returncode}) for "
            f"'{script_path.name}' in {workdir}.\n"
            f"--- stdout (tail) ---\n{tail_out}\n"
            f"--- stderr (tail) ---\n{tail_err}"
        )
    return result.stdout


def _box_region_cmd(lx: float, ly: float, xy: float, zlo: float, zhi: float) -> str:
    """LAMMPS ``region box ...`` command reproducing the shared in-plane box.

    Uses a ``prism`` region when the shared box is sheared (non-zero xy tilt),
    else a plain ``block`` region.
    """
    if abs(xy) > 1e-9:
        return (
            f"region box prism 0.0 {lx:.6f} 0.0 {ly:.6f} "
            f"{zlo:.6f} {zhi:.6f} {xy:.6f} 0.0 0.0"
        )
    return f"region box block 0.0 {lx:.6f} 0.0 {ly:.6f} {zlo:.6f} {zhi:.6f}"


def _read_layer_geometry(path: Path) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """Return (in-plane Cartesian xy, 2x2 cell, z-min, z-max) of a supercell."""
    at = read(str(path), format="lammps-data")
    pos = at.get_positions()
    cell2 = np.array(at.cell)[:2, :2]
    return pos[:, :2], cell2, float(pos[:, 2].min()), float(pos[:, 2].max())


def _minimize_interface_gap(
    *,
    workdir: Path,
    settings: GlobalSettings,
    settings_path: Path,
    atom_style: str,
    total_types: int,
    masses_block: str,
    lx: float,
    ly: float,
    xy: float,
    bot_path: Path,
    bot_off: int,
    bot_zext: Tuple[float, float],
    top_path: Path,
    top_off: int,
    top_zext: Tuple[float, float],
    top_shift_xy: np.ndarray,
    initial_guess: float,
) -> float:
    """Find one interface's interlayer COM z-gap by REAL minimization.

    Builds a 2-layer trial (bottom layer at z=0, top layer at ``initial_guess``
    with the interface's relative in-plane offset applied), includes the real
    registered potential, minimizes at fixed box, and reads the COM z-gap
    ``xcm(g2,z) - xcm(g1,z)`` -- mirroring the ``stack_multilayer_sheet`` idiom.
    """
    zlo = min(bot_zext[0], top_zext[0]) - 10.0
    zhi = max(bot_zext[1], top_zext[1] + initial_guess) + 10.0
    box_region = _box_region_cmd(lx, ly, xy, zlo, zhi)

    dx, dy = float(top_shift_xy[0]), float(top_shift_xy[1])
    disp = ""
    if abs(dx) > 1e-12 or abs(dy) > 1e-12:
        disp = f"displace_atoms  g2 move {dx:.6f} {dy:.6f} 0.0 units box\n"

    script = f"""clear
units           metal
atom_style      {atom_style}
boundary        p p p

{box_region}
create_box      {total_types} box

read_data       {bot_path} add append shift 0.0 0.0 0.0 group g1 offset {bot_off} 0 0 0 0 nocoeff
read_data       {top_path} add append shift 0.0 0.0 {initial_guess:.6f} group g2 offset {top_off} 0 0 0 0 nocoeff
{disp}{masses_block}
include         {settings_path}
run             0

min_style       {settings.simulation.min_style}
{settings.simulation.minimization_command}
variable        z1 equal xcm(g1,z)
variable        z2 equal xcm(g2,z)
variable        lat_c equal v_z2-v_z1
print           "RESULT_LAT_C ${{lat_c}}"
"""
    script_path = workdir / f"min_interface_{bot_off}_{top_off}.lmp"
    script_path.write_text(script, encoding="utf-8")
    out = _run_lammps(workdir, script_path)

    value: Optional[float] = None
    for line in out.splitlines():
        if line.startswith("RESULT_LAT_C"):
            value = float(line.split()[1])
    if value is None:
        raise RuntimeError(
            f"Could not parse interface spacing (RESULT_LAT_C) from minimize "
            f"trial '{script_path.name}'."
        )
    return abs(value)


def assemble_hetero_stack(
    matched: MatchedStack,
    sheets: List[SheetConfig],
    hetero_potentials: HeteroPotentials,
    hetero_stacking: str,
    settings: GlobalSettings,
    workdir: PathLike,
) -> HeteroStack:
    """Stack two matched supercells into ONE genuinely-periodic LAMMPS cell.

    Consumes the two per-material coincidence supercells (which already share
    one in-plane box, Task 3) and the registered potential spine (Task 5), and
    writes a single ``hetero.data`` with:

    * **Cross-material registry (RI offset).** :func:`scan_registry_index` on the
      two interface layers gives the AA (max-RI) / AB (min-RI) lateral offset;
      the choice follows the non-reference (top) material's ``stack_type``. The
      offset is applied as a **one-sided relative in-plane displacement of the
      non-reference material's layers only** -- the reference stays at the
      origin and the two materials are NEVER re-centered to a common centroid
      (the old branch's bug, which cancelled the offset).
    * **Per-interface z-spacing by real minimization.** For the cross-material
      interface a 2-layer trial is minimized with the real registered potential
      (:func:`_minimize_interface_gap`); the resulting COM z-gap is the interface
      spacing (no fixed nominal gap).
    * **Layer ordering.** ``'grouped'`` = all of material 0's layers then all of
      material 1's; ``'alternating'`` interleaves. Each material's own
      :func:`calculate_layer_shifts` gives its intra-material AA/AB shifts.
    * **Exact type alignment.** Each layer is read with its
      ``hetero_potentials.type_offsets`` so assembled global types match the PM.

    Every LAMMPS invocation runs with ``cwd = workdir`` so the settings file's
    run-dir-relative ``provenance/potentials/...`` references resolve.

    Args:
        matched: The two matched per-material supercells (Task 3).
        sheets: Per-material sheet configs (exactly two; index 0 = supercell_a).
        hetero_potentials: Registered potential spine (Task 5) supplying the
            settings path and per-material type offsets.
        hetero_stacking: ``'grouped'`` or ``'alternating'``.
        settings: Global simulation settings.
        workdir: Run directory (also where ``hetero.data`` is written).

    Returns:
        :class:`HeteroStack` with the written data path, the carried settings
        path, per-layer bookkeeping, and the surviving RI offset.

    Raises:
        ValueError: On an unknown stacking, a material count other than two, an
            empty layers list, or mixed atom styles.
        RuntimeError: If a LAMMPS run fails or does not produce ``hetero.data``.
    """
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    stacking = hetero_stacking.lower()
    if stacking not in ("grouped", "alternating"):
        raise ValueError(
            f"hetero_stacking must be 'grouped' or 'alternating', got "
            f"{hetero_stacking!r}."
        )

    n_mats = len(sheets)
    if n_mats != 2:
        raise ValueError(
            f"assemble_hetero_stack builds one interface between exactly two "
            f"matched materials (MatchedStack holds a pair); got {n_mats} sheets."
        )

    mat_paths = [Path(matched.supercell_a), Path(matched.supercell_b)]
    type_offsets = list(hetero_potentials.type_offsets)

    n_layers: List[int] = []
    for sheet in sheets:
        if not sheet.layers:
            raise ValueError(f"SheetConfig for {sheet.mat!r} has an empty layers list.")
        n_layers.append(int(sheet.layers[0]))

    ref_index = 0 if matched.reference == "a" else 1
    nonref_index = 1 - ref_index

    styles = {_detect_atom_style(p) for p in mat_paths}
    if len(styles) > 1:
        raise ValueError(
            f"Cannot assemble materials with differing atom_style {sorted(styles)} "
            f"into one data file."
        )
    atom_style = styles.pop()

    # In-plane geometry (shared box) + per-material atom xy / z-extent.
    carts: Dict[int, np.ndarray] = {}
    z_ext: Dict[int, Tuple[float, float]] = {}
    cell2: Optional[np.ndarray] = None
    for m, path in enumerate(mat_paths):
        cart_xy, this_cell2, zmn, zmx = _read_layer_geometry(path)
        carts[m] = cart_xy
        z_ext[m] = (zmn, zmx)
        if cell2 is None:
            cell2 = this_cell2
        elif not np.allclose(this_cell2, cell2, atol=1e-4):
            # The matched supercells MUST share one in-plane box (build_matched_
            # supercells' guarantee). If they don't, stacking them into one cell
            # would create a periodic seam -- fail loud rather than emit a
            # silently mismatched structure.
            raise ValueError(
                "Matched supercells do not share one in-plane box "
                f"(material {m} cell {this_cell2.tolist()} != {cell2.tolist()}); "
                "cannot assemble a seamless periodic stack."
            )
    assert cell2 is not None
    lx, ly, xy = float(cell2[0, 0]), float(cell2[1, 1]), float(cell2[1, 0])

    # Cross-material RI anchor: scan shifts its SECOND argument, so pass the
    # reference material first and the non-reference second -> the returned
    # offset displaces the non-reference material's layers (one-sided).
    ri_result = scan_registry_index(carts[ref_index], carts[nonref_index], cell2)
    top_stack_type = sheets[nonref_index].stack_type.upper()
    ri_offset = np.array(
        ri_result.aa_shift if top_stack_type == "AA" else ri_result.ab_shift,
        dtype=float,
    )
    material_anchor: Dict[int, np.ndarray] = {
        ref_index: np.zeros(2, dtype=float),
        nonref_index: ri_offset,
    }

    # Intra-material AA/AB shifts (independent of the cross-material RI offset).
    dims = {"xlo": 0.0, "xhi": lx, "ylo": 0.0, "yhi": ly, "zlo": 0.0, "zhi": 1.0}
    intra_shifts: Dict[int, List[Tuple[float, float]]] = {}
    for m in range(n_mats):
        intra_shifts[m] = calculate_layer_shifts(
            sheets[m].mat, dims, n_layers=n_layers[m], stacking_type=sheets[m].stack_type
        )

    # Layer ordering.
    if stacking == "grouped":
        seq = [(m, k) for m in range(n_mats) for k in range(n_layers[m])]
    else:  # alternating
        seq = []
        for k in range(max(n_layers)):
            for m in range(n_mats):
                if k < n_layers[m]:
                    seq.append((m, k))

    total_types = len(hetero_potentials.pm.types)
    masses_block = hetero_potentials.pm.get_masses_string()
    settings_path = Path(hetero_potentials.settings_path)
    initial_guess = settings.geometry.lat_c_default

    # Per-interface spacing by real minimization, cached per distinct pair.
    gap_cache: Dict[Tuple[int, int], float] = {}

    def _gap(bot_m: int, top_m: int) -> float:
        key = (bot_m, top_m)
        if key in gap_cache:
            return gap_cache[key]
        if bot_m == top_m:
            if n_layers[bot_m] > 1:
                rel = np.array(intra_shifts[bot_m][1], dtype=float) - np.array(
                    intra_shifts[bot_m][0], dtype=float
                )
            else:
                rel = np.zeros(2, dtype=float)
        else:
            rel = material_anchor[top_m] - material_anchor[bot_m]
        gap = _minimize_interface_gap(
            workdir=workdir,
            settings=settings,
            settings_path=settings_path,
            atom_style=atom_style,
            total_types=total_types,
            masses_block=masses_block,
            lx=lx,
            ly=ly,
            xy=xy,
            bot_path=mat_paths[bot_m],
            bot_off=type_offsets[bot_m],
            bot_zext=z_ext[bot_m],
            top_path=mat_paths[top_m],
            top_off=type_offsets[top_m],
            top_zext=z_ext[top_m],
            top_shift_xy=rel,
            initial_guess=initial_guess,
        )
        gap_cache[key] = gap
        return gap

    # Place layers in z, accumulating per-interface gaps.
    layers_spec: List[HeteroStackLayer] = []
    z_cursor = 0.0
    prev_m: Optional[int] = None
    interface_spacing: Optional[float] = None
    for idx, (m, k) in enumerate(seq):
        if idx == 0:
            z_layer = 0.0
        else:
            assert prev_m is not None
            gap = _gap(prev_m, m)
            z_cursor += gap
            z_layer = z_cursor
            if prev_m != m and interface_spacing is None:
                interface_spacing = gap
        anchor = material_anchor[m]
        intra = intra_shifts[m][k] if k < len(intra_shifts[m]) else (0.0, 0.0)
        layers_spec.append(
            HeteroStackLayer(
                material_index=m,
                layer_index=k,
                z=z_layer,
                xy_offset=(float(anchor[0] + intra[0]), float(anchor[1] + intra[1])),
                group=f"_m{m}l{k}",
                type_offset=int(type_offsets[m]),
                source=mat_paths[m],
            )
        )
        prev_m = m

    # Box z-window from the actual (shifted) atom extents so nothing wraps.
    zmins = [z_ext[s.material_index][0] + s.z for s in layers_spec]
    zmaxs = [z_ext[s.material_index][1] + s.z for s in layers_spec]
    box_region = _box_region_cmd(lx, ly, xy, min(zmins) - 10.0, max(zmaxs) + 10.0)

    # Render + run the assembly.
    output_path = workdir / "hetero.data"
    layers_ctx = [
        {
            "material_index": s.material_index,
            "layer_index": s.layer_index,
            "z": s.z,
            "path": str(s.source),
            "group": s.group,
            "type_offset": s.type_offset,
            "shift_x": s.xy_offset[0],
            "shift_y": s.xy_offset[1],
        }
        for s in layers_spec
    ]
    context = {
        "atom_style": atom_style,
        "box_region": box_region,
        "total_types": total_types,
        "layers": layers_ctx,
        "masses_block": masses_block,
        "settings_path": str(settings_path),
        "output_path": str(output_path),
    }
    env = Environment(
        loader=PackageLoader("src.templates"), trim_blocks=True, lstrip_blocks=True
    )
    template = env.get_template("hetero/stack_hetero.lmp")
    script_path = workdir / "stack_hetero.lmp"
    script_path.write_text(template.render(context), encoding="utf-8")
    _run_lammps(workdir, script_path)

    if not output_path.exists():
        raise RuntimeError(
            f"assemble_hetero_stack: LAMMPS did not produce {output_path}."
        )

    return HeteroStack(
        data_path=output_path,
        settings_path=settings_path,
        layers=layers_spec,
        ri_offset=ri_offset,
        reference_index=ref_index,
        nonreference_index=nonref_index,
        interface_spacing=float(interface_spacing) if interface_spacing is not None else 0.0,
        stacking=stacking,
        total_types=total_types,
        material_offsets={m: material_anchor[m] for m in range(n_mats)},
    )


# =============================================================================
# Top-level entrypoint: config -> ready heterostructure (Tasks 3-5 tied together)
# =============================================================================


def build_hetero_structure(
    config: SheetOnSheetSimulationConfig,
    settings: GlobalSettings,
    workdir: PathLike,
) -> HeteroStack:
    """Build a ready heterostructure stack from a two-material config.

    This is the single entrypoint a later sim builder calls to obtain a
    self-consistent hetero stack (``hetero.data`` + ``system.in.settings``). It
    ties the Phase-B structure-builder pieces together, in the order the pieces
    require:

    1. **Monolayers.** Build each material's orthogonalized monolayer with the
       existing :func:`~src.builders.components.build_monolayer` (first return
       value is the data-file path).
    2. **Match (Task 3).** :func:`build_matched_supercells` finds the two
       monolayers' coincidence lattice on their actual in-plane cells, tiles
       each, and strains the smaller-area supercell onto the larger so both
       output supercells share exactly one in-plane periodic box.
    3. **Register potentials (Task 5).** :func:`register_hetero_potentials` is
       geometry-independent, so it runs before assembly: it registers each
       material as its own :class:`PotentialManager` component (disjoint,
       contiguous atom-type blocks), writes ``system.in.settings``, and exposes
       the per-material ``type_offsets`` assembly needs.
    4. **Assemble (Task 4).** :func:`assemble_hetero_stack` stacks the two matched
       supercells into one genuinely-periodic ``hetero.data`` -- applying the
       cross-material RI offset one-sided, finding the interface z-spacing by
       real minimization, and aligning atom types to the registered spine.

    Scope: the **two-material** case (one interface) -- what the coincidence
    match, the registry scan, and the single-interface assembler all handle. A
    general N-material generalization is intentionally out of scope here.

    Args:
        config: Two-material sheet-on-sheet config. ``config.sheets`` supplies
            the per-material :class:`SheetConfig` list (from the ``[2D-1]`` /
            ``[2D-2]`` sections) and ``config.general.hetero_stacking`` the layer
            ordering (``'grouped'`` | ``'alternating'``).
        settings: Global simulation settings, threaded to every sub-step (in
            practice the same object as ``config.settings``; the passed
            ``settings`` is authoritative).
        workdir: Run directory. All artefacts (per-material supercells,
            ``lammps/system.in.settings``, staged ``provenance/potentials``, and
            the final ``hetero.data``) are written under it.

    Returns:
        The assembled :class:`HeteroStack`; ``.data_path`` and ``.settings_path``
        are the ready heterostructure structure + potentials.

    Raises:
        ValueError: If fewer than two materials are configured (a
            heterostructure needs an interface between two materials).
    """
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    sheets: List[SheetConfig] = list(config.sheets)
    if len(sheets) < 2:
        raise ValueError(
            f"build_hetero_structure needs >= 2 materials (a heterostructure "
            f"interface); config.sheets has {len(sheets)}."
        )

    # 1. Monolayer per material (build_monolayer returns the path as element 0).
    mono_paths = [Path(build_monolayer(sheet)[0]) for sheet in sheets]

    # 2. Match + strain smaller onto larger -> two supercells, one shared box.
    #    Thread the optional [lattice_match] config through to the coincidence
    #    search; when unset, build_matched_supercells' own defaults apply. This
    #    is the knob a user needs to build genuinely different-lattice pairs
    #    (a wider strain_tol / larger max_supercell than the 2% / 10 defaults).
    lm = config.lattice_match
    match_kwargs = (
        dict(strain_tol=lm.strain_tol, max_supercell=lm.max_supercell, area_tol=lm.area_tol)
        if lm is not None
        else {}
    )
    matched = build_matched_supercells(mono_paths[0], mono_paths[1], workdir, **match_kwargs)

    # 3. Register the materials' potential/type spine (geometry-independent).
    hetero_potentials = register_hetero_potentials(sheets, settings, workdir)

    # 4. Assemble into one periodic hetero.data aligned to that spine.
    stack = assemble_hetero_stack(
        matched,
        sheets,
        hetero_potentials,
        config.general.hetero_stacking,
        settings,
        workdir,
    )
    return stack
