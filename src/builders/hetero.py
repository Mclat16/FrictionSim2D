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

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from ase import Atoms
from ase.build import make_supercell
from ase.data import atomic_masses, atomic_numbers, chemical_symbols
from ase.io import read, write

from ..core.config import GlobalSettings, SheetConfig
from ..core.lattice_matching import MatchResult, _max_strain, find_coincidence_lattice
from ..core.potential_manager import PotentialManager

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
