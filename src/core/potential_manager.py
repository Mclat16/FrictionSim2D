"""Potential management for FrictionSim2D.

This module handles the assignment and generation of interatomic potentials 
for LAMMPS simulations.
It supports hybrid pair styles with proper indexing, automatic mixing rules
for cross-interactions, and layer-specific potential handling for multi-layer
2D materials.

Supported potentials include:
    - With internal LJ: AIREBO, COMB, COMB3, ReaxFF, REBOMOS
    - Requiring explicit LJ: Stillinger-Weber (SW), Tersoff, REBO, EDIP, MEAM,
    EAM, BOP, Morse, SW/MOD, EXTEP, Vashishta
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Union, Iterator
from collections import defaultdict
import logging
import re
import shutil

from ase.data import atomic_masses, atomic_numbers

from .config import ComponentConfig, GlobalSettings
from .utils import count_atomtypes, lj_params, cifread

logger = logging.getLogger(__name__)

POTENTIALS_WITH_INTERNAL_LJ = {'airebo', 'rebomos', 'comb', 'comb3', 'reaxff', 'reax/c'}
POTENTIALS_REQUIRING_LJ = {'sw', 'tersoff', 'rebo', 'edip', 'meam', 'eam', 'bop',
                            'morse',  'sw/mod', 'extep', 'vashishta'}
REAXFF_TYPES = {'reaxff', 'reax/c'}


@dataclass
class AtomType:
    """Metadata for a single LAMMPS atom type.
    
    Attributes:
        type_id: Global LAMMPS type ID (1-indexed).
        component: Component name (e.g., 'tip', 'sheet', 'sub').
        element: Chemical element symbol (e.g., 'Mo', 'S').
        pot_label: Label for pair_coeff (e.g., 'Mo1', 'S', 'Si').
        layer: Layer index (0-indexed) for multi-layer materials, None otherwise.
        region: Langevin region suffix ('fix', 'thermo'), None for mobile/standard.
    """
    type_id: int
    component: str
    element: str
    pot_label: str
    layer: Optional[int] = None
    region: Optional[str] = None

    @property
    def group_name(self) -> str:
        """Generate LAMMPS-style group name for this type."""
        if self.layer is not None:
            return f"{self.component}_l{self.layer + 1}_t{self.type_id}"
        if self.region:
            return f"{self.component}_{self.region}_t{self.type_id}"
        return f"{self.component}_t{self.type_id}"


class TypeRegistry:
    """Centralized registry for atom types with query methods for LAMMPS pair_coeff generation."""

    def __init__(self):
        self._types: Dict[int, AtomType] = {}
        self._next_id: int = 1

    def register(self, component: str, element: str, pot_label: str,
                    layer: Optional[int] = None, region: Optional[str] = None) -> int:
        """Register a new atom type and return its ID."""
        type_id = self._next_id
        self._next_id += 1
        self._types[type_id] = AtomType(
            type_id=type_id,
            component=component,
            element=element,
            pot_label=pot_label,
            layer=layer,
            region=region
        )
        return type_id

    def __len__(self) -> int:
        return len(self._types)

    def __iter__(self) -> Iterator[AtomType]:
        """Iterate over all types in type_id order."""
        for tid in sorted(self._types.keys()):
            yield self._types[tid]

    # === Query methods ===

    def ids_by_component(self, component: str) -> List[int]:
        """All type IDs for a component."""
        return sorted([t.type_id for t in self._types.values() if t.component == component])

    def ids_by_component_element(self, component: str, element: str) -> List[int]:
        """Type IDs for a specific element in a component."""
        return sorted([t.type_id for t in self._types.values()
                if t.component == component and t.element == element])

    def ids_by_component_layer(self, component: str, layer: int) -> List[int]:
        """Type IDs for a specific layer of a component."""
        return sorted([t.type_id for t in self._types.values()
                if t.component == component and t.layer == layer])

    def ids_by_component_layer_element(self, component: str, layer: int, element: str) -> List[int]:
        """Type IDs for element in specific layer."""
        return sorted([t.type_id for t in self._types.values()
                if t.component == component and t.layer == layer and t.element == element])

    def ids_by_component_region(self, component: str, region: Optional[str]) -> List[int]:
        """Type IDs for a specific region of a component."""
        return sorted([t.type_id for t in self._types.values()
                if t.component == component and t.region == region])

    def elements_in_component(self, component: str) -> List[str]:
        """Unique elements in a component (preserving order of first occurrence)."""
        seen = set()
        result = []
        for t in self:
            if t.component == component and t.element not in seen:
                seen.add(t.element)
                result.append(t.element)
        return result

    def build_null_map(self, component: str, layer: Optional[int] = None) -> List[str]:
        """Build NULL-padded pot_label list for pair_coeff.
        
        Returns a list where each position corresponds to a type ID.
        Types matching the component (and optionally layer) get their pot_label,
        others get 'NULL'.
        """
        result = []
        for tid in range(1, self._next_id):
            atype = self._types.get(tid)
            if atype and atype.component == component:
                if layer is None or atype.layer == layer:
                    result.append(atype.pot_label)
                    continue
            result.append("NULL")
        return result

    def get_element_map(self, component: str) -> Dict[str, List[int]]:
        """Get element -> [type_ids] mapping for a component (for backward compatibility)."""
        result: Dict[str, List[int]] = defaultdict(list)
        for t in self._types.values():
            if t.component == component:
                result[t.element].append(t.type_id)
        return {k: sorted(v) for k, v in result.items()}

    def format_type_range(self, types: List[int]) -> str:
        """Format a list of type IDs into LAMMPS range notation.

        Args:
            types: List of type IDs.

        Returns:
            LAMMPS range notation string (e.g., '1*3' or '5').
        """
        if not types:
            return ""
        types = sorted(types)
        if len(types) == 1:
            return str(types[0])
        return f"{types[0]}*{types[-1]}"

    def get_element_type_range(self, component: str, element: str) -> str:
        """Get the LAMMPS type range string for an element in a component.

        Args:
            component: Component name.
            element: Element symbol.

        Returns:
            Format like '1*3' for types 1,2,3 or just '5' for single type.
        """
        all_types = self.ids_by_component_element(component, element)
        return self.format_type_range(all_types)

    def get_group_string(self, component: str) -> str:
        """Return space-separated type IDs for a component (for LAMMPS grouping).

        Args:
            component: Name of the component.

        Returns:
            Space-separated string of type IDs (e.g., "1 2 3").
        """
        all_ids = self.ids_by_component(component)
        return " ".join(map(str, all_ids))

    def get_layer_group_string(self, component: str, layer: int) -> str:
        """Return type IDs for a specific layer of a component.

        Args:
            component: Name of the component.
            layer: Layer index (0-based).

        Returns:
            Space-separated string of type IDs for that layer.
        """
        all_ids = self.ids_by_component_layer(component, layer)
        return " ".join(map(str, all_ids))

    def get_region_group_string(self, component: str, region: Optional[str]) -> str:
        """Return type IDs for a specific region of a component."""
        all_ids = self.ids_by_component_region(component, region)
        return " ".join(map(str, all_ids))

class PotentialManager:
    """Manages potential assignments, type mapping, and pair_coeff generation.

    This class handles:
        - Sequential atom type ID assignment across all components
        - Hybrid pair_style construction with proper indexing
        - Self-interactions (many-body potentials like Tersoff, SW)
        - Cross-interactions (LJ between different components)
        - Interlayer interactions for multi-layer 2D materials
        - Langevin thermostat region type expansion (3x types)

    Attributes:
        components: List of registered component data dictionaries.
        types: TypeRegistry containing all atom type metadata.
        potential_usage: Tracks how many times each potential type is used.
    """

    def __init__(self, settings: GlobalSettings,
                use_langevin: Optional[bool] = None,
                potentials_dir: Optional[Union[str, Path]] = None,
                potentials_prefix: Optional[str] = None):
        """Initialize the PotentialManager.

        Args:
            settings: Global simulation settings containing potential and
                thermostat config.
            use_langevin: Override for Langevin type expansion. If None, uses
                settings. Set to False for temporary calculations like lat_c
                finding.
            potentials_dir: Absolute path to where potential files should be copied.
            potentials_prefix: String prefix for pair_coeff paths (e.g.,
                'afm/.../potentials'). If None, falls back to the copied file path.
        """
        self.settings = settings

        self.potentials_dir: Optional[Path] = Path(potentials_dir).resolve() if potentials_dir else None
        self.potentials_prefix = potentials_prefix

        if use_langevin is not None:
            self.use_langevin = use_langevin
        else:
            self.use_langevin = settings.thermostat.type == 'langevin'

        self.components: Dict[str, Dict[str, Any]] = {}
        self.types = TypeRegistry()

        self.potential_usage: Dict[str, int] = defaultdict(int)
        self.potential_indices: Dict[str, int] = defaultdict(int)

        self.self_interaction_commands: List[str] = []
        self.cross_interaction_commands: List[str] = []
        self.lj_overrides: Dict[Tuple[str, str], Tuple[float, float]] = {}

        self.virtual_atom_type: Optional[int] = None

    @property
    def lj_cutoff(self) -> float:
        """Expose the configured LJ cutoff."""
        return self.settings.potential.lj_cutoff

    @staticmethod
    def _normalize_element_symbol(value: str) -> str:
        """Normalize element-like tokens to canonical symbol case."""
        token = value.strip()
        if not token:
            return token
        return token[0].upper() + token[1:].lower()

    @staticmethod
    def _normalize_potential_type(value: str) -> str:
        """Normalize potential style names to lowercase keys."""
        return value.strip().lower()

    @classmethod
    def _pair_key(cls, el1: str, el2: str) -> Tuple[str, str]:
        """Build an order-independent key for an element pair."""
        a = cls._normalize_element_symbol(el1)
        b = cls._normalize_element_symbol(el2)
        sorted_pair = sorted((a, b))
        return (sorted_pair[0], sorted_pair[1])

    @classmethod
    def _parse_override_pair(cls, pair_key: str) -> Tuple[str, str]:
        """Parse a pair key like 'Mo-S', 'Mo_S', 'Mo S', or 'Mo,S'."""
        tokens = [tok for tok in re.split(r'[^A-Za-z]+', pair_key) if tok]
        if len(tokens) != 2:
            raise ValueError(
                f"Invalid LJ override pair '{pair_key}'. "
                "Expected two element symbols (e.g., 'Mo-S')."
            )
        return cls._normalize_element_symbol(tokens[0]), cls._normalize_element_symbol(tokens[1])

    @staticmethod
    def _parse_override_values(pair_key: str, values: Any) -> Tuple[float, float]:
        """Parse epsilon/sigma override values from list/tuple or dict."""
        if isinstance(values, (list, tuple)) and len(values) >= 2:
            epsilon = float(values[0])
            sigma = float(values[1])
            return epsilon, sigma

        if isinstance(values, dict):
            if 'epsilon' not in values or 'sigma' not in values:
                raise ValueError(
                    f"Invalid LJ override values for '{pair_key}'. "
                    "Dict entries must include 'epsilon' and 'sigma'."
                )
            return float(values['epsilon']), float(values['sigma'])

        raise ValueError(
            f"Invalid LJ override values for '{pair_key}'. "
            "Use [epsilon, sigma] or {epsilon: ..., sigma: ...}."
        )

    def set_lj_overrides(self, overrides: Optional[Dict[str, Any]]) -> None:
        """Set user-provided LJ overrides keyed by element pair.

        Example:
            {
            "Mo-Mo": [1.0624, 3.878597],
            "Mo-S": [0.4124, 3.75114],
            "S-S": [0.198443, 3.62368]
            }
        """
        self.lj_overrides = {}
        if not overrides:
            return

        for pair_key, values in overrides.items():
            el1, el2 = self._parse_override_pair(str(pair_key))
            epsilon, sigma = self._parse_override_values(str(pair_key), values)
            self.lj_overrides[self._pair_key(el1, el2)] = (epsilon, sigma)

    def _get_lj_params(self, el1: str, el2: str) -> Tuple[float, float]:
        """Return LJ parameters, preferring user overrides when present."""
        norm_el1 = self._normalize_element_symbol(el1)
        norm_el2 = self._normalize_element_symbol(el2)
        override = self.lj_overrides.get(self._pair_key(norm_el1, norm_el2))
        if override is not None:
            return override
        return lj_params(norm_el1, norm_el2)

    @staticmethod
    def _format_lj_value(value: float) -> str:
        """Format LJ numeric values with high precision and no forced rounding."""
        return f"{float(value):.15g}"

    def get_single_component_commands(
        self,
        config: ComponentConfig,
        elements: List[str]
    ) -> List[str]:
        """Generate LAMMPS interatomic potential settings for single-component setup.

        This is useful for standalone simulations like amorphisation where
        only one component is used and no cross-interactions are needed.

        Args:
            config: Component configuration with pot_type and pot_path.
            elements: List of element symbols in the material.

        Returns:
            List of LAMMPS commands (pair_style, pair_coeff, mass).
        """
        commands = []
        pot_type = config.pot_type.lower()
        pot_path = self._get_potential_path(config.pot_path)

        commands.append(f"pair_style {pot_type}")

        element_str = ' '.join(elements)
        commands.append(f"pair_coeff * * {pot_path} {element_str}")

        for i, elem in enumerate(elements, 1):
            try:
                mass = atomic_masses[atomic_numbers[elem]]
                commands.append(f"mass {i} {mass:.6f}")
            except (KeyError, IndexError):
                logger.warning("Could not find mass for element '%s', using 1.0", elem)
                commands.append(f"mass {i} 1.0")

        return commands

    def register_component(
        self,
        name: str,
        config: ComponentConfig,
        n_layers: int = 1,
        regions: Optional[List[Optional[str]]] = None,
    ) -> Dict[str, List[int]]:
        """Register a material component and assign unique global atom type IDs.

        For multi-layer 2D materials, each layer gets its own set of atom types.
        For Langevin thermostat (when self.use_langevin is True), each element
        gets 3 types (mobile, fixed, thermo).

        Args:
            name: Unique identifier (e.g., 'tip', 'sheet', 'sub').
            config: The material configuration.
            n_layers: Number of layers (for 2D materials, default 1).
            regions: Optional list of region identifiers to expand per type.

        Returns:
            Dict mapping element names to their global atom type IDs.
        """
        cif_data = cifread(config.cif_path)
        elements = cif_data['elements']

        pot_type = self._normalize_potential_type(config.pot_type)
        pot_counts = count_atomtypes(config.pot_path, elements, pot_type=pot_type)

        self.potential_usage[pot_type] += n_layers if self.is_sheet_lj(pot_type) else 1

        apply_langevin = self.use_langevin and name in ('tip', 'sub')

        if n_layers > 1:
            self._assign_types(name, elements, pot_counts, n_layers, use_layer_ids=True, regions=regions)
        elif apply_langevin:
            self._assign_types(name, elements, pot_counts, regions=[None, 'fix', 'thermo'])
        else:
            self._assign_types(name, elements, pot_counts, regions=regions)

        component_map = self.types.get_element_map(name)

        self.components[name] = {
            'name': name,
            'config': config,
            'map': component_map,
            'elements': elements,
            'n_layers': n_layers,
            'use_langevin': self.use_langevin
        }

        return component_map

    def register_virtual_atom(self) -> int:
        """Register a virtual atom type for SMD spring attachment.
        
        The virtual atom is used in the 'virtual_atom' drive method where
        a massless particle is moved and the real atoms are tethered to it.
        It has minimal LJ interactions with all other types.
        
        Returns:
            The type ID assigned to the virtual atom.
        """
        if self.virtual_atom_type is not None:
            logger.warning("Virtual atom already registered as type %d", self.virtual_atom_type)
            return self.virtual_atom_type

        virtual_type = self.types.register(
            component='virtual',
            element='Virtual',
            pot_label='Virtual',
            layer=None,
            region=None
        )
        self.virtual_atom_type = virtual_type
        logger.info("Registered virtual atom as type %d", virtual_type)
        return virtual_type

    def _assign_types(
        self,
        name: str,
        elements: List[str],
        pot_counts: Dict[str, int],
        n_layers: int = 1,
        use_layer_ids: bool = False,
        regions: Optional[List[Optional[str]]] = None
    ) -> None:
        """Assign atom types for components (with optional layer and region support).

        Uses element-first ordering to match the LAMMPS renumbering loop in
        stack_multilayer_sheet: for each element, assign types for all layers,
        then move to the next element.

        Args:
            name: Component name.
            elements: List of element symbols.
            pot_counts: Dict of atom counts per element.
            n_layers: Number of layers to create types for (default 1).
            use_layer_ids: If True, assign layer IDs to types; if False, layer=None.
            regions: List of region identifiers for Langevin thermostat (e.g., [None, 'fix', 'thermo']).
                If None, defaults to [None] (standard single type per element).
        """
        if regions is None:
            regions = [None]

        for el in elements:
            count = pot_counts.get(el, 1)
            for layer_idx in range(n_layers):
                for t in range(count):
                    pot_label = el if count == 1 else f"{el}{t+1}"
                    layer = layer_idx if use_layer_ids else None
                    for region in regions:
                        self.types.register(
                            component=name,
                            element=el,
                            pot_label=pot_label,
                            layer=layer,
                            region=region
                        )

    def _get_potential_path(self, original_path: str) -> str:
        """Return a LAMMPS-friendly potential path and stage file locally if requested.

        If `potentials_dir` is configured, copies the potential file there (once)
        and returns either `<potentials_prefix>/<filename>` if provided, or the
        absolute staged path. Otherwise, returns the original path unchanged.
        """
        if not self.potentials_dir:
            return original_path

        src_path = Path(original_path)
        dest_dir = self.potentials_dir
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_path = dest_dir / src_path.name

        if not dest_path.exists():
            shutil.copy2(src_path, dest_path)

        if self.potentials_prefix:
            return f"{self.potentials_prefix}/{src_path.name}"

        return str(dest_path)

    def is_sheet_lj(self, pot_type: str) -> bool:
        """Determine if potential requires explicit LJ for interlayer/cross interactions.

        Args:
            pot_type: The potential type string (e.g., 'sw', 'tersoff', 'airebo').

        Returns:
            True if explicit LJ is needed, False if potential handles it internally.
        """
        pot_lower = pot_type.lower()
        if pot_lower in POTENTIALS_WITH_INTERNAL_LJ:
            return False
        if pot_lower in POTENTIALS_REQUIRING_LJ:
            return True

        logger.warning("Unknown potential type '%s', assuming LJ is required.", pot_type)
        return True

    def add_self_interaction(self, component_name: str,
                            layer: Optional[int] = None):
        """Add self-interaction (many-body potential) for a component.
        
        For multi-layer materials, can specify a specific layer or all layers.

        Args:
            component_name: Name of the registered component.
            layer: Specific layer index (0-based) or None for all/single layer.
        """
        comp = self.components.get(component_name)
        if not comp:
            raise ValueError(f"Component '{component_name}' not registered.")
        c_conf = comp['config']
        n_layers = comp['n_layers']

        pot_type = self._normalize_potential_type(c_conf.pot_type)
        self.potential_indices[pot_type] += 1
        pot_index = self.potential_indices[pot_type]

        needs_index = self.potential_usage[pot_type] > 1
        index_str = f" {pot_index}" if needs_index else ""

        if n_layers > 1 and self.is_sheet_lj(pot_type):
            layers_to_process = [layer] if layer is not None else range(n_layers)
            for l in layers_to_process:
                self._add_layer_self_interaction(comp, l, pot_index)
        else:
            atom_list = self.types.build_null_map(component_name)
            pot_path = self._get_potential_path(c_conf.pot_path)
            cmd = f"pair_coeff * * {pot_type}{index_str} {pot_path} {' '.join(atom_list)}"
            self.self_interaction_commands.append(f"{cmd} # {component_name}")

    def _add_layer_self_interaction(self, comp: Dict, layer: int,
                                    base_pot_index: int):
        """Add self-interaction for a specific layer.

        Args:
            comp: Component dictionary.
            layer: Layer index (0-based).
            base_pot_index: Base potential index for hybrid style.
        """
        c_conf = comp['config']
        name = comp['name']
        pot_type = self._normalize_potential_type(c_conf.pot_type)

        pot_index = base_pot_index + layer
        self.potential_indices[pot_type] = pot_index

        needs_index = self.potential_usage[pot_type] > 1
        index_str = f" {pot_index}" if needs_index else ""

        atom_list = self.types.build_null_map(name, layer=layer)

        pot_path = self._get_potential_path(c_conf.pot_path)
        cmd = f"pair_coeff * * {pot_type}{index_str} {pot_path} {' '.join(atom_list)}"
        self.self_interaction_commands.append(f"{cmd} # {name} Layer {layer+1}")

    def add_cross_interaction(
        self,
        comp1_name: str,
        comp2_name: str,
        interaction_type: str = "lj/cut",
        custom_params: Optional[Dict[str, float]] = None
    ):
        """Add LJ cross-interaction between two components.

        Args:
            comp1_name: Name of first component.
            comp2_name: Name of second component.
            interaction_type: Interaction style (default 'lj/cut').
            custom_params: Optional dict with 'epsilon', 'sigma', 'cutoff'
                overrides.
        """
        elements1 = self.types.elements_in_component(comp1_name)
        elements2 = self.types.elements_in_component(comp2_name)

        if not elements1 or not elements2:
            raise ValueError(f"Components '{comp1_name}' and/or '{comp2_name}' not found.")

        for el1 in elements1:
            for el2 in elements2:
                if custom_params:
                    epsilon = custom_params.get('epsilon', 0.0)
                    sigma = custom_params.get('sigma', 3.0)
                else:
                    epsilon, sigma = self._get_lj_params(el1, el2)

                types1 = self.types.get_element_type_range(comp1_name, el1)
                types2 = self.types.get_element_type_range(comp2_name, el2)

                if int(types1.split('*', maxsplit=1)[0]) > int(types2.split('*', maxsplit=1)[0]):
                    types1, types2 = types2, types1

                eps_str = self._format_lj_value(epsilon)
                sig_str = self._format_lj_value(sigma)
                cmd = f"pair_coeff {types1} {types2} {interaction_type} {eps_str} {sig_str}"
                self.cross_interaction_commands.append(f"{cmd} # {el1}({comp1_name})-{el2}({comp2_name})")

    def add_interlayer_interaction(
        self,
        component_name: str,
        layer_pairs: Optional[List[Tuple[int, int]]] = None,
        low_interaction: bool = False
    ):
        """Add LJ interactions between layers of a multi-layer 2D material.

        Args:
            component_name: Name of the sheet component.
            layer_pairs: List of (layer_i, layer_j) tuples, or None for all
                pairs.
            low_interaction: If True, use near-zero epsilon (ghost/weak
                interaction).
        """
        comp = self.components.get(component_name)
        if not comp:
            raise ValueError(f"Component '{component_name}' not registered.")
        n_layers = comp['n_layers']
        elements = comp['elements']

        if n_layers < 2:
            logger.warning("Component '%s' has only 1 layer, skipping interlayer interaction.", component_name)
            return

        if layer_pairs is None:
            layer_pairs = [(i, j) for i in range(n_layers) for j in range(i+1, n_layers)]

        for layer_i, layer_j in layer_pairs:
            for el1 in elements:
                for el2 in elements:
                    epsilon, sigma = self._get_lj_params(el1, el2)
                    if low_interaction:
                        epsilon = 1e-100  # Near-zero for ghost interaction

                    types_i = self.types.ids_by_component_layer_element(component_name, layer_i, el1)
                    types_j = self.types.ids_by_component_layer_element(component_name, layer_j, el2)

                    if not types_i or not types_j:
                        continue

                    t1_str = self.types.format_type_range(types_i)
                    t2_str = self.types.format_type_range(types_j)

                    if types_i[0] > types_j[0]:
                        t1_str, t2_str = t2_str, t1_str

                    eps_str = self._format_lj_value(epsilon)
                    sig_str = self._format_lj_value(sigma)
                    cmd = f"pair_coeff {t1_str} {t2_str} lj/cut {eps_str} {sig_str}"
                    self.cross_interaction_commands.append(
                        f"{cmd} # {el1}(L{layer_i+1})-{el2}(L{layer_j+1})"
                    )

    def add_ghost_lj(
        self,
        component_name: str,
        max_real_distance: int = 1
    ):
        """Add all interlayer LJ interactions with distance-based ghost handling.

        Layer pairs within max_real_distance get real LJ interactions.
        Layer pairs beyond max_real_distance get ghost (near-zero epsilon)
        interactions to lower computational costs with minimal effect on dynamics.

        Args:
            component_name: Name of the multi-layer component.
            max_real_distance: Maximum layer separation for real LJ interactions.
                - 1 (default): Only adjacent pairs (L1-L2, L2-L3, L3-L4) are
                    real.
                - 2: Adjacent + next-nearest (L1-L3, L2-L4) are real.
                - n_layers or higher: All pairs are real (no ghost
                    interactions).

        Examples:
            For a 4-layer sheet with max_real_distance=1:
                Real LJ: L1-L2, L2-L3, L3-L4
                Ghost:   L1-L3, L1-L4, L2-L4

            For a 4-layer sheet with max_real_distance=2:
                Real LJ: L1-L2, L2-L3, L3-L4, L1-L3, L2-L4
                Ghost:   L1-L4
        """
        comp = self.components.get(component_name)
        if not comp:
            raise ValueError(f"Component '{component_name}' not registered.")
        n_layers = comp['n_layers']

        if n_layers < 2:
            logger.warning("Component '%s' has only 1 layer, skipping interlayer interactions.", component_name)
            return

        real_pairs = []
        ghost_pairs = []

        for i in range(n_layers):
            for j in range(i + 1, n_layers):
                distance = j - i
                if distance <= max_real_distance:
                    real_pairs.append((i, j))
                else:
                    ghost_pairs.append((i, j))

        if real_pairs:
            self.add_interlayer_interaction(component_name, layer_pairs=real_pairs, low_interaction=False)

        if ghost_pairs:
            self.add_interlayer_interaction(component_name, layer_pairs=ghost_pairs, low_interaction=True)

    def calculate_gap(self, comp1_name: str, comp2_name: str,
                        buffer: float = 0.5) -> float:
        """Calculate the recommended gap between two components.

        Based on the maximum sigma from UFF mixing rules plus a buffer.

        Args:
            comp1_name: Name of first component.
            comp2_name: Name of second component.
            buffer: Additional distance to add (default 0.5 Angstrom).

        Returns:
            Recommended gap distance in Angstroms.
        """
        comp1 = self.components.get(comp1_name)
        if not comp1:
            raise ValueError(f"Component '{comp1_name}' not registered.")
        comp2 = self.components.get(comp2_name)
        if not comp2:
            raise ValueError(f"Component '{comp2_name}' not registered.")

        max_sigma = 0.0
        for e1 in set(comp1['elements']):
            for e2 in set(comp2['elements']):
                _, sigma = self._get_lj_params(e1, e2)
                max_sigma = max(max_sigma, sigma)

        return max_sigma + buffer

    def get_masses_string(self) -> str:
        """Generate LAMMPS mass commands for all registered types.

        Returns:
            String with mass commands, one per line.
        """
        element_types: Dict[str, List[int]] = defaultdict(list)
        for atype in self.types:
            element_types[atype.element].append(atype.type_id)

        lines = []
        for element, types in element_types.items():
            types = sorted(types)
            if element == 'Virtual':
                type_range = self.types.format_type_range(types)
                lines.append(f"mass {type_range} 1.0 #Virtual atom")
            else:
                try:
                    mass = atomic_masses[atomic_numbers[element]]
                    type_range = self.types.format_type_range(types)
                    lines.append(f"mass {type_range} {mass:.6f} #{element}")
                except (KeyError, IndexError):
                    logger.warning("Could not find mass for element '%s', using 1.0", element)
                    type_range = self.types.format_type_range(types)
                    lines.append(f"mass {type_range} 1.0 #{element} (unknown)")

        return "\n".join(lines)

    def get_layer_groups_string(self) -> str:
        """Generate LAMMPS group commands for multi-layer components only.

        Only outputs layer groups for components with n_layers > 1.
        Single-layer components define their groups in the main LAMMPS script.

        Returns:
            String with group commands like:
                group layer_1 type 1 2 9 10
                group layer_2 type 3 4 11 12
        """
        lines = []
        for comp in self.components.values():
            name = comp['name']
            n_layers = comp['n_layers']

            if n_layers > 1:
                for layer in range(n_layers):
                    types = self.types.get_layer_group_string(name, layer)
                    if types:
                        lines.append(f"group layer_{layer+1} type {types}")

        return "\n".join(lines)

    def get_component_groups_string(self) -> str:
        """Generate LAMMPS group commands for all components.
        
        For Langevin components (tip, sub), generates:
            - {component}_all: All types for component
            - {component}_fix: Fix region types
            - {component}_thermo: Thermo region types
        
        For sheet components, generates:
            - 2D_all: All sheet types
            - layer_N: Types for each layer (if multi-layer)
        
        For mobile group, generates union of thermo regions if Langevin is used.
        
        Returns:
            String with group commands, one per line.
        """
        lines = []
        has_langevin = False
        langevin_thermo_groups = []
        for comp in self.components.values():
            name = comp['name']
            n_layers = comp['n_layers']
            use_langevin = comp.get('use_langevin', False)

            all_types = self.types.ids_by_component(name)

            if name == 'sheet':
                if all_types:
                    types_str = ' '.join(map(str, all_types))
                    lines.append(f"group 2D_all type {types_str}")

                core_types = [atype.type_id for atype in self.types if atype.component == name and atype.region is None]
                edge_types = [atype.type_id for atype in self.types if atype.component == name and atype.region == 'edge']

                if core_types:
                    types_str = ' '.join(map(str, core_types))
                    lines.append(f"group 2D_core type {types_str}")

                if edge_types:
                    types_str = ' '.join(map(str, edge_types))
                    lines.append(f"group 2D_edge type {types_str}")

                if n_layers > 1:
                    for layer in range(n_layers):
                        layer_types = self.types.get_layer_group_string(name, layer)
                        if layer_types:
                            lines.append(f"group layer_{layer+1} type {layer_types}")

            elif use_langevin and name in ('tip', 'sub'):
                # Langevin components: split into all, fix, thermo
                has_langevin = True

                normal_types = []
                fix_types = []
                thermo_types = []

                for atype in self.types:
                    if atype.component == name:
                        if atype.region is None:
                            normal_types.append(atype.type_id)
                        elif atype.region == 'fix':
                            fix_types.append(atype.type_id)
                        elif atype.region == 'thermo':
                            thermo_types.append(atype.type_id)

                if all_types:
                    types_str = ' '.join(map(str, sorted(all_types)))
                    lines.append(f"group {name}_all type {types_str}")

                if fix_types:
                    types_str = ' '.join(map(str, sorted(fix_types)))
                    lines.append(f"group {name}_fix type {types_str}")

                if thermo_types:
                    types_str = ' '.join(map(str, sorted(thermo_types)))
                    lines.append(f"group {name}_thermo type {types_str}")
                    langevin_thermo_groups.append(f"{name}_thermo")

            else:
                if all_types:
                    types_str = ' '.join(map(str, all_types))
                    lines.append(f"group {name}_all type {types_str}")

        if has_langevin and langevin_thermo_groups:
            lines.append(f"group mobile union {' '.join(langevin_thermo_groups)}")

        return "\n".join(lines)

    def _is_single_potential(self) -> bool:
        """Check if only one potential style is in use (no hybrid needed)."""
        total = sum(self.potential_usage.values())
        return total <= 1 and not self.cross_interaction_commands

    @staticmethod
    def _strip_hybrid_prefix(cmd: str) -> str:
        """Strip the pot_type and index from a hybrid-style pair_coeff command.

        Converts 'pair_coeff * * sw 1 /path Mo S # comment'
        to 'pair_coeff * * /path Mo S # comment' for non-hybrid mode.
        """
        parts = cmd.split('#', 1)
        main = parts[0].strip()
        comment = f" # {parts[1].strip()}" if len(parts) > 1 else ""

        tokens = main.split()
        # pair_coeff * * pot_type [index] path elements...
        # Find the path token (starts with / or contains .)
        for i in range(3, len(tokens)):
            if '/' in tokens[i] or '.' in tokens[i]:
                cleaned = ' '.join(tokens[:3] + tokens[i:])
                return f"{cleaned}{comment}"

        return cmd

    def write_file(self, output_path: Path):
        """Write the complete potential settings file.

        Args:
            output_path: Path for the output file (e.g., system.in.settings).
        """
        lj_cutoff = self.settings.potential.lj_cutoff
        has_lj = bool(self.cross_interaction_commands) or self.virtual_atom_type is not None
        use_hybrid = (not self._is_single_potential()) or self.virtual_atom_type is not None

        # Build pair_style parts; ReaxFF needs NULL + keywords when in hybrid
        style_parts = []
        for pot_type, count in self.potential_usage.items():
            for _ in range(count):
                if use_hybrid and pot_type.lower() in REAXFF_TYPES:
                    safezone = self.settings.potential.reaxff_safezone
                    mincap = self.settings.potential.reaxff_mincap
                    style_parts.append(
                        f"{pot_type} NULL safezone {safezone} mincap {mincap}"
                    )
                else:
                    style_parts.append(pot_type)

        if has_lj:
            style_parts.append(f"lj/cut {lj_cutoff}")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(self.get_masses_string())
            f.write("\n")
            f.write(self.get_component_groups_string())
            f.write("\n")

            if use_hybrid:
                f.write(f"pair_style hybrid {' '.join(style_parts)}\n")
            elif style_parts:
                pot = style_parts[0]
                if pot.lower() in REAXFF_TYPES:
                    safezone = self.settings.potential.reaxff_safezone
                    mincap = self.settings.potential.reaxff_mincap
                    f.write(
                        f"pair_style {pot} NULL"
                        f" safezone {safezone}"
                        f" mincap {mincap}\n"
                    )
                else:
                    f.write(f"pair_style {pot}\n")

            if self.self_interaction_commands:
                f.write("# Self-interactions (intra-component)\n")
                for cmd in self.self_interaction_commands:
                    if use_hybrid:
                        f.write(f"{cmd}\n")
                    else:
                        f.write(f"{self._strip_hybrid_prefix(cmd)}\n")
                f.write("\n")

            if self.cross_interaction_commands:
                f.write("# Cross-interactions (inter-component LJ)\n")
                for cmd in self.cross_interaction_commands:
                    f.write(f"{cmd}\n")
                f.write("\n")

            if self.virtual_atom_type is not None:
                f.write("# Virtual atom (minimal LJ)\n")
                f.write(f"pair_coeff * {self.virtual_atom_type} lj/cut 1e-100 1e-100\n")

        logger.info("Wrote potential settings to %s", output_path)
