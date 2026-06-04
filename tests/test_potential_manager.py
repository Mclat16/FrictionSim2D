"""Tests for PotentialManager.

Tests cover the 7 main use cases via the core methods:
1. Single-component: Tip standalone (for amorphisation)
2. Single-component: Substrate standalone (for amorphisation)
3. Single-component: Sheet standalone (single/multi-layer)
4. Langevin-ready: Tip with 3 region types
5. Langevin-ready: Substrate with 3 region types
6. Full system: AFM (tip + sheet + substrate)
7. Full system: Sheet-on-sheet (3-4 layers with ghost interactions)

The builders (AFMBuilder, SheetOnSheetBuilder) call these core methods directly.
Tests verify the core methods work correctly for all use cases.

"""

import pytest
import tempfile
from pathlib import Path

from src.core.potential_manager import (
    PotentialManager,
    POTENTIALS_WITH_INTERNAL_LJ,
    POTENTIALS_REQUIRING_LJ,
)
from src.core.config import ComponentConfig


class TestPotentialManagerInit:
    """Tests for PotentialManager initialization."""

    def test_init_with_langevin_settings(self, mock_settings):
        """PM should detect Langevin from settings."""
        mock_settings.thermostat.type = 'langevin'
        pm = PotentialManager(mock_settings)
        assert pm.use_langevin is True

    def test_init_with_nose_hoover_settings(self, mock_settings):
        """PM should not use Langevin with Nose-Hoover thermostat."""
        mock_settings.thermostat.type = 'nose-hoover'
        pm = PotentialManager(mock_settings)
        assert pm.use_langevin is False

    def test_init_with_explicit_langevin_override(self, mock_settings):
        """PM should respect explicit use_langevin override."""
        mock_settings.thermostat.type = 'langevin'
        pm = PotentialManager(mock_settings, use_langevin=False)
        assert pm.use_langevin is False

    def test_lj_cutoff_property(self, mock_settings):
        """PM should expose lj_cutoff from settings via property."""
        pm = PotentialManager(mock_settings)
        assert pm.lj_cutoff == mock_settings.potential.lj_cutoff
        assert pm.lj_cutoff == 11.0


class TestIsSheetLJ:
    """Tests for is_sheet_lj potential classification."""

    def test_potentials_with_internal_lj(self, mock_settings):
        """Potentials with internal LJ should return False."""
        pm = PotentialManager(mock_settings)
        for pot in POTENTIALS_WITH_INTERNAL_LJ:
            assert pm.is_sheet_lj(pot) is False
            assert pm.is_sheet_lj(pot.upper()) is False  # Case insensitive

    def test_potentials_requiring_lj(self, mock_settings):
        """Potentials requiring explicit LJ should return True."""
        pm = PotentialManager(mock_settings)
        for pot in POTENTIALS_REQUIRING_LJ:
            assert pm.is_sheet_lj(pot) is True
            assert pm.is_sheet_lj(pot.upper()) is True

    def test_unknown_potential_defaults_to_true(self, mock_settings):
        """Unknown potentials should default to requiring LJ (safe default)."""
        pm = PotentialManager(mock_settings)
        assert pm.is_sheet_lj('unknown_potential') is True


class TestNormalizationGuards:
    """Tests for defensive normalization inside PotentialManager."""

    def test_register_component_normalizes_potential_usage_key(
        self,
        mock_settings,
        monkeypatch,
    ):
        """Mixed-case pot_type should map to a single lowercase usage key."""
        # Monkeypatch path validation functions in the modules that use them
        monkeypatch.setattr(
            'src.core.config.get_potential_path',
            lambda _path: Path(_path)
        )
        monkeypatch.setattr(
            'src.core.config.get_material_path',
            lambda _path: Path(_path)
        )
        monkeypatch.setattr(
            'src.core.potential_manager.cifread',
            lambda _path: {'elements': ['Si']}
        )
        monkeypatch.setattr(
            'src.core.potential_manager.count_atomtypes',
            lambda _path, _elements, pot_type=None: {'Si': 1}
        )

        from src.core.config import TipConfig
        config_upper = TipConfig(
            mat='Si', pot_type='SW', pot_path='/tmp/Si.sw', cif_path='/tmp/Si.cif',
            r=20.0, amorph='c', dspring=0.1, s=1.0
        )
        pm = PotentialManager(mock_settings, use_langevin=False)
        pm.register_component('tip', config_upper)

        assert 'sw' in pm.potential_usage
        assert 'SW' not in pm.potential_usage

    def test_add_self_interaction_uses_normalized_key_for_indexing(
        self,
        mock_settings,
        monkeypatch,
    ):
        """Potential indexing should stay consistent when pot_type casing differs."""
        # Monkeypatch path validation functions in the modules that use them
        monkeypatch.setattr(
            'src.core.config.get_potential_path',
            lambda _path: Path(_path)
        )
        monkeypatch.setattr(
            'src.core.config.get_material_path',
            lambda _path: Path(_path)
        )
        monkeypatch.setattr(
            'src.core.potential_manager.cifread',
            lambda _path: {'elements': ['Si']}
        )
        monkeypatch.setattr(
            'src.core.potential_manager.count_atomtypes',
            lambda _path, _elements, pot_type=None: {'Si': 1}
        )

        # Create minimal configs without needing real files
        from src.core.config import TipConfig, SubstrateConfig
        config_upper = TipConfig(
            mat='Si', pot_type='SW', pot_path='/tmp/Si.sw', cif_path='/tmp/Si.cif',
            r=20.0, amorph='c', dspring=0.1, s=1.0
        )
        config_lower = SubstrateConfig(
            mat='Si', pot_type='sw', pot_path='/tmp/Si.sw', cif_path='/tmp/Si.cif',
            thickness=10.0, amorph='c'
        )

        pm = PotentialManager(mock_settings, use_langevin=False)
        pm.register_component('tip', config_upper)
        pm.register_component('sub', config_lower)

        pm.add_self_interaction('tip')
        pm.add_self_interaction('sub')

        assert len(pm.self_interaction_commands) == 2
        assert ' sw ' in pm.self_interaction_commands[0].lower()
        assert ' sw ' in pm.self_interaction_commands[1].lower()
        assert 'sw 1' in pm.self_interaction_commands[0].lower()
        assert 'sw 2' in pm.self_interaction_commands[1].lower()

    def test_mixed_case_elements_use_override_in_cross_interaction(self, mock_settings):
        """Mixed-case element symbols should still resolve override parameters."""
        pm = PotentialManager(mock_settings, use_langevin=False)
        pm.set_lj_overrides({'mo-s': [0.5, 3.5]})

        pm.types.register(component='a', element='MO', pot_label='MO')
        pm.types.register(component='b', element='s', pot_label='s')
        pm.add_cross_interaction('a', 'b')

        command = pm.cross_interaction_commands[0]
        assert 'lj/cut 0.5 3.5' in command


class TestLJOverrides:
    """Tests for user-provided LJ override behavior."""

    def test_set_lj_overrides_parses_pair_keys(self, mock_settings):
        """Pair keys should accept multiple separators and normalize order."""
        pm = PotentialManager(mock_settings, use_langevin=False)
        pm.set_lj_overrides({
            'Mo-S': [0.4124, 3.75114],
            'S_Mo': [0.4999, 3.9],
        })

        # Last assignment for the same canonical pair wins
        assert pm.lj_overrides[('Mo', 'S')] == (0.4999, 3.9)

    def test_interlayer_uses_override_values(
        self, mock_settings, mos2_sheet_config
    ):
        """Interlayer pair_coeff should use override epsilon/sigma for matching pairs."""
        import src.core.potential_manager as pm_module

        def mock_cifread(path):
            return {'elements': ['Mo', 'S']}

        def mock_count(path, elements, pot_type=None):
            return {el: 1 for el in elements}

        original_cifread = pm_module.cifread
        original_count = pm_module.count_atomtypes
        pm_module.cifread = mock_cifread
        pm_module.count_atomtypes = mock_count

        try:
            pm = PotentialManager(mock_settings, use_langevin=False)
            pm.set_lj_overrides({
                'Mo-Mo': [1.0624, 3.878597],
                'Mo-S': [0.4124, 3.75114],
                'S-S': [0.198443, 3.62368],
            })
            pm.register_component('sheet', mos2_sheet_config, n_layers=2)
            pm.add_interlayer_interaction('sheet')

            joined = "\n".join(pm.cross_interaction_commands)
            assert 'lj/cut 1.0624 3.878597' in joined
            assert 'lj/cut 0.4124 3.75114' in joined
            assert 'lj/cut 0.198443 3.62368' in joined
        finally:
            pm_module.cifread = original_cifread
            pm_module.count_atomtypes = original_count


# =========================================================================
# Use Case 1: Single-component Tip (amorphisation)
# =========================================================================

class TestSingleComponentTip:
    """Tests for standalone tip simulation setup."""

    def test_tip_standard_types(
        self, mock_settings, si_tip_config, mock_cifread_si, mock_count_atomtypes_si
    ):
        """Single Si tip should get 1 atom type."""
        mock_settings.thermostat.type = 'nose-hoover'  # No Langevin
        pm = PotentialManager(mock_settings, use_langevin=False)
        
        # Use core method
        result = pm.register_component('tip', si_tip_config)
        pm.add_self_interaction('tip')
        
        assert 'Si' in result
        assert len(result['Si']) == 1
        assert result['Si'] == [1]
        assert len(pm.types) == 1

    def test_tip_self_interaction_only(
        self, mock_settings, si_tip_config, mock_cifread_si, mock_count_atomtypes_si
    ):
        """Single tip should have self-interaction but no cross-interactions."""
        pm = PotentialManager(mock_settings, use_langevin=False)
        pm.register_component('tip', si_tip_config)
        pm.add_self_interaction('tip')
        
        assert len(pm.self_interaction_commands) == 1
        assert len(pm.cross_interaction_commands) == 0
        assert 'tip' in pm.self_interaction_commands[0]

    def test_tip_group_string(
        self, mock_settings, si_tip_config, mock_cifread_si, mock_count_atomtypes_si
    ):
        """Group string should contain all tip types."""
        pm = PotentialManager(mock_settings, use_langevin=False)
        pm.register_component('tip', si_tip_config)
        
        group_str = pm.types.get_group_string('tip')
        assert group_str == '1'


# =========================================================================
# Use Case 2: Single-component Substrate (amorphisation)
# =========================================================================

class TestSingleComponentSubstrate:
    """Tests for standalone substrate simulation setup."""

    def test_sub_standard_types(
        self, mock_settings, si_sub_config, mock_cifread_si, mock_count_atomtypes_si
    ):
        """Single Si substrate should get 1 atom type."""
        pm = PotentialManager(mock_settings, use_langevin=False)
        
        result = pm.register_component('sub', si_sub_config)
        pm.add_self_interaction('sub')
        
        assert 'Si' in result
        assert len(result['Si']) == 1
        assert len(pm.types) == 1

    def test_sub_no_cross_interactions(
        self, mock_settings, si_sub_config, mock_cifread_si, mock_count_atomtypes_si
    ):
        """Standalone substrate should have no cross-interactions."""
        pm = PotentialManager(mock_settings, use_langevin=False)
        pm.register_component('sub', si_sub_config)
        pm.add_self_interaction('sub')
        
        assert len(pm.cross_interaction_commands) == 0


# =========================================================================
# Use Case 3: Single-component Sheet (single and multi-layer)
# =========================================================================

class TestSingleComponentSheet:
    """Tests for standalone sheet simulation setup."""

    def test_sheet_single_layer(
        self, mock_settings, mos2_sheet_config, mock_cifread_mos2, mock_count_atomtypes_mos2_simple
    ):
        """Single-layer sheet should get types for each element."""
        pm = PotentialManager(mock_settings, use_langevin=False)
        
        result = pm.register_component('sheet', mos2_sheet_config, n_layers=1)
        pm.add_self_interaction('sheet')
        
        assert 'Mo' in result
        assert 'S' in result
        assert len(pm.types) == 2

    def test_sheet_multi_layer_types(
        self, mock_settings, mos2_sheet_config, mock_cifread_mos2, mock_count_atomtypes_mos2_simple
    ):
        """Multi-layer sheet should get types per layer."""
        pm = PotentialManager(mock_settings, use_langevin=False)
        
        result = pm.register_component('sheet', mos2_sheet_config, n_layers=2)
        
        # 2 elements × 2 layers = 4 types
        assert len(pm.types) == 4

    def test_sheet_multi_layer_interlayer_lj(
        self, mock_settings, mos2_sheet_config, mock_cifread_mos2, mock_count_atomtypes_mos2_simple
    ):
        """Multi-layer sheet should have interlayer LJ interactions."""
        pm = PotentialManager(mock_settings, use_langevin=False)
        pm.register_component('sheet', mos2_sheet_config, n_layers=2)
        pm.add_self_interaction('sheet')
        pm.add_interlayer_interaction('sheet')
        
        # Should have interlayer LJ commands (Mo-Mo, Mo-S, S-Mo, S-S between layers)
        assert len(pm.cross_interaction_commands) > 0


# =========================================================================
# Use Case 4 & 5: Langevin-ready Components (Tip & Substrate)
# =========================================================================

class TestLangevinComponents:
    """Tests for Langevin thermostat region setup."""

    def test_tip_langevin_3x_types(
        self, mock_settings, si_tip_config, mock_cifread_si, mock_count_atomtypes_si
    ):
        """Tip with Langevin should get 3 types per element."""
        pm = PotentialManager(mock_settings, use_langevin=True)  # Enable Langevin
        
        result = pm.register_component('tip', si_tip_config)
        
        # Si has 1 pot type × 3 regions = 3 types
        assert 'Si' in result
        assert len(result['Si']) == 3
        assert len(pm.types) == 3

    def test_sub_langevin_3x_types(
        self, mock_settings, si_sub_config, mock_cifread_si, mock_count_atomtypes_si
    ):
        """Substrate with Langevin should get 3 types per element."""
        pm = PotentialManager(mock_settings, use_langevin=True)
        
        result = pm.register_component('sub', si_sub_config)
        
        assert len(result['Si']) == 3
        assert len(pm.types) == 3

    def test_sheet_no_langevin_expansion(
        self, mock_settings, mos2_sheet_config, mock_cifread_mos2, mock_count_atomtypes_mos2_simple
    ):
        """Sheets should never get Langevin expansion regardless of PM setting."""
        pm = PotentialManager(mock_settings, use_langevin=True)
        
        result = pm.register_component('sheet', mos2_sheet_config)
        
        # Should only have 2 types (Mo, S), not 6 (Mo×3, S×3)
        assert len(pm.types) == 2

    def test_langevin_group_definitions(
        self, mock_settings, si_tip_config, mock_cifread_si, mock_count_atomtypes_si
    ):
        """Langevin types should have proper group definitions (mobile, fix, thermo)."""
        pm = PotentialManager(mock_settings, use_langevin=True)
        pm.register_component('tip', si_tip_config)
        
        # Check generated group names from registered atom types
        group_names = [atype.group_name for atype in pm.types]
        
        # Should have tip_t1, tip_fix_t1, tip_thermo_t1
        assert any('tip_t' in g and '_fix' not in g and '_thermo' not in g for g in group_names)
        assert any('_fix' in g for g in group_names)
        assert any('_thermo' in g for g in group_names)


# =========================================================================
# Use Case 6: AFM System (Tip + Sheet + Substrate)
# =========================================================================

class TestAFMSystem:
    """Tests for complete AFM system setup."""

    def test_afm_all_components_registered(
        self, mock_settings, si_tip_config, si_sub_config, mos2_sheet_config,
        mock_cifread_si, mock_count_atomtypes_si
    ):
        """AFM system should register all three components."""
        # Need to mock both Si and MoS2 cifread
        import src.core.potential_manager as pm_module
        
        def mock_cifread(path):
            if 'MoS2' in str(path) or 'h-Mo' in str(path):
                return {'elements': ['Mo', 'S']}
            return {'elements': ['Si']}
        
        def mock_count(path, elements, pot_type=None):
            return {el: 1 for el in elements}
        
        original_cifread = pm_module.cifread
        original_count = pm_module.count_atomtypes
        pm_module.cifread = mock_cifread
        pm_module.count_atomtypes = mock_count
        
        try:
            pm = PotentialManager(mock_settings, use_langevin=False)
            # Register all components
            pm.register_component('sub', si_sub_config)
            pm.register_component('tip', si_tip_config)
            pm.register_component('sheet', mos2_sheet_config, n_layers=1)
            
            # Add self-interactions
            pm.add_self_interaction('sub')
            pm.add_self_interaction('tip')
            pm.add_self_interaction('sheet')
            
            # Si(sub) + Si(tip) + Mo + S = 4 types
            assert len(pm.types) == 4
        finally:
            pm_module.cifread = original_cifread
            pm_module.count_atomtypes = original_count

    def test_afm_cross_interactions(
        self, mock_settings, si_tip_config, si_sub_config, mos2_sheet_config
    ):
        """AFM system should have cross-interactions between all pairs."""
        import src.core.potential_manager as pm_module
        
        def mock_cifread(path):
            if 'MoS2' in str(path) or 'h-Mo' in str(path):
                return {'elements': ['Mo', 'S']}
            return {'elements': ['Si']}
        
        def mock_count(path, elements, pot_type=None):
            return {el: 1 for el in elements}
        
        original_cifread = pm_module.cifread
        original_count = pm_module.count_atomtypes
        pm_module.cifread = mock_cifread
        pm_module.count_atomtypes = mock_count
        
        try:
            pm = PotentialManager(mock_settings, use_langevin=False)
            # Register
            pm.register_component('sub', si_sub_config)
            pm.register_component('tip', si_tip_config)
            pm.register_component('sheet', mos2_sheet_config, n_layers=1)
            
            # Self-interactions
            pm.add_self_interaction('sub')
            pm.add_self_interaction('tip')
            pm.add_self_interaction('sheet')
            
            # Cross-interactions (like AFMBuilder does)
            pm.add_cross_interaction('sub', 'tip')
            pm.add_cross_interaction('sub', 'sheet')
            pm.add_cross_interaction('tip', 'sheet')
            
            # Should have LJ cross-interaction commands
            assert len(pm.cross_interaction_commands) >= 3
            
            # Check all component pairs are covered
            cross_str = ' '.join(pm.cross_interaction_commands)
            assert 'lj' in cross_str.lower()
        finally:
            pm_module.cifread = original_cifread
            pm_module.count_atomtypes = original_count


# =========================================================================
# Use Case 7: Sheet-on-Sheet System
# =========================================================================

class TestSheetOnSheetSystem:
    """Tests for sheet-on-sheet friction simulation setup."""

    def test_sheet_on_sheet_4_layers(
        self, mock_settings, mos2_multilayer_config, mock_cifread_mos2, mock_count_atomtypes_mos2_simple
    ):
        """Sheet-on-sheet should create 4 layers with separate types."""
        pm = PotentialManager(mock_settings, use_langevin=False)
        
        pm.register_component('sheet', mos2_multilayer_config, n_layers=4)
        pm.add_self_interaction('sheet')
        
        # 2 elements × 4 layers = 8 types
        assert len(pm.types) == 8

    def test_sheet_on_sheet_layer_groups(
        self, mock_settings, mos2_multilayer_config, mock_cifread_mos2, mock_count_atomtypes_mos2_simple
    ):
        """Each layer should have its own group of types."""
        pm = PotentialManager(mock_settings, use_langevin=False)
        pm.register_component('sheet', mos2_multilayer_config, n_layers=4)
        pm.add_self_interaction('sheet')
        
        for layer in range(4):
            layer_types = pm.types.get_layer_group_string('sheet', layer)
            assert layer_types != ''
            # Each layer should have 2 types (Mo, S)
            assert len(layer_types.split()) == 2

    def test_sheet_on_sheet_ghost_interactions(
        self, mock_settings, mos2_multilayer_config, mock_cifread_mos2, mock_count_atomtypes_mos2_simple
    ):
        """Non-adjacent layers should get ghost (low epsilon) interactions."""
        pm = PotentialManager(mock_settings, use_langevin=False)
        pm.register_component('sheet', mos2_multilayer_config, n_layers=4)
        pm.add_self_interaction('sheet')
        pm.add_ghost_lj('sheet', max_real_distance=1)
        
        # Check we have cross-interaction commands
        assert len(pm.cross_interaction_commands) > 0
        
        # Some should be ghost (1e-100 epsilon)
        ghost_commands = [c for c in pm.cross_interaction_commands 
                         if '1e-100' in c.lower() or '1.00e-100' in c.lower()]
        real_commands = [c for c in pm.cross_interaction_commands 
                        if '1e-100' not in c.lower() and '1.00e-100' not in c.lower()]
        
        # With 4 layers and max_real_distance=1:
        # Real: L1-L2, L2-L3, L3-L4 (3 pairs)
        # Ghost: L1-L3, L1-L4, L2-L4 (3 pairs)
        # Each pair has element combinations: Mo-Mo, Mo-S, S-Mo, S-S (4 combos)
        assert len(ghost_commands) > 0
        assert len(real_commands) > 0

    def test_sheet_on_sheet_layer_groups_string(
        self, mock_settings, mos2_multilayer_config, mock_cifread_mos2, mock_count_atomtypes_mos2_simple
    ):
        """Layer groups string should define all layer groups."""
        pm = PotentialManager(mock_settings, use_langevin=False)
        pm.register_component('sheet', mos2_multilayer_config, n_layers=4)
        pm.add_self_interaction('sheet')
        
        layer_groups = pm.get_layer_groups_string()
        
        assert 'layer_1' in layer_groups
        assert 'layer_2' in layer_groups
        assert 'layer_3' in layer_groups
        assert 'layer_4' in layer_groups


# =========================================================================
# get_single_component_commands Tests (for amorphisation)
# =========================================================================

class TestSingleComponentCommands:
    """Tests for get_single_component_commands method (used by make_amorphous)."""

    def test_get_commands_for_sw(self, mock_settings, si_tip_config):
        """Should generate valid LAMMPS commands for SW potential."""
        pm = PotentialManager(mock_settings, use_langevin=False)
        commands = pm.get_single_component_commands(si_tip_config, ['Si'])
        
        assert len(commands) >= 3  # pair_style, pair_coeff, mass
        assert any('pair_style' in c for c in commands)
        assert any('pair_coeff' in c for c in commands)
        assert any('mass' in c for c in commands)
        # SW should include itself
        assert any('sw' in c for c in commands)

    def test_get_commands_includes_all_elements(self, mock_settings, mos2_sheet_config):
        """Commands should include all elements."""
        pm = PotentialManager(mock_settings, use_langevin=False)
        commands = pm.get_single_component_commands(mos2_sheet_config, ['Mo', 'S'])
        
        # Should have pair_style, pair_coeff, and 2 mass commands
        mass_commands = [c for c in commands if 'mass' in c]
        assert len(mass_commands) == 2

    def test_get_commands_hybrid_for_lj_requiring(self, mock_settings, si_tip_config):
        """get_single_component_commands uses bare pair_style for single-component setup."""
        pm = PotentialManager(mock_settings, use_langevin=False)
        commands = pm.get_single_component_commands(si_tip_config, ['Si'])
        
        pair_style = [c for c in commands if 'pair_style' in c][0]
        # Single-component amorphisation commands use a bare pair_style (no LJ needed)
        assert 'sw' in pair_style
        assert pair_style.strip() == 'pair_style sw'


# =========================================================================
# File Output Tests
# =========================================================================

class TestFileOutput:
    """Tests for LAMMPS file generation."""

    def test_write_file_creates_directory(
        self, mock_settings, si_tip_config, mock_cifread_si, mock_count_atomtypes_si
    ):
        """write_file should create output directory if needed."""
        pm = PotentialManager(mock_settings, use_langevin=False)
        pm.register_component('tip', si_tip_config)
        pm.add_self_interaction('tip')
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / 'subdir' / 'system.in.settings'
            pm.write_file(output_path)
            
            assert output_path.exists()
            assert output_path.parent.exists()

    def test_write_file_contains_masses(
        self, mock_settings, si_tip_config, mock_cifread_si, mock_count_atomtypes_si
    ):
        """Output file should contain mass commands."""
        pm = PotentialManager(mock_settings, use_langevin=False)
        pm.register_component('tip', si_tip_config)
        pm.add_self_interaction('tip')
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / 'system.in.settings'
            pm.write_file(output_path)
            
            content = output_path.read_text()
            assert 'mass' in content
            assert 'Si' in content

    def test_write_file_pair_style(
        self, mock_settings, si_tip_config, mock_cifread_si, mock_count_atomtypes_si
    ):
        """Output file should contain pair_style command."""
        pm = PotentialManager(mock_settings, use_langevin=False)
        pm.register_component('tip', si_tip_config)
        pm.add_self_interaction('tip')
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / 'system.in.settings'
            pm.write_file(output_path)
            
            content = output_path.read_text()
            assert 'pair_style' in content
            assert 'sw' in content  # Si uses SW potential

    def test_write_file_layer_groups_only_for_multilayer(
        self, mock_settings, mos2_sheet_config, mock_cifread_mos2, mock_count_atomtypes_mos2_simple
    ):
        """Layer groups should only appear for multi-layer components."""
        # Single layer
        pm = PotentialManager(mock_settings, use_langevin=False)
        pm.register_component('sheet', mos2_sheet_config, n_layers=1)
        pm.add_self_interaction('sheet')
        
        layer_groups = pm.get_layer_groups_string()
        assert layer_groups == ''  # No layer groups for single layer

        # Multi-layer
        pm2 = PotentialManager(mock_settings, use_langevin=False)
        pm2.register_component('sheet', mos2_sheet_config, n_layers=4)
        pm2.add_self_interaction('sheet')
        
        layer_groups = pm2.get_layer_groups_string()
        assert 'layer_1' in layer_groups
        assert 'layer_4' in layer_groups

    def test_write_file_includes_sheet_edge_groups(
        self, mock_settings, mos2_sheet_config, mock_cifread_mos2, mock_count_atomtypes_mos2_simple
    ):
        """Sheet edge typing should emit core and edge groups."""
        pm = PotentialManager(mock_settings, use_langevin=False)
        pm.register_component('sheet', mos2_sheet_config, n_layers=1, regions=[None, 'edge'])
        pm.add_self_interaction('sheet')

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / 'system.in.settings'
            pm.write_file(output_path)

            content = output_path.read_text()
            assert 'group 2D_core type' in content
            assert 'group 2D_edge type' in content


# =========================================================================
# Gap Calculation Tests
# =========================================================================

class TestGapCalculation:
    """Tests for gap calculation between components."""

    def test_calculate_gap_same_elements(
        self, mock_settings, si_tip_config, si_sub_config, 
        mock_cifread_si, mock_count_atomtypes_si
    ):
        """Gap between same elements should use their sigma."""
        pm = PotentialManager(mock_settings, use_langevin=False)
        pm.register_component('tip', si_tip_config)
        pm.register_component('sub', si_sub_config)
        
        gap = pm.calculate_gap('tip', 'sub', buffer=0.5)
        
        # Si-Si sigma from UFF + buffer
        assert gap > 0.5  # At least the buffer
        assert gap < 10.0  # Reasonable upper bound

    def test_calculate_gap_different_elements(self, mock_settings, tmp_path):
        """Gap between different elements should use mixing rules."""
        import src.core.potential_manager as pm_module
        from src.core.config import SheetConfig, TipConfig
        
        def mock_cifread(path):
            if 'MoS2' in str(path):
                return {'elements': ['Mo', 'S']}
            return {'elements': ['Si']}
        
        def mock_count(path, elements, pot_type=None):
            return {el: 1 for el in elements}
        
        si_sw = tmp_path / 'Si.sw'
        mos2_sw = tmp_path / 'MoS2.sw'
        si_cif = tmp_path / 'Si.cif'
        mos2_cif = tmp_path / 'MoS2.cif'
        for p in [si_sw, mos2_sw, si_cif, mos2_cif]:
            p.write_text('# dummy', encoding='utf-8')
        
        tip_config = TipConfig(
            mat='Si', pot_type='sw', pot_path=str(si_sw),
            cif_path=str(si_cif), r=20.0, amorph='c', dspring=0.1,
        )
        sheet_config = SheetConfig(
            mat='MoS2', pot_type='sw', pot_path=str(mos2_sw),
            cif_path=str(mos2_cif), x=50.0, y=50.0, layers=[1]
        )
        
        original_cifread = pm_module.cifread
        original_count = pm_module.count_atomtypes
        pm_module.cifread = mock_cifread
        pm_module.count_atomtypes = mock_count
        
        try:
            pm = PotentialManager(mock_settings, use_langevin=False)
            pm.register_component('tip', tip_config)
            pm.register_component('sheet', sheet_config)
            
            gap = pm.calculate_gap('tip', 'sheet', buffer=0.5)
            
            # Should be positive and reasonable
            assert gap > 0.5
            assert gap < 10.0
        finally:
            pm_module.cifread = original_cifread
            pm_module.count_atomtypes = original_count


# =========================================================================
# Edge Cases and Error Handling
# =========================================================================

class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_unregistered_component_raises_error(self, mock_settings):
        """Interlayer setup on unregistered component should raise ValueError."""
        pm = PotentialManager(mock_settings)
        
        with pytest.raises(ValueError, match="not registered"):
            pm.add_ghost_lj('nonexistent')

    def test_add_cross_interaction_missing_component(self, mock_settings):
        """Cross-interaction with missing component should raise error."""
        pm = PotentialManager(mock_settings)
        
        with pytest.raises(ValueError, match="not found"):
            pm.add_cross_interaction('comp1', 'comp2')

    def test_multiple_components_type_ids_sequential(
        self, mock_settings, si_tip_config, si_sub_config,
        mock_cifread_si, mock_count_atomtypes_si
    ):
        """Multiple components should get sequential type IDs."""
        pm = PotentialManager(mock_settings, use_langevin=False)
        
        tip_map = pm.register_component('tip', si_tip_config)
        sub_map = pm.register_component('sub', si_sub_config)
        
        # Tip gets type 1, sub gets type 2
        assert tip_map['Si'] == [1]
        assert sub_map['Si'] == [2]
        assert len(pm.types) == 2

    def test_group_string_unregistered_returns_empty(self, mock_settings):
        """Group string for unregistered component should be empty."""
        pm = PotentialManager(mock_settings)
        assert pm.types.get_group_string('nonexistent') == ''

    def test_add_self_interaction_before_registration(self, mock_settings):
        """Adding self-interaction before registration should raise error."""
        pm = PotentialManager(mock_settings)
        
        with pytest.raises(ValueError, match="not registered"):
            pm.add_self_interaction('nonexistent')
