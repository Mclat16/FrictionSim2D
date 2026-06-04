import pytest
from pathlib import Path
from src.core.simulation_base import SimulationBase
from src.core.config import AFMSimulationConfig, load_settings

# Mock class to allow instantiation of abstract SimulationBase
class ConcreteSimulation(SimulationBase):
    def build(self): pass
    def write_inputs(self): pass


@pytest.fixture
def afm_config(tmp_path):
    """Minimal valid AFMSimulationConfig for SimulationBase tests."""
    pot_path = tmp_path / "dummy.sw"
    cif_path = tmp_path / "dummy.cif"
    pot_path.write_text("# dummy potential\n", encoding="utf-8")
    cif_path.write_text("# dummy cif\n", encoding="utf-8")
    settings = load_settings().model_dump()
    return AFMSimulationConfig(**{
        'general': {'temp': 300.0, 'force': 10.0, 'scan_speed': 2.0},
        'tip': {'mat': 'Si', 'pot_type': 'sw', 'pot_path': str(pot_path),
                'cif_path': str(cif_path), 'r': 10.0, 'dspring': 0.0},
        'sub': {'mat': 'Si', 'pot_type': 'sw', 'pot_path': str(pot_path),
            'cif_path': str(cif_path), 'thickness': 10.0, 'amorph': 'a'},
        '2D': {'mat': 'h-MoS2', 'pot_type': 'sw', 'pot_path': str(pot_path),
               'cif_path': str(cif_path), 'x': 50.0, 'y': 50.0, 'layers': [1]},
        'lj_override': {},
        'settings': settings,
    })


def test_directory_creation(tmp_path, afm_config):
    # afm_config comes from the local fixture above
    builder = ConcreteSimulation(afm_config, output_dir=tmp_path)
    
    builder._create_directories()
    
    assert (tmp_path / 'visuals').exists()
    assert (tmp_path / 'results').exists()

def test_template_rendering(tmp_path, afm_config):
    builder = ConcreteSimulation(afm_config, output_dir=tmp_path)
    
    # Render a real template or a mock one if you inject a mock env
    # Testing a real template:
    context = {'atom_style': 'molecular'}
    result = builder.render_template('common/init.lmp', context)
    
    assert "atom_style      molecular" in result