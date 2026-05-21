import pytest
from unittest.mock import patch
from src.interfaces.atomsk import AtomskWrapper

def test_atomsk_binary_not_found():
    """Ensure error is raised if atomsk is missing."""
    with patch("shutil.which", return_value=None), \
         patch.dict("os.environ", {}, clear=True):
        with pytest.raises(RuntimeError, match="Atomsk binary not found"):
            AtomskWrapper()

def test_atomsk_run_success():
    """Test generic run command construction."""
    with patch("shutil.which", return_value="/usr/bin/atomsk"):
        wrapper = AtomskWrapper()
        
        with patch("subprocess.run") as mock_run:
            wrapper.run(["input.xsf", "output.cfg"])
            
            mock_run.assert_called_once()
            args = mock_run.call_args[0][0]
            assert args[0] == "/usr/bin/atomsk"
            assert "-ow" in args
            assert "-v" in args
            assert "0" in args
            assert "input.xsf" in args
            assert "output.cfg" in args

def test_create_slab():
    """Test create_slab command logic."""
    with patch("shutil.which", return_value="/bin/atomsk"):
        wrapper = AtomskWrapper()
        with patch("subprocess.run") as mock_run:
            wrapper.create_slab("cell.cif", "out.lmp")
            args = mock_run.call_args[0][0]
            # Expected: atomsk cell.cif -duplicate 2 2 1 -orthogonal-cell out.lmp
            assert "-orthogonal-cell" in args
            assert "-duplicate" in args
