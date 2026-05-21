"""Interface wrapper for the Atomsk executable.

This module handles interaction with the external 'atomsk' binary for
generating, manipulating, and converting atomic structures.
"""

import os
import shutil
import subprocess
import threading
from pathlib import Path
from typing import List, Optional, Union

class AtomskError(RuntimeError):
    """Raised when Atomsk execution fails."""

class AtomskWrapper:
    """Wraps the Atomsk binary for Pythonic usage."""

    _lock = threading.Lock()

    def __init__(self, executable_path: Optional[Union[str, Path]] = None):
        """Initialize the wrapper.

        Args:
            executable_path: Path to atomsk binary (searches PATH if None).

        Raises:
            RuntimeError: If atomsk cannot be found.
        """
        self.executable = (
            executable_path
            or os.environ.get("ATOMSK_PATH")
            or shutil.which("atomsk")
        )

        if not self.executable:
            raise RuntimeError(
                "Atomsk binary not found! Install Atomsk and ensure it is in your PATH, "
                "or set ATOMSK_PATH environment variable.\n"
                "Installation: https://atomsk.univ-lille.fr/installation.php"
            )

        self.executable = str(self.executable)

    def run(self, args: List[str], verbose: bool = False) -> None:
        """Execute atomsk command.

        Args:
            args: Command arguments (e.g. ['file.cif', '-duplicate', '2', '2', '1']).
            verbose: If True, show atomsk output.
        """
        cmd = [self.executable, '-ow', '-v', '0'] + [str(a) for a in args]
        self._remove_existing_output(args)

        stdout_setting = None if verbose else subprocess.PIPE
        stderr_setting = None if verbose else subprocess.PIPE

        with self._lock:
            try:
                subprocess.run(
                    cmd,
                    input="n\n",
                    check=True,
                    stdout=stdout_setting,
                    stderr=stderr_setting,
                    text=True
                )
            except subprocess.CalledProcessError as e:
                error_msg = f"Atomsk command failed with exit code {e.returncode}"
                if e.stderr:
                    error_msg += f"\nError Output: {e.stderr}"
                elif not verbose:
                    error_msg += ". Run with verbose=True to see details."
                raise AtomskError(error_msg) from e

    def _remove_existing_output(self, args: List[str]) -> None:
        """Remove existing output file to prevent atomsk prompt hang.

        Args:
            args: Atomsk command arguments.
        """
        if not args:
            return

        input_file = None
        for arg in args:
            arg_str = str(arg)
            if not arg_str.startswith('-') and ('.' in arg_str or '/' in arg_str):
                input_file = Path(arg_str).resolve() if Path(arg_str).exists() else Path(arg_str)
                break

        for arg in reversed(args):
            arg_str = str(arg)
            if arg_str.startswith('-'):
                continue
            if '.' in arg_str or '/' in arg_str:
                output_path = Path(arg_str)
                if input_file is not None:
                    input_resolved = input_file.resolve() if input_file.exists() else input_file
                    output_resolved = output_path.resolve() if output_path.exists() else output_path
                    if input_resolved == output_resolved:
                        return

                if output_path.exists() and output_path.is_file():
                    output_path.unlink()
                break

    def convert(self, input_file: Union[str, Path], output_file: Union[str, Path],
                options: Optional[List[str]] = None) -> None:
        """Convert file from one format to another.

        Args:
            input_file: Source file path.
            output_file: Destination file path (format inferred from extension).
            options: Additional flags.
        """
        args = [str(input_file), str(output_file)]
        if options:
            args.extend(options)
        self.run(args)

    def orthogonalize(self, input_file: Union[str, Path],
                        output_file: Union[str, Path]) -> None:
        """Convert structure to orthogonal cell."""
        self.run([str(input_file), "-orthogonal-cell", str(output_file)])

    def duplicate(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        input_file: Union[str, Path],
        output_file: Union[str, Path],
        nx: int,
        ny: int,
        nz: int,
        center: bool = False
    ) -> None:
        """Duplicate system in x, y, z directions.

        Args:
            input_file: Input file path.
            output_file: Output file path.
            nx: X multiplication factor.
            ny: Y multiplication factor.
            nz: Z multiplication factor.
            center: If True, center to center of mass.
        """
        args = [str(input_file), "-duplicate", str(nx), str(ny), str(nz)]
        if center:
            args.extend(["-center", "com"])
        args.append(str(output_file))
        self.run(args)

    def create_slab(
        self,
        cif_path: Union[str, Path],
        output_path: Union[str, Path],
        pre_duplicate: Optional[List[int]] = None
    ) -> None:
        """Create orthogonalized slab from CIF file.

        Args:
            cif_path: Input CIF file path.
            output_path: Output file path.
            pre_duplicate: Duplication factors [nx, ny, nz]. Defaults to [2, 2, 1].
        """
        if pre_duplicate is None:
            pre_duplicate = [2, 2, 1]

        args = [
            str(cif_path),
            "-duplicate", *map(str, pre_duplicate),
            "-orthogonal-cell",
            str(output_path)
        ]
        self.run(args)

    def center(self, input_file: Union[str, Path], output_file: Union[str, Path]) -> None:
        """Center system to center of mass.

        Args:
            input_file: Input file path.
            output_file: Output file path.
        """
        input_path = Path(input_file).resolve()
        output_path = Path(output_file).resolve()

        if input_path == output_path:
            temp_output = input_path.parent / f".tmp_{input_path.name}"
            self.run([str(input_path), "-center", "com", str(temp_output)])
            shutil.move(str(temp_output), str(input_path))
        else:
            self.run([str(input_file), "-center", "com", str(output_file)])

