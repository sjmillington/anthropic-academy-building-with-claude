import os
import sys


def main() -> None:
    """Launch Jupyter Lab via `uv run jupyter lab`."""
    args = ["uv", "run", "jupyter", "lab", *sys.argv[1:]]
    os.execvp(args[0], args)
