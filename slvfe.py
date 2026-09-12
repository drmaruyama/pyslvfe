#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convenience entry point for running this tool without `pip install`ing
it first:

    cd <directory containing parameters_fe, soln/, refs/>
    python /path/to/slvfe.py

(If you *have* installed this package -- `pip install .` from this
repository -- use the `pyslvfe` command instead, or `python -m slvfe`;
this script and both of those run the exact same code in
`src/slvfe/main.py`.)

The command is deliberately not called `slvfe` when installed via pip,
to avoid shadowing the original Fortran `slvfe` binary on $PATH.
"""
import sys
from pathlib import Path

# Make the local src/slvfe package importable without installing it.
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from slvfe.main import main  # noqa: E402

if __name__ == "__main__":
    main()
