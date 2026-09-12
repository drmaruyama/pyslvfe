# -*- coding: utf-8 -*-
"""Allows `python -m slvfe` in addition to the `pyslvfe` console script
installed by pyproject.toml, and the top-level `slvfe.py` convenience
script."""
from .main import main

if __name__ == '__main__':
    main()
