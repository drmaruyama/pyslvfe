# -*- coding: utf-8 -*-
"""
Port of `program sfemain` (slvfe.F90).

This module holds the actual CLI logic as a plain function (`main()`)
so it can be invoked several ways:
    - `python slvfe.py`            (top-level convenience script)
    - `python -m slvfe`            (via src/slvfe/__main__.py)
    - the `pyslvfe` console script (after `pip install .`)

In every case, it mirrors the Fortran executable: it reads
`parameters_fe` (namelist `fevars`) from the current directory, plus
`soln/parameters_er` (namelist `ene_param`, only if `ljlrc = 'yes'`),
and the `soln/`, `refs/` data files described by those namelists.
"""
from __future__ import annotations

import sys

from .config import SysVars
from .exceptions import SlvfeError
from .reader import defcond, datread
from .sfecalc import SfeCalcState, chmpot
from .output import OutputState, wrtresl


def run() -> None:
    """The actual computation, as a plain function that raises
    `SlvfeError` on user-facing/data errors (bad input, inconsistent
    parameters, ...). Library users should call this directly and
    catch `SlvfeError` themselves; `main()` below is the CLI wrapper
    that instead prints a clean message and exits.
    """
    sv = SysVars()
    sv.init_sysvars()
    defcond(sv)

    cs = SfeCalcState()
    for cntrun in range(1, sv.numrun + 1):
        datread(sv, cntrun)
        for prmcnt in range(1, sv.prmmax + 1):
            chmpot(sv, cs, prmcnt, cntrun)

    ost = OutputState()
    wrtresl(sv, ost)


def main() -> None:
    """CLI entry point (`pyslvfe`, `python -m slvfe`, `python slvfe.py`).

    Catches `SlvfeError` and prints a one-line message instead of a
    full traceback -- matching the behavior of the Fortran original's
    `stop 'message'`, and of this port's own earlier
    `raise SystemExit(...)`-based error handling.
    """
    try:
        run()
    except SlvfeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
