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

from .config import SysVars
from .reader import defcond, datread
from .sfecalc import SfeCalcState, chmpot
from .output import OutputState, wrtresl


def main() -> None:
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
