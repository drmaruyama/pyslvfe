# -*- coding: utf-8 -*-
"""
Port of the `sysvars` module (sfemain.F90): global run parameters read
from the `parameters_fe` Fortran namelist, plus the arrays that other
modules used to access as globals.

Namelist parsing uses the small self-contained parser in
`namelist_parser.py` (no external dependency required).

Internal structure
-------------------
The Fortran `sysvars` module mixed two very different kinds of state
into one flat set of module-level globals: scalar run *parameters*
(`clcond`, `kT`, `numslv`, ...) and large *data arrays* populated while
reading input files (`rddst`, `rdcor`, `chmpt`, ...). This file keeps
that distinction internally -- `Config` holds the former, `RunData` the
latter -- while `SysVars` remains a single flat object that forwards
attribute access to whichever of the two owns a given name. This means
every other module in this package can keep writing `sv.numslv`,
`sv.rddst`, `sv.get_suffix(...)`, etc. exactly as before; only this
file needs to know the two are actually separate objects
(`sv.cfg` / `sv.data`, also directly accessible if wanted).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from .exceptions import SlvfeError
from .namelist_parser import read_namelist


def zero_padded_str(n: int, digits: int) -> str:
    """Port of `zero_padded_str` in sfemain.F90."""
    if digits <= 0:
        raise ValueError("Invalid args to zero_padded_str()")
    return str(n).zfill(digits)


@dataclass
class Config:
    """Scalar run parameters: namelist values (defaults mirror
    sfemain.F90) plus the handful of scalars derived from them
    (`numslv`, `ermax`, `kT`, ...). No large arrays live here -- see
    `RunData`.
    """
    # ---- namelist parameters ----
    clcond: str = 'merge'
    uvread: str = 'yes'
    slfslt: str = 'yes'
    ljlrc: str = 'not'
    infchk: str = 'not'
    meshread: str = 'not'
    cumuint: str = 'not'
    write_mesherror: str = 'cnd'

    extsln: str = 'lin'
    slncor: str = 'not'
    refmerge: str = 'yes'
    readwgtfl: str = 'yes'

    invmtrx: str = 'reg'
    zerosft: str = 'eczr'
    wrtzrsft: str = 'not'
    wgtfnform: str = 'harm'
    wgtf2smpl: str = 'yes'

    normalize: str = 'yes'
    showdst: str = 'not'

    functional: str = 'pyhnc'

    numprm: int = 0
    numprm_def_inf_yes: int = 11
    numprm_def_inf_not: int = 5

    numsln: int = 0
    numref: int = 0
    numdiv: int = 0
    maxsln: int = 0
    maxref: int = 0
    numrun: int = 0
    prmmax: int = 0

    numslv: int = 0
    ermax: int = 0

    inptemp: float = 300.0
    temp: float = 0.0
    kT: float = 0.0
    slfeng: float = 0.0
    avevolume: float = 0.0
    et: float = 0.0

    pickgr: int = 3
    msemin: int = 1
    msemax: int = 5
    mesherr: float = 0.1

    extthres_soln: int = 1
    extthres_refs: int = 1
    minthres_soln: int = 0
    minthres_refs: int = 0

    norm_error: float = 1.0e-8
    itrmax: int = 100

    ermax_limit: int = 15000
    digits_of_suffix: int = 2

    zero: float = 0.0
    tiny: float = 1.0e-8
    large: int = 500000

    solndirec: str = 'soln'
    refsdirec: str = 'refs'
    wgtslnfl: str = 'weight_soln'
    wgtreffl: str = 'weight_refs'
    slndnspf: str = 'engsln'
    slncorpf: str = 'corsln'
    refdnspf: str = 'engref'
    refcorpf: str = 'corref'
    aveuvfile: str = 'aveuv.tt'
    engmeshfile: str = 'EngMesh'
    cumuintfl: str = 'cumsfe'

    force_calculation: bool = False
    strict_ewald_parameters: bool = False
    check_parameters_er: bool = True

    # ---- derived / runtime flags ----
    suffix_of_engsln_is_tt: bool = False
    suffix_of_engref_is_tt: bool = False

    # ------------------------------------------------------------------
    def get_suffix(self, n: int, suffix_is_tt: bool = False) -> str:
        """Port of `get_suffix` in sfemain.F90."""
        if not suffix_is_tt:
            return zero_padded_str(n, self.digits_of_suffix)
        return 't' * self.digits_of_suffix

    # ------------------------------------------------------------------
    def init_sysvars(self, parmfname: str = 'parameters_fe') -> None:
        """Port of `init_sysvars` in sfemain.F90."""
        p = Path(parmfname)
        if p.exists():
            nml = read_namelist(str(p))
            if 'fevars' in nml:
                for key, value in nml['fevars'].items():
                    if hasattr(self, key):
                        setattr(self, key, value)
                    else:
                        # Silently ignore unknown/unused namelist keys
                        # (e.g. keys only relevant to other ERmod programs).
                        pass

        if self.clcond == 'merge':
            self.suffix_of_engsln_is_tt, count_soln = (
                self._check_tt_or_numeric_files(self.solndirec, self.slndnspf)
            )
            self.suffix_of_engref_is_tt, count_refs = (
                self._check_tt_or_numeric_files(self.refsdirec, self.refdnspf)
            )
            # re-read the namelist: numsln/numref set above may have been
            # explicitly overridden by the user in the namelist, so give
            # user-supplied values priority (mirrors the Fortran code,
            # which re-reads the namelist file after auto-detection).
            if p.exists():
                nml = read_namelist(str(p))
                if 'fevars' in nml:
                    for key in ('numsln', 'numref', 'numdiv'):
                        if key in nml['fevars']:
                            setattr(self, key, nml['fevars'][key])

            if self.numsln <= 0 or self.numsln > count_soln:
                self.numsln = count_soln
            if self.numref <= 0 or self.numref > count_refs:
                self.numref = count_refs

            if self.numdiv <= 0 or self.numdiv >= self.numsln:
                self.numdiv = self.numsln
            if self.numsln % self.numdiv != 0:
                for i in range(self.numdiv + 1, self.numsln + 1):
                    if self.numsln % i == 0:
                        break
                self.numdiv = i

            if self.refmerge == 'not':
                if self.numdiv > self.numref:
                    raise SlvfeError(
                        "With refmerge = 'not', numdiv needs to be not "
                        "larger than numref"
                    )
                if self.numref % self.numdiv != 0:
                    kept = self.numdiv * (self.numref // self.numdiv)
                    print(f" Note: only {kept} files out of {self.numref} "
                          f"engref and corref files prepared")
                    self.numref = kept

        if self.numprm <= 0:
            self.numprm = (self.numprm_def_inf_yes if self.infchk == 'yes'
                            else self.numprm_def_inf_not)

        if self.pickgr < self.msemin:
            raise SlvfeError(" Incorrect setting: pickgr < msemin not allowed")
        if self.pickgr > self.msemax:
            raise SlvfeError(" Incorrect setting: pickgr > msemax not allowed")
        if self.pickgr > self.numprm:
            raise SlvfeError(" Incorrect setting: pickgr > numprm not allowed")

        if self.et != 0.0:
            self.functional = 'thnc'

    # ------------------------------------------------------------------
    def _check_tt_or_numeric_files(self, dirname: str, trunk: str) -> tuple[bool, int]:
        """Port of the nested `check_tt_or_numeric_files` subroutine.

        Returns (use_tt, nfiles).
        """
        sufmax = 99
        opnfile = Path(dirname) / f"{trunk}.{self.get_suffix(1, True)}"
        if opnfile.exists():
            for count_suf in range(1, sufmax + 1):
                opnfile2 = Path(dirname) / f"{trunk}.{self.get_suffix(count_suf)}"
                if opnfile2.exists():
                    raise SlvfeError(
                        f"{trunk}.tt is not supposed to coexist with "
                        f"{trunk}.01, {trunk}.02, ..."
                    )
            return True, 1

        nfiles = 0
        for count_suf in range(1, sufmax + 1):
            opnfile2 = Path(dirname) / f"{trunk}.{self.get_suffix(count_suf)}"
            if opnfile2.exists():
                nfiles = count_suf
            else:
                if count_suf == 1:
                    raise SlvfeError(
                        f"Neither {trunk}.01 nor {trunk}.tt exists. Perhaps "
                        f"{dirname} part is not calculated yet?"
                    )
                break
        return False, nfiles


@dataclass
class RunData:
    """Arrays populated while reading input files (`defcond`/`datread`
    in reader.py) and consumed by the numerical core (`sfecalc.py`) and
    the output routines (`output.py`).

    NOTE: unlike the Fortran code these are stored 0-indexed. `pti`
    (solvent species id) therefore runs 0..numslv-1 here, vs. 1..numslv
    in Fortran. Anywhere the Fortran code used `pti` as an array index
    or compared it to a 1-based id read from file, 1 has been
    subtracted when porting to Python.
    """
    nummol: Optional[np.ndarray] = field(default=None)
    rduvmax: Optional[np.ndarray] = field(default=None)
    rduvcore: Optional[np.ndarray] = field(default=None)
    rdcrd: Optional[np.ndarray] = field(default=None)
    rddst: Optional[np.ndarray] = field(default=None)
    rddns: Optional[np.ndarray] = field(default=None)
    rdslc: Optional[np.ndarray] = field(default=None)
    rdcor: Optional[np.ndarray] = field(default=None)
    rdspec: Optional[np.ndarray] = field(default=None)   # 0-based species id per bin
    chmpt: Optional[np.ndarray] = field(default=None)    # shape (numslv+1, prmmax, numrun)
    aveuv: Optional[np.ndarray] = field(default=None)
    uvene: Optional[np.ndarray] = field(default=None)
    blockuv: Optional[np.ndarray] = field(default=None)
    svgrp: Optional[np.ndarray] = field(default=None)
    svinf: Optional[np.ndarray] = field(default=None)
    wgtsln: Optional[np.ndarray] = field(default=None)
    wgtref: Optional[np.ndarray] = field(default=None)

    # engfile(1..5) from defcond (interactive / basic / range mode only)
    engfile: list = field(default_factory=lambda: [''] * 5)


class SysVars:
    """Flat facade over `Config` + `RunData`.

    Every other module in this package accesses run parameters and
    data arrays alike as `sv.<name>` (attribute *and* method access,
    e.g. `sv.numslv`, `sv.rddst`, `sv.get_suffix(...)`), matching the
    Fortran code's single `sysvars` module. This class provides exactly
    that flat interface while keeping the two kinds of state (settings
    vs. big arrays) in separate, independently testable dataclasses
    internally. `sv.cfg` and `sv.data` are also available directly if
    you want the organized view instead of the flat one.

    Trade-off: attribute access goes through `__getattr__`, which is
    somewhat slower than a plain dataclass field lookup. This matters
    only inside `sfecalc.py`'s per-bin Python loops over very large
    `gemax`; see the performance note in README.md. If that ever
    becomes a bottleneck in practice, the fix is to read the handful of
    hot values (`sv.kT`, `sv.zero`, ...) into local variables once
    before entering such a loop, rather than reverting this split.
    """

    def __init__(self) -> None:
        object.__setattr__(self, 'cfg', Config())
        object.__setattr__(self, 'data', RunData())

    def __getattr__(self, name: str):
        # Only called when normal lookup (instance/class __dict__) misses,
        # i.e. for anything that isn't literally 'cfg' or 'data'.
        cfg = object.__getattribute__(self, 'cfg')
        if hasattr(cfg, name):
            return getattr(cfg, name)
        data = object.__getattribute__(self, 'data')
        if hasattr(data, name):
            return getattr(data, name)
        raise AttributeError(
            f"{type(self).__name__!r} object has no attribute {name!r}"
        )

    def __setattr__(self, name: str, value) -> None:
        if name in ('cfg', 'data'):
            object.__setattr__(self, name, value)
            return
        cfg = object.__getattribute__(self, 'cfg')
        if hasattr(cfg, name):
            setattr(cfg, name, value)
            return
        data = object.__getattribute__(self, 'data')
        if hasattr(data, name):
            setattr(data, name, value)
            return
        raise AttributeError(
            f"{type(self).__name__!r} has no config/data field named "
            f"{name!r}; add it to Config or RunData in config.py first "
            f"(this strictness catches typos that a plain object would "
            f"otherwise silently accept as a new attribute)."
        )
