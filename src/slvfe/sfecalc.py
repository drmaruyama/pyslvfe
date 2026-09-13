# -*- coding: utf-8 -*-
"""
Port of the `sfecalc` module (slvfe.F90): the numerical core of the
solvation free-energy calculation, including the two former-GPU solver
calls (`posv_wrap` / `syevr_wrap`, see solver.py).

Indexing convention used throughout this file (differs from Fortran):
    - Solvent species id `pti` is 0-based here (0 .. numslv-1),
      vs. 1-based in Fortran (1 .. numslv).
    - All bin indices (iduv, iduvp, k, m, ...) are 0-based.
    - The Fortran `cnt` flag ("1 = solution, 2 = reference solvent") is
      replaced by the `System` enum below, so call sites read
      `System.SOLUTION` / `System.REFERENCE` instead of bare `1` / `2`.
      `System` is an `IntEnum`, so it still compares equal to the plain
      integers used in a couple of physics formulas (`pyhnc`'s special
      third branch, see `PYHNC_INDIRECT`).
"""
from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import Callable, Optional

import numpy as np
from scipy import sparse

from .config import SysVars
from .fortran_utils import nint
from .solver import posv_wrap, syevr_wrap
from .uvcorrect import ljcorrect, LJState
from .exceptions import SlvfeError


class System(IntEnum):
    """Which of the two parallel sub-systems a computation refers to.

    Replaces the Fortran `cnt` flag (1 = solution, 2 = reference
    solvent) used throughout `sfecalc` / `slvfe.F90`.
    """
    SOLUTION = 1
    REFERENCE = 2


# `pyhnc`'s third, "special case" branch (used only for the sdrcv-based
# correction term in `chmpot`). It is not a third "system" -- it's a
# distinct closure-formula variant -- so it is kept as a separate,
# clearly-named constant rather than a member of `System`.
PYHNC_INDIRECT = 3


@dataclass
class SfeCalcState:
    """Working arrays local to one `chmpot` call, plus the handful of
    values that were Fortran `SAVE` (persistent) local variables."""
    gemax: int = 0
    idrduv: Optional[np.ndarray] = None
    uvmax: Optional[np.ndarray] = None
    uvcrd: Optional[np.ndarray] = None
    edist: Optional[np.ndarray] = None
    edens: Optional[np.ndarray] = None
    edscr: Optional[np.ndarray] = None
    ecorr: Optional[np.ndarray] = None
    uvspec: Optional[np.ndarray] = None
    slncv: Optional[np.ndarray] = None
    inscv: Optional[np.ndarray] = None
    sdrcv: Optional[np.ndarray] = None
    zrsln: Optional[np.ndarray] = None
    zrref: Optional[np.ndarray] = None
    zrsdr: Optional[np.ndarray] = None

    # Fortran `logical, save :: first_time` in getinscv
    invmtrx_first_time: bool = True
    # Fortran `real, allocatable, save :: cumsfe(:,:)` in chmpot
    cumsfe: Optional[np.ndarray] = None
    # Fortran `uvcorrect` module's SAVE variables (see uvcorrect.ljcorrect)
    lj_state: LJState = field(default_factory=LJState)


# ---------------------------------------------------------------------
# small helper functions (wgtmxco, cvfcen, getwght, zeroec, wgtdst,
# sfewgt, thnc, pyhnc)
# ---------------------------------------------------------------------

def wgtmxco(sv: SysVars, pti: int) -> float:
    return 1.0 / sv.nummol[pti]


def wgtdst(sv: SysVars, cs: SfeCalcState, iduv: int, system: System,
           systype: str, wgttype: str) -> float:
    """Weight of bin `iduv` for the "harm"/"geom"/"smpl" weighting
    schemes used to average slncv/inscv/uvcrd over a species.

    For `System.REFERENCE`, using `systype in ('slncv', 'extsl')` is a
    programming error *unless* the caller also asked for the "just use
    the reference density directly" shortcut (`wgttype == 'smpl'` or
    `sv.wgtf2smpl == 'yes'`), which takes priority over that check --
    mirroring the Fortran `jdg` flag's sequential `if` assignments,
    where the later "use fref directly" assignment could silently
    override the earlier "this is a bug" marker.
    """
    fsln = cs.edist[iduv]
    fref = cs.edens[iduv]

    use_fref_directly = (system == System.REFERENCE
                          and (wgttype == 'smpl' or sv.wgtf2smpl == 'yes'))
    if use_fref_directly:
        return fref

    if system == System.REFERENCE and systype in ('slncv', 'extsl'):
        raise SlvfeError(' Bug in the program')

    if wgttype == 'smpl':
        # System.REFERENCE + 'smpl' was already handled by
        # `use_fref_directly` above, so this is always System.SOLUTION.
        return fsln
    if wgttype == 'geom':
        factor = fsln * fref
        return math.sqrt(factor) if factor > sv.zero else 0.0
    # 'harm' (default)
    factor = fsln + fref
    return fsln * fref / factor if factor > sv.zero else 0.0


def getwght(sv: SysVars, cs: SfeCalcState, pti: int, system: System,
            systype: str, wgttype: str, engtype: str) -> np.ndarray:
    gemax = cs.gemax
    weight = np.zeros(gemax, dtype=np.float64)
    mask = cs.uvspec == pti
    idx = np.nonzero(mask)[0]
    for iduv in idx:
        weight[iduv] = wgtdst(sv, cs, iduv, system, systype, wgttype)
    if engtype == 'yes':
        sel = mask & (weight > 0)
        minuv = np.min(np.abs(cs.uvcrd[sel]))
        for iduv in idx:
            ampl = sv.nummol[pti] * (abs(cs.uvcrd[iduv]) - minuv)
            weight[iduv] = math.exp(-ampl / sv.kT) * weight[iduv]
    ampl = weight[mask].sum()
    if ampl > sv.zero:
        weight = weight / ampl
    else:
        raise SlvfeError(f' Zero weight at {pti + 1}')
    return weight


def cvfcen(sv: SysVars, cs: SfeCalcState, pti: int, system: System,
           systype: str, wgttype: str, engtype: str) -> float:
    weight = getwght(sv, cs, pti, system, systype, wgttype, engtype)
    mask = cs.uvspec == pti
    errtag = False
    if systype == 'slncv':
        if system == System.SOLUTION:
            vals = cs.slncv
        else:
            errtag = True
            vals = None
    elif systype == 'inscv':
        vals = cs.sdrcv if system == System.SOLUTION else cs.inscv
    elif systype == 'uvcrd':
        vals = cs.uvcrd
    else:
        errtag = True
        vals = None
    if errtag:
        raise SlvfeError(' Bug in the program')
    return float(np.sum(vals[mask] * weight[mask]))


def zeroec(sv: SysVars, cs: SfeCalcState, pti: int, system: System) -> int:
    """Returns the (0-based) bin index at which the solute-solvent
    energy coordinate crosses zero for species `pti`.

    Raises if no such bin exists for this species/mesh (e.g. the mesh
    is too coarse to have more than one bin for this species). The
    Fortran original left the return value undefined in this case
    (uninitialized local variable); we raise instead of silently
    returning a wrong bin index -- or, worse, letting a Python `None`
    slip into a NumPy index expression, which is silently interpreted
    as `numpy.newaxis` rather than raising.
    """
    k = None
    for iduv in range(cs.gemax - 1):
        if cs.uvspec[iduv] != pti:
            continue
        if cs.uvcrd[iduv] <= 0.0 and cs.uvcrd[iduv + 1] >= 0.0:
            factor = abs(cs.uvcrd[iduv])
            ampl = cs.uvcrd[iduv + 1]
            if ampl > factor + sv.tiny:
                k = iduv
            elif ampl < factor - sv.tiny:
                k = iduv + 1
            elif abs(ampl - factor) <= sv.tiny:
                if system == System.SOLUTION:
                    lcsln, lcref = cs.edist[iduv], cs.edist[iduv + 1]
                else:
                    lcsln, lcref = cs.edens[iduv], cs.edens[iduv + 1]
                k = iduv if lcsln >= lcref else iduv + 1
            else:
                raise SlvfeError('Bug in the program')
    if k is None:
        raise SlvfeError(
            f"zeroec: no bin found where the solute-solvent energy "
            f"coordinate changes sign for species {pti + 1} "
            f"(mesh too coarse for this grouping?)"
        )
    return k


def sfewgt(fsln: float, fref: float) -> float:
    if fsln >= fref:
        return 1.0
    factor = (fsln - fref) / (fsln + fref)
    return 1.0 - factor ** 2


def thnc(sv: SysVars, indpmf: float, et: float, system: System) -> float:
    factor = indpmf / sv.kT
    if system == System.SOLUTION:
        intg = factor / 2.0
    elif system == System.REFERENCE:
        if indpmf < et:
            factor = 0.0
        intg = factor / 2.0
    else:
        raise SlvfeError("Incorrct cnt argument in thnc")
    return intg


def pyhnc(sv: SysVars, indpmf: float, system: System | int) -> float:
    """``system`` is usually a `System` member, but also accepts the
    literal `PYHNC_INDIRECT` (3) for the sdrcv-based special case used
    in `chmpot` (see module docstring)."""
    factor = indpmf / sv.kT
    if system in (System.SOLUTION, System.REFERENCE):
        if factor < -sv.zero:      # PY
            if system == System.SOLUTION:
                intg = factor + factor / (math.exp(-factor) - 1.0)
            else:
                intg = math.log(1.0 - factor) * (1.0 / factor - 1.0)
            intg += 1.0
        else:                       # HNC
            intg = factor / 2.0
    elif system == PYHNC_INDIRECT:
        if factor > sv.zero:
            intg = 1.0 - math.log(1.0 + factor) * (1.0 / factor + 1.0)
        else:
            intg = -factor / 2.0
    else:
        raise SlvfeError("Incorrct cnt argument in pyhnc")
    return intg


# ---------------------------------------------------------------------
# `functional` dispatch table
#
# Each entry computes (lcsln, lcref) -- the "closure" contributions from
# the solution side and the reference-solvent side respectively -- for
# one bin, given its slncv/inscv values. This replaces the Fortran
# `select case(functional)` block in `chmpot`. Because the functional
# form is fixed for the whole `chmpot` call, the *lookup* happens once
# (see `chmpot` below) rather than once per bin, which is both clearer
# and cheaper than re-testing `functional` on every iteration.
# ---------------------------------------------------------------------

def _functional_pyhnc(sv: SysVars, slncv: float, inscv: float) -> tuple[float, float]:
    return pyhnc(sv, slncv, System.SOLUTION), pyhnc(sv, inscv, System.REFERENCE)


def _functional_mpyhnc(sv: SysVars, slncv: float, inscv: float) -> tuple[float, float]:
    return thnc(sv, slncv, -math.inf, System.SOLUTION), pyhnc(sv, inscv, System.REFERENCE)


def _functional_hnc(sv: SysVars, slncv: float, inscv: float) -> tuple[float, float]:
    return (thnc(sv, slncv, -math.inf, System.SOLUTION),
            thnc(sv, inscv, -math.inf, System.REFERENCE))


def _functional_zero(sv: SysVars, slncv: float, inscv: float) -> tuple[float, float]:
    return 0.0, 0.0


def _functional_thnc(sv: SysVars, slncv: float, inscv: float) -> tuple[float, float]:
    return thnc(sv, slncv, sv.et, System.SOLUTION), thnc(sv, inscv, sv.et, System.REFERENCE)


FUNCTIONAL_TABLE = {
    'pyhnc': _functional_pyhnc,
    'mpyhnc': _functional_mpyhnc,
    'hnc': _functional_hnc,
    'zero': _functional_zero,
    'thnc': _functional_thnc,
}


# ---------------------------------------------------------------------
# distnorm / distshow
# ---------------------------------------------------------------------

def distnorm(sv: SysVars, cs: SfeCalcState) -> None:
    for system in (System.SOLUTION, System.REFERENCE):
        if system == System.SOLUTION:
            edhst = cs.edist.copy()
            edmcr = cs.edscr.copy() if sv.slncor == 'yes' else None
        else:
            edhst = cs.edens.copy()
            edmcr = cs.ecorr.copy()

        for pti in range(sv.numslv):
            mask = cs.uvspec == pti
            factor = edhst[mask].sum()
            factor = sv.nummol[pti] / factor if factor > sv.zero else 0.0
            edhst[mask] *= factor

        if not (system == System.SOLUTION and sv.slncor != 'yes'):
            errtmp = sv.norm_error + 1.0
            itrcnt = 0
            correc = np.ones(cs.gemax, dtype=np.float64)
            # `ampl = correc[mask] @ edmcr[mask, :]` is a BLAS matrix-vector
            # product; some optimized BLAS builds can leave the CPU's
            # "divide by zero" floating-point flag set afterwards (e.g. via
            # internal approximate-reciprocal SIMD instructions), which
            # NumPy then reports at the *next* floating-point operation --
            # here, the division just below -- even though that division is
            # already guarded (`np.where(ampl > sv.zero, ampl, 1.0)` never
            # lets a zero reach the denominator). This block suppresses
            # that known-spurious warning; it does not change any computed
            # value.
            with np.errstate(divide='ignore', invalid='ignore'):
                while errtmp > sv.norm_error and itrcnt <= sv.itrmax:
                    lcsln_total = np.zeros(cs.gemax, dtype=np.float64)
                    for pti in range(sv.numslv):
                        mask = cs.uvspec == pti
                        ampl = correc[mask] @ edmcr[mask, :]
                        contrib = np.where(ampl > sv.zero,
                                            sv.nummol[pti] / np.where(ampl > sv.zero, ampl, 1.0),
                                            0.0)
                        lcsln_total += contrib
                    lcsln_total /= sv.numslv
                    correc = lcsln_total * edhst
                    edmcr = edmcr * np.outer(correc, correc)
                    sel = edhst > sv.zero
                    errtmp = np.max(np.abs(correc[sel] - 1.0)) if np.any(sel) else 0.0
                    itrcnt += 1
                    if itrcnt >= sv.itrmax:
                        raise SlvfeError(
                            ' The optimization of the correlation matrix\n'
                            f'  did not converge with an error of {errtmp}'
                        )

        if system == System.SOLUTION:
            cs.edist[:] = edhst
            if sv.slncor == 'yes':
                cs.edscr[:, :] = edmcr
        else:
            cs.edens[:] = edhst
            cs.ecorr[:, :] = edmcr


def distshow(sv: SysVars, cs: SfeCalcState) -> None:
    ilist = np.arange(cs.gemax)
    print()
    for pti in range(sv.numslv):
        for system in (System.SOLUTION, System.REFERENCE):
            if system == System.SOLUTION:
                edhst = cs.edist
                label = " SOLUTION" if sv.numslv == 1 else f" SOLUTION for{pti + 1:4d}-th species"
            else:
                edhst = cs.edens
                label = " INSERTION" if sv.numslv == 1 else f" INSERTION for{pti + 1:4d}-th species"
            print(label)

            mask = (cs.uvspec == pti) & (edhst > sv.zero)
            if not np.any(mask):
                continue
            ecmin = int(ilist[mask].min())
            ecmax = int(ilist[mask].max())
            window = edhst[ecmin:ecmax + 1]
            k = int(np.count_nonzero(window > sv.zero))
            ratio = k / (ecmax - ecmin + 1)
            factor = window[window > sv.zero].sum()

            if system == System.REFERENCE:
                for iduv in range(ecmin, ecmax + 1):
                    if edhst[iduv] <= sv.zero:
                        print(f"     No sampling at {iduv + 1:5d}"
                              f" with energy {cs.uvcrd[iduv]:14.6g}")

            if system == System.SOLUTION:
                print(f"     Nonzero component ratio in solution  = {ratio:12.4g}")
            else:
                print(f"     Nonzero component ratio at insertion = {ratio:12.4g}")
            print(f"          Number of interacting molecules = {factor:12.4g}")

            if system == System.SOLUTION:
                print(f"     Minimum energy in solution   ={cs.uvcrd[ecmin]:15.7g}")
                print(f"     Maximum energy in solution   ={cs.uvcrd[ecmax]:15.7g}")
            else:
                print(f"     Minimum energy at insertion  ={cs.uvcrd[ecmin]:15.7g}")
                print(f"     Maximum energy at insertion  ={cs.uvcrd[ecmax]:15.7g}")


# ---------------------------------------------------------------------
# getslncv
# ---------------------------------------------------------------------

def _slncv_zeroshift_mxco(sv: SysVars, cs: SfeCalcState, pti: int) -> float:
    factor = wgtmxco(sv, pti)
    return (factor * cvfcen(sv, cs, pti, System.SOLUTION, 'slncv', sv.wgtfnform, 'not')
            - (1.0 - factor) * cvfcen(sv, cs, pti, System.REFERENCE, 'uvcrd', 'smpl', 'yes'))


# `zerosft` dispatch for getslncv's additive-constant fixup. Keyed by
# `sv.zerosft`; replaces the Fortran `select case(zerosft)` block.
_SLNCV_ZEROSHIFT: dict[str, Callable[[SysVars, SfeCalcState, int], float]] = {
    'eczr': lambda sv, cs, pti: 0.0,
    'orig': lambda sv, cs, pti: 0.0,
    'mxco': _slncv_zeroshift_mxco,
    'zero': lambda sv, cs, pti: cvfcen(sv, cs, pti, System.SOLUTION, 'slncv', sv.wgtfnform, 'yes'),
    'cntr': lambda sv, cs, pti: cvfcen(sv, cs, pti, System.SOLUTION, 'slncv', sv.wgtfnform, 'not'),
}


def getslncv(sv: SysVars, cs: SfeCalcState) -> None:
    gemax = cs.gemax
    ofdmp = 10  # factor to suppress the integer overflow

    # NOTE: min_rddst / min_rddns in the Fortran code are taken over the
    # *raw* (ungrouped) rddst/rddns arrays (module-level, from sysread),
    # not the grouped edist/edens -- see the original getslncv, which
    # uses `rddst`/`rddns` directly.
    min_rddst = sv.rddst[sv.rddst > sv.zero].min()
    min_rddns = sv.rddns[sv.rddns > sv.zero].min()

    ext_target = np.ones(gemax, dtype=bool)
    for iduv in range(gemax):
        m1 = ofdmp * sv.extthres_soln
        factor = cs.edist[iduv] / min_rddst
        j = m1 if factor > m1 else nint(factor)
        m2 = ofdmp * sv.extthres_refs
        factor2 = cs.edens[iduv] / min_rddns
        k = m2 if factor2 > m2 else nint(factor2)
        if j < sv.extthres_soln or k < sv.extthres_refs:
            ext_target[iduv] = False

    slncv = cs.slncv
    mask = ext_target
    slncv[mask] = -sv.kT * np.log(cs.edist[mask] / cs.edens[mask]) - cs.uvcrd[mask]

    for iduv in range(gemax):
        if ext_target[iduv]:
            continue
        if cs.edist[iduv] <= sv.zero:
            slncv[iduv] = 0.0
            continue
        pti = cs.uvspec[iduv]

        m = 0
        for iduvp in range(iduv):
            if cs.uvspec[iduvp] == pti and ext_target[iduvp] and m < iduvp:
                m = iduvp
        k = gemax - 1
        for iduvp in range(gemax - 1, iduv, -1):
            if cs.uvspec[iduvp] == pti and ext_target[iduvp] and k > iduvp:
                k = iduvp

        if sv.extsln == 'sim':
            if abs(m - iduv) < abs(k - iduv):
                factor = slncv[m]
            elif abs(m - iduv) > abs(k - iduv):
                factor = slncv[k]
            else:
                factor = (slncv[m] + slncv[k]) / 2.0
        else:
            if ext_target[m] and ext_target[k]:
                j = k if abs(m - iduv) >= abs(k - iduv) else m
            elif ext_target[m] and not ext_target[k]:
                j = m
            elif (not ext_target[m]) and ext_target[k]:
                j = k
            else:
                raise SlvfeError(f"Extrapolation is not possible at {cs.uvcrd[iduv]}")

            work = np.zeros(gemax, dtype=np.float64)
            for iduvp in range(gemax):
                if cs.uvspec[iduvp] == pti and ext_target[iduvp]:
                    if iduvp == iduv:
                        raise SlvfeError(' A bug in program or data')
                    factor = cs.uvcrd[iduvp] - cs.uvcrd[j]
                    if iduvp < iduv:
                        factor = -factor - 2.0 * (cs.uvcrd[j] - cs.uvcrd[iduv])
                    work[iduvp] = math.exp(-factor / sv.kT) * wgtdst(
                        sv, cs, iduvp, System.SOLUTION, 'extsl', sv.wgtfnform
                    )
            sel = work > sv.zero
            factor = work[sel].sum()
            work = work / factor
            mat11 = np.sum(work[sel] * cs.uvcrd[sel])
            mat22 = np.sum(work[sel] * cs.uvcrd[sel] * cs.uvcrd[sel])
            mat12 = np.sum(work[sel] * slncv[sel])
            mat21 = np.sum(work[sel] * cs.uvcrd[sel] * slncv[sel])
            w1 = (mat22 * mat12 - mat11 * mat21) / (mat22 - mat11 ** 2)
            w2 = (mat21 - mat11 * mat12) / (mat22 - mat11 ** 2)
            factor = w1 + w2 * cs.uvcrd[iduv]
        slncv[iduv] = factor

    for pti in range(sv.numslv):
        try:
            cvzero = _SLNCV_ZEROSHIFT[sv.zerosft](sv, cs, pti)
        except KeyError:
            raise SlvfeError(' zerosft not properly set ')
        mask = cs.uvspec == pti
        slncv[mask] -= cvzero
        cs.zrsln[pti] = cvzero


# ---------------------------------------------------------------------
# getinscv
# ---------------------------------------------------------------------

@dataclass
class _SystemArrays:
    """The set of `SfeCalcState` arrays associated with one `System`.

    Replaces the repeated `if cnt == 1: ... else: ...` array-selection
    blocks that appeared throughout the Fortran-derived `getinscv`.
    `target` is a *view* onto `cs.sdrcv` / `cs.inscv` (not a copy), so
    writing into it (`arrs.target[:] = ...`) mutates `cs` in place, the
    same way the Fortran code wrote directly into `sdrcv`/`inscv`.
    """
    edvec: np.ndarray       # cs.edist (solution) / cs.edens (reference)
    edmcr: np.ndarray       # cs.edscr (solution) / cs.ecorr (reference)
    target: np.ndarray      # cs.sdrcv (solution) / cs.inscv (reference)
    zeroshift: np.ndarray   # cs.zrsdr (solution) / cs.zrref (reference)


def _system_arrays(cs: SfeCalcState, system: System) -> _SystemArrays:
    if system == System.SOLUTION:
        return _SystemArrays(edvec=cs.edist, edmcr=cs.edscr,
                              target=cs.sdrcv, zeroshift=cs.zrsdr)
    return _SystemArrays(edvec=cs.edens, edmcr=cs.ecorr,
                          target=cs.inscv, zeroshift=cs.zrref)


def _zeroshift_ref_energy(sv: SysVars, cs: SfeCalcState, pti: int,
                           system: System, arrs: _SystemArrays) -> float:
    """"reference solute-solvent energy" additive constant (first
    `zerosft` block in the Fortran `getinscv`). Unlike the other two
    dispatch tables in this file, an unrecognized `zerosft` silently
    falls back to 0.0 here, matching the Fortran `case default`."""
    if sv.zerosft in ('eczr', 'orig'):
        return arrs.target[zeroec(sv, cs, pti, system)]
    if sv.zerosft == 'mxco':
        return cvfcen(sv, cs, pti, system, 'inscv', 'smpl', 'yes')
    return 0.0


def _pmf_zeroshift_mxco(sv: SysVars, cs: SfeCalcState, pti: int,
                         system: System, zerouv: np.ndarray) -> float:
    factor = wgtmxco(sv, pti)
    return (factor * cvfcen(sv, cs, pti, system, 'inscv', sv.wgtfnform, 'not')
            - (1.0 - factor) * zerouv[pti])


# `zerosft` dispatch for fixing the additive constant of the indirect
# PMF (second `zerosft` block in the Fortran `getinscv`).
_PMF_ZEROSHIFT: dict[str, Callable[[SysVars, SfeCalcState, int, System, np.ndarray], float]] = {
    'eczr': lambda sv, cs, pti, system, zerouv: -zerouv[pti],
    'orig': lambda sv, cs, pti, system, zerouv: -zerouv[pti],
    'mxco': _pmf_zeroshift_mxco,
    'zero': lambda sv, cs, pti, system, zerouv: cvfcen(sv, cs, pti, system, 'inscv', sv.wgtfnform, 'yes'),
    'cntr': lambda sv, cs, pti, system, zerouv: cvfcen(sv, cs, pti, system, 'inscv', sv.wgtfnform, 'not'),
}


def getinscv(sv: SysVars, cs: SfeCalcState) -> None:
    if cs.invmtrx_first_time:
        if sv.invmtrx not in ('gce', 'reg', 'evd'):
            sv.invmtrx = 'reg'
        cs.invmtrx_first_time = False

    for system in (System.SOLUTION, System.REFERENCE):
        if system == System.SOLUTION and sv.slncor != 'yes':
            continue
        invmtrx_cnt = sv.invmtrx
        arrs = _system_arrays(cs, system)

        edvec = arrs.edvec.copy()
        edmcr = arrs.edmcr.copy()

        dns = edvec[np.newaxis, :]          # dns(iduv)  -> broadcast over columns
        dnsp = edvec[:, np.newaxis]         # dnsp(iduvp)-> broadcast over rows
        both_pos = (dns > sv.zero) & (dnsp > sv.zero)
        dmcr = np.where(both_pos, edmcr - dnsp * dns, 0.0)
        eye = np.eye(cs.gemax, dtype=np.float64)
        edmcr = np.where(both_pos, dmcr, eye)

        ddiff = -sv.kT * (cs.edist - cs.edens)

        inv_info = 1
        if invmtrx_cnt in ('gce', 'reg'):
            edmcr_inv = edmcr.copy()
            work = np.where(edvec > sv.zero, ddiff, 0.0)

            if invmtrx_cnt == 'gce':
                for pti in range(sv.numslv):
                    k = zeroec(sv, cs, pti, system)
                    edmcr_inv[k, :] = 0.0
                    edmcr_inv[:, k] = 0.0
                    edmcr_inv[k, k] = 1.0
                    work[k] = 0.0
            else:  # regularization
                regfac = np.zeros(sv.numslv, dtype=np.float64)
                regcnt = np.zeros(sv.numslv, dtype=np.float64)
                pos = edvec > sv.zero
                for iduv in np.nonzero(pos)[0]:
                    pti = cs.uvspec[iduv]
                    regfac[pti] += edmcr[iduv, iduv]
                    regcnt[pti] += 1.0
                regfac = regfac / regcnt / regcnt
                for iduv in np.nonzero(pos)[0]:
                    pti = cs.uvspec[iduv]
                    sel = (cs.uvspec == pti) & pos
                    edmcr_inv[sel, iduv] += regfac[pti]

            x, inv_info = posv_wrap(edmcr_inv, work)
            if inv_info == 0:
                arrs.target[:] = x
            else:
                print("Cholesky-based linear solver failed to converge. "
                      "Falling back to slower EVD-based solver.", file=sys.stderr)
                invmtrx_cnt = 'evd'

        if invmtrx_cnt == 'evd':
            eigval, eigvec, evd_info = syevr_wrap(edmcr)
            if evd_info != 0:
                raise SlvfeError("Failed inversion of correlation matrix")
            pos = edvec > sv.zero
            work = np.zeros(cs.gemax, dtype=np.float64)
            pti0 = sv.numslv  # first `numslv` eigenvalues skipped (0-based start index)
            for iduv in range(pti0, cs.gemax):
                factor = np.sum(ddiff[pos] * eigvec[pos, iduv])
                work[iduv] = factor / eigval[iduv]
            for iduv in range(cs.gemax):
                arrs.target[iduv] = float(eigvec[iduv, pti0:cs.gemax] @ work[pti0:cs.gemax])

        # reference solute-solvent energy to fix the additive constant
        zerouv = np.zeros(sv.numslv, dtype=np.float64)
        for pti in range(sv.numslv):
            zerouv[pti] = _zeroshift_ref_energy(sv, cs, pti, system, arrs)

        # conversion from solute-solvent potential to indirect PMF
        pos = edvec > sv.zero
        new_target = np.zeros(cs.gemax, dtype=np.float64)
        new_target[pos] = ddiff[pos] / edvec[pos] - arrs.target[pos]
        arrs.target[:] = new_target

        # fixing the additive constant for indirect PMF
        for pti in range(sv.numslv):
            try:
                cvzero = _PMF_ZEROSHIFT[sv.zerosft](sv, cs, pti, system, zerouv)
            except KeyError:
                raise SlvfeError(' zerosft not properly set ')
            mask = cs.uvspec == pti
            arrs.target[mask] -= cvzero
            arrs.zeroshift[pti] = cvzero


# ---------------------------------------------------------------------
# chmpot
# ---------------------------------------------------------------------

def chmpot(sv: SysVars, cs: SfeCalcState, prmcnt: int, cntrun: int) -> None:
    """``prmcnt`` and ``cntrun`` are 1-based (matching the Fortran call
    convention retained in main.py)."""
    group = int(sv.svgrp[prmcnt - 1])
    inft = int(sv.svinf[prmcnt - 1])
    numslv = sv.numslv
    ermax = sv.ermax

    # --- mesh grouping: idrduv (ermax,) maps original bin -> grouped bin ---
    uvmax = np.zeros(numslv, dtype=np.int64)
    for pti in range(numslv):
        k = (sv.rduvcore[pti] * inft) // 100
        uvmax[pti] = (sv.rduvmax[pti] - k) // group
    gemax = int(uvmax.sum())
    cs.gemax = gemax

    rduvmax_cum = np.concatenate(([0], np.cumsum(sv.rduvmax)))
    uvmax_cum = np.concatenate(([0], np.cumsum(uvmax)))
    idrduv = np.zeros(ermax, dtype=np.int64)
    for pti in range(numslv):
        cnt0 = int(rduvmax_cum[pti])
        rmax = int(sv.rduvmax[pti])
        m = int(uvmax[pti])
        j = int(uvmax_cum[pti])
        local = np.arange(rmax)
        k2 = np.minimum(local // group, m - 1)
        idrduv[cnt0:cnt0 + rmax] = j + k2
    if int(rduvmax_cum[-1]) != ermax:
        raise SlvfeError(
            f"Error: The total no. of meshes does not match with input "
            f"(Sum should be {ermax} but was {int(rduvmax_cum[-1])})"
        )
    cs.idrduv = idrduv
    cs.uvmax = uvmax

    # --- grouped coordinate / distribution / density / species arrays ---
    gpnum = np.bincount(idrduv, minlength=gemax)
    uvcrd = np.bincount(idrduv, weights=sv.rdcrd, minlength=gemax)
    uvcrd = np.divide(uvcrd, gpnum, out=np.zeros_like(uvcrd), where=gpnum > 0)
    uvspec = np.zeros(gemax, dtype=np.int64)
    uvspec[idrduv] = sv.rdspec
    for pti in range(numslv):
        cnt0 = int(rduvmax_cum[pti + 1]) - 1
        k0 = int(uvmax_cum[pti + 1]) - 1
        uvcrd[k0] = sv.rdcrd[cnt0]
    cs.uvcrd = uvcrd
    cs.uvspec = uvspec

    edist = np.bincount(idrduv, weights=sv.rddst, minlength=gemax)
    edens = np.bincount(idrduv, weights=sv.rddns, minlength=gemax)
    cs.edist = edist
    cs.edens = edens

    # G: (ermax, gemax) grouping indicator; edscr = G^T rdslc G, ecorr = G^T rdcor G
    G = sparse.csr_matrix((np.ones(ermax), (np.arange(ermax), idrduv)),
                           shape=(ermax, gemax))
    if sv.slncor == 'yes':
        cs.edscr = np.asarray(G.T @ sv.rdslc @ G)
    else:
        cs.edscr = None
    cs.ecorr = np.asarray(G.T @ sv.rdcor @ G)

    cs.slncv = np.zeros(gemax, dtype=np.float64)
    cs.inscv = np.zeros(gemax, dtype=np.float64)
    cs.zrsln = np.zeros(numslv, dtype=np.float64)
    cs.zrref = np.zeros(numslv, dtype=np.float64)
    if sv.slncor == 'yes':
        cs.sdrcv = np.zeros(gemax, dtype=np.float64)
        cs.zrsdr = np.zeros(numslv, dtype=np.float64)

    if sv.normalize == 'yes':
        distnorm(sv, cs)
    if sv.showdst == 'yes':
        distshow(sv, cs)

    getslncv(sv, cs)
    getinscv(sv, cs)

    if sv.uvread == 'not':
        for pti in range(numslv):
            mask = cs.uvspec == pti
            sv.aveuv[pti] = np.sum(cs.uvcrd[mask] * cs.edist[mask])
        if sv.ljlrc == 'yes':
            ljcorrect(sv, cs.lj_state, cntrun)
    else:
        if prmcnt == 1 and sv.ljlrc == 'yes':
            ljcorrect(sv, cs.lj_state, cntrun)

    cumu_process = (sv.cumuint == 'yes' and group == sv.pickgr and inft == 0)
    if cumu_process and cntrun == 1:
        cs.cumsfe = np.zeros((gemax, sv.numrun + 1), dtype=np.float64)

    pos_soln = sv.rddst[sv.rddst > sv.zero]
    pos_refs = sv.rddns[sv.rddns > sv.zero]
    soln_zero = sv.minthres_soln * (pos_soln.min() if pos_soln.size else 0.0)
    refs_zero = sv.minthres_refs * (pos_refs.min() if pos_refs.size else 0.0)

    try:
        functional_fn = FUNCTIONAL_TABLE[sv.functional.lower()]
    except KeyError:
        raise SlvfeError("Incorrct functional")

    for pti in range(numslv):
        uvpot = 0.0
        slvfe = 0.0
        for iduv in range(gemax):
            if cs.uvspec[iduv] != pti:
                continue
            if cs.edist[iduv] <= soln_zero and cs.edens[iduv] <= refs_zero:
                if cumu_process:
                    cs.cumsfe[iduv, cntrun] = uvpot + slvfe
                continue

            uvpot += cs.uvcrd[iduv] * cs.edist[iduv]
            slvfe += -sv.kT * (cs.edist[iduv] - cs.edens[iduv])

            lcent = -(cs.slncv[iduv] + cs.zrsln[pti] + cs.uvcrd[iduv])
            if (sv.slncor == 'yes' and cs.edist[iduv] > soln_zero
                    and cs.edens[iduv] <= refs_zero):
                ampl = lcent * cs.edens[iduv] / cs.edist[iduv]
                lcent = ampl - (cs.zrsln[pti] + cs.uvcrd[iduv]) * (
                    1.0 - cs.edens[iduv] / cs.edist[iduv])
            slvfe += lcent * cs.edist[iduv]

            lcsln, lcref = functional_fn(sv, cs.slncv[iduv], cs.inscv[iduv])

            if (sv.slncor == 'yes' and cs.edist[iduv] > soln_zero
                    and cs.edens[iduv] <= refs_zero):
                lcsln = pyhnc(sv, cs.sdrcv[iduv] + cs.zrsdr[pti], PYHNC_INDIRECT)

            ampl = sfewgt(cs.edist[iduv], cs.edens[iduv])
            factor = ampl * lcsln + (1.0 - ampl) * lcref
            slvfe += sv.kT * factor * (cs.edist[iduv] - cs.edens[iduv])

            if cumu_process:
                cs.cumsfe[iduv, cntrun] = uvpot + slvfe

        sv.chmpt[pti + 1, prmcnt - 1, cntrun - 1] = slvfe + sv.aveuv[pti]

    if cumu_process and cntrun == sv.numrun:
        cs.cumsfe[:, 0] = cs.cumsfe[:, 1:sv.numrun + 1].mean(axis=1)
        _write_cumsfe(sv, cs)
        cs.cumsfe = None

    sv.chmpt[0, prmcnt - 1, cntrun - 1] = sv.chmpt[1:numslv + 1, prmcnt - 1, cntrun - 1].sum()
    if sv.slfslt == 'yes':
        sv.chmpt[0, prmcnt - 1, cntrun - 1] += sv.slfeng

    if sv.wrtzrsft == 'yes':
        print('  Zero shift for solution             = '
              + ''.join(f'{v:12.4f}' for v in cs.zrsln))
        print('  Zero shift for reference solvent    = '
              + ''.join(f'{v:12.4f}' for v in cs.zrref))
        if sv.slncor == 'yes':
            print('  Zero shift for solution correlation = '
                  + ''.join(f'{v:12.4f}' for v in cs.zrsdr))


def _write_cumsfe(sv: SysVars, cs: SfeCalcState) -> None:
    numslv = sv.numslv
    ge_perslv = cs.gemax // numslv
    for cntdiv in range(0, sv.numrun + 1):
        if sv.numrun == 1 and cntdiv != 0:
            continue
        opnfile = (Path(sv.cumuintfl) if cntdiv == 0
                   else Path(f"{sv.cumuintfl}{sv.get_suffix(cntdiv)}"))
        with open(opnfile, 'w') as f:
            if numslv == 1:
                for iduv in range(cs.gemax):
                    f.write(f"{cs.uvcrd[iduv]:15.5g}{cs.cumsfe[iduv, cntdiv]:12.5f}\n")
            else:
                homoform = np.all(cs.uvmax == ge_perslv)
                if homoform:
                    for iduv in range(ge_perslv):
                        coords = [cs.uvcrd[iduv + j * ge_perslv] for j in range(numslv)]
                        if not all(c == coords[0] for c in coords):
                            homoform = False
                            break
                if homoform:
                    for iduv in range(ge_perslv):
                        vals = [cs.cumsfe[iduv + j * ge_perslv, cntdiv] for j in range(numslv)]
                        total = sum(vals)
                        f.write(f"{cs.uvcrd[iduv]:15.5g}{total:12.5f}"
                                + ''.join(f'{v:12.5f}' for v in vals) + "\n")
                else:
                    for iduv in range(cs.gemax):
                        f.write(f"{cs.uvcrd[iduv]:15.5g}{cs.uvspec[iduv] + 1:5d}"
                                f"{cs.cumsfe[iduv, cntdiv]:12.5f}\n")
