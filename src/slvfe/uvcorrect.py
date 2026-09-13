# -*- coding: utf-8 -*-
"""
Port of the `uvcorrect` module (sfecorrect.F90): the Lennard-Jones
long-range correction, applied only when ``ljlrc == 'yes'``.

Species indexing convention in this file:
    - `ljtype[:, 0]`      -> solute sites
    - `ljtype[:, pti + 1]` -> solvent species `pti` (0-based, matching
      the rest of the port), i.e. offset by one column vs. sfecalc.py's
      species index, to make room for the solute column.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from .config import SysVars
from .exceptions import SlvfeError
from .namelist_parser import read_namelist

SOLUTE_FILE = 'SltInfo'
SOLVENT_FILE = 'MolPrm'
LJTABLE_FILE = 'LJTable'
ENE_CONFNAME = 'parameters_er'

LJFMT_EPS_cal_SGM_nm, LJFMT_EPS_Rminh, LJFMT_EPS_J_SGM_A, \
    LJFMT_A_C, LJFMT_C12_C6, LJFMT_TABLE = range(6)
LJSWT_POT_CHM, LJSWT_POT_GMX, LJSWT_FRC_CHM, LJSWT_FRC_GMX = range(4)
LJCMB_ARITH, LJCMB_GEOM = range(2)

_SGMCNV = 1.7817974362806784        # Rmin/2 -> sigma, 2**(5/6)
_LENCNV = 10.0                        # nm -> Angstrom
_ENGCNV = 1.0 / 4.184                 # kJ/mol -> kcal/mol
_PI = math.pi


@dataclass
class LJState:
    first_time: bool = True
    ljcorr: Optional[np.ndarray] = None

    ljformat: int = LJFMT_EPS_Rminh
    ljswitch: int = LJSWT_POT_CHM
    cmbrule: int = LJCMB_ARITH
    lwljcut: float = 10.0
    upljcut: float = 12.0

    ptsite: Optional[np.ndarray] = None      # (numslv+1,) site counts
    ljtype_max: int = 0
    ljtype: Optional[np.ndarray] = None      # (maxsite, numslv+1)
    ljlensq_mat: Optional[np.ndarray] = None
    ljene_mat: Optional[np.ndarray] = None

    # enelj() switching parameters, computed once
    enelj_first_time: bool = True
    rbin: float = 1.0e-3
    numbin: int = 0
    do_swth: bool = False
    lwljcut2: float = 0.0
    upljcut2: float = 0.0
    lwljcut3: float = 0.0
    upljcut3: float = 0.0
    lwljcut6: float = 0.0
    upljcut6: float = 0.0
    repA: float = 0.0
    repB: float = 0.0
    repC: float = 0.0
    attA: float = 0.0
    attB: float = 0.0
    attC: float = 0.0


def ljcorrect(sv: SysVars, st: LJState, cntrun: int) -> None:
    """Port of `ljcorrect`. ``cntrun`` is 1-based.

    ``st`` holds the state that the Fortran original kept in ``SAVE``
    variables (computed once, on the first call, and reused after
    that). It is owned by the caller's `SfeCalcState` (see
    `sfecalc.SfeCalcState.lj_state`) rather than a module-level global,
    so that running this package's pipeline more than once in the same
    Python process (e.g. from a test suite, or a script that processes
    several independent systems in a loop) doesn't silently reuse a
    previous, unrelated run's LJ correction.
    """
    numslv = sv.numslv
    if st.first_time:
        _set_keyparam(sv, st)
        print(f"  Be sure that the solvent distribution is homogeneous "
              f"(radial distribution function is essentially unity) when "
              f"the solvent molecule is separated beyond distance of "
              f"{st.lwljcut:7.1f} Angstrom in any direction from any atom "
              f"within the solute molecule")
        print()
        _get_ljtable(sv, st)
        st.ljcorr = np.zeros(numslv, dtype=np.float64)
        for pti in range(numslv):
            st.ljcorr[pti] = _calc_ljlrc(sv, st, pti)
        print('  LJ long-range correction    =   '
              + ''.join(f'{v:12.4f}' for v in st.ljcorr))
        print()
        st.first_time = False

    sv.aveuv[:numslv] += st.ljcorr[:numslv]
    if sv.uvread != 'not' and sv.clcond == 'merge':
        sv.blockuv[1:numslv + 1, cntrun - 1] += st.ljcorr[:numslv]
        sv.blockuv[0, cntrun - 1] += st.ljcorr[:numslv].sum()


def _set_keyparam(sv: SysVars, st: LJState) -> None:
    volm_min = 1.40e4

    keyfile = Path(sv.refsdirec) / ENE_CONFNAME
    if not keyfile.exists():
        raise SlvfeError("The parameters_er file is not found in the refs directory")

    nml = read_namelist(str(keyfile))
    p = nml.get('ene_param', {})

    st.ljformat = p.get('ljformat', LJFMT_EPS_Rminh)
    st.ljswitch = p.get('ljswitch', LJSWT_POT_CHM)
    st.cmbrule = p.get('cmbrule', LJCMB_ARITH)
    st.upljcut = p.get('upljcut', 12.0)
    st.lwljcut = p.get('lwljcut', st.upljcut - 2.0)

    if sv.avevolume <= 0.0:
        sv.avevolume = float(input(
            "  What is the average volume of reference solvent? "
            "(in Angstrom^3)\n"))
    if sv.avevolume < volm_min:
        print("  Warning: your input volume seems too small")
        print(f"           This warning appears when your input is less "
              f"than {volm_min:8.1f}")
        print("  Re-type the volume in Angstrom^3 (NOT in nm^3)")
        sv.avevolume = float(input())


def _get_ljtable(sv: SysVars, st: LJState) -> None:
    numslv = sv.numslv
    refsdirec = Path(sv.refsdirec)

    # --- site counts ---
    ptsite = np.zeros(numslv + 1, dtype=np.int64)
    for pti in range(numslv + 1):
        molfile = (refsdirec / SOLUTE_FILE if pti == 0
                   else refsdirec / f"{SOLVENT_FILE}{pti}")
        with open(molfile) as f:
            ptsite[pti] = sum(1 for line in f if line.strip())
    st.ptsite = ptsite

    total_sites = int(ptsite.sum())
    ljlen_temp_table = np.zeros(total_sites, dtype=np.float64)
    ljene_temp_table = np.zeros(total_sites, dtype=np.float64)
    maxsite = int(ptsite.max())
    ljtype = np.zeros((maxsite, numslv + 1), dtype=np.int64)
    ljtype_max = 0

    for pti in range(numslv + 1):
        molfile = (refsdirec / SOLUTE_FILE if pti == 0
                   else refsdirec / f"{SOLVENT_FILE}{pti}")
        stmax = int(ptsite[pti])
        ljtype_temp = np.zeros(stmax, dtype=np.int64)
        ljlen_temp = np.zeros(stmax, dtype=np.float64)
        ljene_temp = np.zeros(stmax, dtype=np.float64)

        with open(molfile) as f:
            for sid in range(stmax):
                linebuf = f.readline()
                parts = linebuf.split()
                # New format: m, mass, atmtype, atmname, xst(1), xst(2), xst(3)  (7 tokens)
                # Old format: m, atmtype, xst(1), xst(2), xst(3)                 (5 tokens)
                # We need xst(2) (epsilon-like) and xst(3) (sigma-like).
                if len(parts) >= 7:
                    xst2, xst3 = float(parts[5]), float(parts[6])
                elif len(parts) >= 5:
                    xst2, xst3 = float(parts[3]), float(parts[4])
                else:
                    raise SlvfeError(f"Cannot parse molfile line: {linebuf!r}")

                if st.ljformat == LJFMT_EPS_Rminh:
                    xst3 = _SGMCNV * xst3
                if st.ljformat in (LJFMT_A_C, LJFMT_C12_C6):
                    if xst3 != 0.0:
                        factor = (xst2 / xst3) ** (1.0 / 6.0)
                        xst2 = xst3 / (4.0 * (factor ** 6))
                        xst3 = factor
                    else:
                        xst2 = 0.0
                if st.ljformat in (LJFMT_EPS_J_SGM_A, LJFMT_C12_C6):
                    xst2 = _ENGCNV * xst2
                    xst3 = _LENCNV * xst3

                ljene_temp[sid] = xst2
                ljlen_temp[sid] = xst3

        if st.ljformat == LJFMT_TABLE:
            ljtype_temp[:stmax] = ljene_temp[:stmax].astype(np.int64)
        else:
            for sid in range(stmax):
                lj_is_new = True
                ljtype_found = -1
                for i in range(ljtype_max):
                    if (ljlen_temp_table[i] == ljlen_temp[sid]
                            and ljene_temp_table[i] == ljene_temp[sid]):
                        ljtype_found = i
                        lj_is_new = False
                        break
                if lj_is_new:
                    ljlen_temp_table[ljtype_max] = ljlen_temp[sid]
                    ljene_temp_table[ljtype_max] = ljene_temp[sid]
                    ljtype_found = ljtype_max
                    ljtype_max += 1
                ljtype_temp[sid] = ljtype_found

        ljtype[:stmax, pti] = ljtype_temp

    st.ljtype_max = ljtype_max
    st.ljtype = ljtype

    # --- fill LJ table ---
    if st.ljformat == LJFMT_TABLE:
        tablefile = refsdirec / LJTABLE_FILE
        with open(tablefile) as f:
            ljtype_max = int(f.readline())
            st.ljtype_max = ljtype_max
            ljlensq_mat = np.zeros((ljtype_max, ljtype_max), dtype=np.float64)
            ljene_mat = np.zeros((ljtype_max, ljtype_max), dtype=np.float64)
            for i in range(ljtype_max):
                row = [float(v) for v in f.readline().split()]
                ljlensq_mat[i, :] = np.array(row) ** 2
            for i in range(ljtype_max):
                row = [float(v) for v in f.readline().split()]
                ljene_mat[i, :] = row
        st.ljlensq_mat = ljlensq_mat
        st.ljene_mat = ljene_mat
    else:
        n = ljtype_max
        ljlensq_mat = np.zeros((n, n), dtype=np.float64)
        ljene_mat = np.zeros((n, n), dtype=np.float64)
        lens = ljlen_temp_table[:n]
        engs = ljene_temp_table[:n]
        for i in range(n):
            if st.cmbrule == LJCMB_ARITH:
                ljlensq_mat[:, i] = ((lens + lens[i]) / 2.0) ** 2
            elif st.cmbrule == LJCMB_GEOM:
                ljlensq_mat[:, i] = lens * lens[i]
            else:
                raise SlvfeError("Incorrect cmbrule")
            ljene_mat[:, i] = np.sqrt(engs * engs[i])
        st.ljlensq_mat = ljlensq_mat
        st.ljene_mat = ljene_mat


def _calc_ljlrc(sv: SysVars, st: LJState, pti: int) -> float:
    """``pti`` is 0-based (solvent species index)."""
    dens = sv.nummol[pti] / sv.avevolume
    correction = 0.0
    for ui in range(st.ptsite[0]):
        for vi in range(st.ptsite[pti + 1]):
            ljeps = st.ljene_mat[st.ljtype[ui, 0], st.ljtype[vi, pti + 1]]
            ljsgm2 = st.ljlensq_mat[st.ljtype[ui, 0], st.ljtype[vi, pti + 1]]
            correction += dens * _enelj(st, ljeps, ljsgm2)
    return correction


def _enelj(st: LJState, ljeps: float, ljsgm2: float) -> float:
    if st.enelj_first_time:
        if st.lwljcut > st.upljcut:
            raise SlvfeError(
                "Incorrect setting of lwljcut and upljcut (lwljcut > upljcut)")
        st.numbin = round((st.upljcut - st.lwljcut) / st.rbin)
        if st.numbin >= 1:
            st.do_swth = True
            st.rbin = (st.upljcut - st.lwljcut) / st.numbin
            st.lwljcut2 = st.lwljcut ** 2
            st.upljcut2 = st.upljcut ** 2
            if st.ljswitch == LJSWT_FRC_CHM:
                st.lwljcut3 = st.lwljcut ** 3
                st.upljcut3 = st.upljcut ** 3
                st.lwljcut6 = st.lwljcut3 * st.lwljcut3
                st.upljcut6 = st.upljcut3 * st.upljcut3
            if st.ljswitch == LJSWT_FRC_GMX:
                st.repA, st.repB, st.repC = _calc_gmx_switching_force_params(
                    12, st.lwljcut, st.upljcut)
                st.attA, st.attB, st.attC = _calc_gmx_switching_force_params(
                    6, st.lwljcut, st.upljcut)
        else:
            st.do_swth = False
        st.enelj_first_time = False

    ljsgm6 = ljsgm2 ** 3
    ljsgm3 = math.sqrt(ljsgm6)
    ljint = 0.0

    # r < lwljcut
    if st.ljswitch in (LJSWT_POT_CHM, LJSWT_POT_GMX):
        pass
    elif st.ljswitch == LJSWT_FRC_CHM:
        vdwa = ljsgm6 * ljsgm6 / (st.lwljcut6 * st.upljcut6)
        vdwb = ljsgm6 / (st.lwljcut3 * st.upljcut3)
        edev = 4.0 * ljeps * (vdwa - vdwb)
        ljint += (4.0 * _PI / 3.0) * st.lwljcut3 * edev
    elif st.ljswitch == LJSWT_FRC_GMX:
        vdwa = ljsgm6 * ljsgm6 * st.repC
        vdwb = ljsgm6 * st.attC
        edev = 4.0 * ljeps * (vdwa - vdwb)
        ljint += (4.0 * _PI / 3.0) * st.lwljcut3 * edev
    else:
        raise SlvfeError("Unknown ljswitch")

    # lwljcut < r < upljcut
    if st.do_swth:
        for i in range(1, st.numbin + 1):
            r = st.lwljcut + (i - 0.5) * st.rbin
            dist = r * r
            invr2 = ljsgm2 / dist
            invr6 = invr2 ** 3
            if st.ljswitch == LJSWT_POT_CHM:
                swth = ((2.0 * dist + st.upljcut2 - 3.0 * st.lwljcut2)
                        * ((dist - st.upljcut2) ** 2)
                        / ((st.upljcut2 - st.lwljcut2) ** 3))
                edev = 4.0 * ljeps * invr6 * (invr6 - 1.0) * (1.0 - swth)
            elif st.ljswitch == LJSWT_POT_GMX:
                swfac = (r - st.lwljcut) / (st.upljcut - st.lwljcut)
                swth = (1.0 - 10.0 * swfac ** 3 + 15.0 * swfac ** 4
                        - 6.0 * swfac ** 5)
                edev = 4.0 * ljeps * invr6 * (invr6 - 1.0) * (1.0 - swth)
            elif st.ljswitch == LJSWT_FRC_CHM:
                invr3 = math.sqrt(invr6)
                vdwa = (st.upljcut6 / (st.upljcut6 - st.lwljcut6)
                        * (invr6 - ljsgm6 / st.upljcut6) ** 2)
                vdwb = (st.upljcut3 / (st.upljcut3 - st.lwljcut3)
                        * (invr3 - ljsgm3 / st.upljcut3) ** 2)
                edev = 4.0 * ljeps * (invr6 * (invr6 - 1.0) - (vdwa - vdwb))
            elif st.ljswitch == LJSWT_FRC_GMX:
                swfac = r - st.lwljcut
                vdwa = ljsgm6 * ljsgm6 * (st.repA * swfac ** 3
                                           + st.repB * swfac ** 4 + st.repC)
                vdwb = ljsgm6 * (st.attA * swfac ** 3 + st.attB * swfac ** 4
                                  + st.attC)
                edev = 4.0 * ljeps * (vdwa - vdwb)
            else:
                raise SlvfeError("Unknown ljswitch")
            ljint += 4.0 * _PI * r * r * st.rbin * edev

    # r > upljcut
    invr3 = ljsgm3 / (st.upljcut ** 3)
    ljint += ljeps * ljsgm3 * (16.0 * _PI / 3.0) * ((invr3 ** 3 / 3.0) - invr3)

    return ljint


def _calc_gmx_switching_force_params(pow_: int, lwljcut: float, upljcut: float):
    dfljcut = upljcut - lwljcut
    coeffA = (-pow_ * ((pow_ + 4) * upljcut - (pow_ + 1) * lwljcut)
              / ((upljcut ** (pow_ + 2)) * (dfljcut ** 2)) / 3.0)
    coeffB = (pow_ * ((pow_ + 3) * upljcut - (pow_ + 1) * lwljcut)
              / ((upljcut ** (pow_ + 2)) * (dfljcut ** 3)) / 4.0)
    coeffC = (1.0 / (upljcut ** pow_) - coeffA * (dfljcut ** 3)
              - coeffB * (dfljcut ** 4))
    return coeffA, coeffB, coeffC
