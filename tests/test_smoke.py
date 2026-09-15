# -*- coding: utf-8 -*-
"""Smoke test with synthetic data, bypassing namelist/file I/O, to sanity
check the numerical core (grouping, getslncv, getinscv/solver, chmpot)
without needing real ERmod input directories or the f90nml dependency."""
import numpy as np

from slvfe.config import SysVars
from slvfe.sfecalc import SfeCalcState, chmpot
from slvfe.output import OutputState, wrtresl


def build_synthetic(numslv=1, ermax=6, slncor=False):
    sv = SysVars()
    sv.numslv = numslv
    sv.ermax = ermax
    sv.clcond = 'basic'
    sv.slncor = slncor
    sv.uvread = False
    sv.slfslt = False
    sv.ljlrc = False
    sv.infchk = False
    sv.normalize = False
    sv.showdst = False
    sv.cumuint = False
    sv.invmtrx = 'reg'
    sv.zerosft = 'eczr'
    sv.functional = 'pyhnc'
    sv.wgtfnform = 'harm'
    sv.wgtf2smpl = True
    sv.extsln = 'lin'
    sv.temp = 300.0
    sv.kT = sv.temp * 8.314510e-3 / 4.184
    sv.numrun = 1
    sv.prmmax = 1
    sv.pickgr = 1
    sv.msemin, sv.msemax = 1, 5
    sv.svgrp = np.array([1])
    sv.svinf = np.array([0])
    sv.rduvmax = np.array([ermax])
    sv.rduvcore = np.array([0])
    sv.nummol = np.array([10.0])
    sv.rdspec = np.zeros(ermax, dtype=np.int64)

    sv.rdcrd = np.array([-2.0, -1.0, -0.5, 0.5, 1.0, 2.0])
    sv.rddst = np.array([0.10, 0.50, 1.00, 1.00, 0.50, 0.10])
    sv.rddns = np.array([0.05, 0.40, 0.90, 0.95, 0.45, 0.08])

    rng = np.random.default_rng(0)
    if slncor:
        A = rng.normal(size=(ermax, ermax))
        sv.rdslc = np.diag(sv.rddst) + 0.01 * (A + A.T)
    A2 = rng.normal(size=(ermax, ermax))
    sv.rdcor = np.diag(sv.rddns) + 0.01 * (A2 + A2.T)

    sv.aveuv = np.zeros(numslv)
    sv.chmpt = np.zeros((numslv + 1, sv.prmmax, sv.numrun))
    sv.slfeng = 0.0
    sv.tiny = 1.0e-8
    sv.zero = 0.0
    sv.extthres_soln = 1
    sv.extthres_refs = 1
    sv.minthres_soln = 0
    sv.minthres_refs = 0
    sv.norm_error = 1.0e-8
    sv.itrmax = 100
    sv.et = 0.0
    sv.wrtzrsft = False
    sv.write_mesherror = 'not'
    sv.mesherr = 0.1
    return sv


def run(slncor):
    print(f"--- slncor = {slncor!r} ---")
    sv = build_synthetic(slncor=slncor)
    cs = SfeCalcState()
    chmpot(sv, cs, prmcnt=1, cntrun=1)
    print("gemax =", cs.gemax)
    print("slncv =", cs.slncv)
    print("inscv =", cs.inscv)
    if slncor:
        print("sdrcv =", cs.sdrcv)
    print("chmpt (per-species, total) =", sv.chmpt[:, 0, 0])

    ost = OutputState()
    wrtresl(sv, ost)
    assert np.all(np.isfinite(sv.chmpt))
    print("OK\n")


def build_synthetic_merge(numslv=2, numrun=2, prmmax=2):
    """A second scenario (clcond='merge', numslv>1, numrun>1) to exercise
    `wrtmerge`/`wrtcumu`, which the single-run 'basic' scenario above
    never reaches."""
    ermax = 8  # 4 bins per species (kept >1 bin/species after grouping,
    # so `zeroec` can always find a sign change -- see its docstring)
    sv = SysVars()
    sv.numslv = numslv
    sv.ermax = ermax
    sv.clcond = 'merge'
    sv.slncor = False
    sv.uvread = False
    sv.slfslt = False
    sv.ljlrc = False
    sv.infchk = False
    sv.normalize = False
    sv.showdst = False
    sv.cumuint = False
    sv.invmtrx = 'reg'
    sv.zerosft = 'eczr'
    sv.functional = 'pyhnc'
    sv.wgtfnform = 'harm'
    sv.wgtf2smpl = True
    sv.extsln = 'lin'
    sv.temp = 300.0
    sv.kT = sv.temp * 8.314510e-3 / 4.184
    sv.numrun = numrun
    sv.prmmax = prmmax
    sv.pickgr = 1
    sv.msemin, sv.msemax = 1, 5
    sv.svgrp = np.array([1, 2])
    sv.svinf = np.array([0, 0])
    sv.rduvmax = np.array([4, 4])
    sv.rduvcore = np.array([0, 0])
    sv.nummol = np.array([10.0, 8.0])
    sv.rdspec = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int64)

    sv.rdcrd = np.array([-1.5, -0.5, 0.5, 1.5, -1.5, -0.5, 0.5, 1.5])
    sv.rddst = np.array([0.10, 0.20, 1.00, 0.20, 0.08, 0.15, 0.90, 0.15])
    sv.rddns = np.array([0.05, 0.10, 0.80, 0.10, 0.04, 0.08, 0.70, 0.08])

    rng = np.random.default_rng(1)
    A2 = rng.normal(size=(ermax, ermax))
    sv.rdcor = np.diag(sv.rddns) + 0.01 * (A2 + A2.T)

    sv.aveuv = np.zeros(numslv)
    sv.chmpt = np.zeros((numslv + 1, prmmax, numrun))
    sv.slfeng = 0.0
    sv.tiny = 1.0e-8
    sv.zero = 0.0
    sv.extthres_soln = 1
    sv.extthres_refs = 1
    sv.minthres_soln = 0
    sv.minthres_refs = 0
    sv.norm_error = 1.0e-8
    sv.itrmax = 100
    sv.et = 0.0
    sv.wrtzrsft = False
    sv.write_mesherror = 'not'
    sv.mesherr = 0.1
    return sv


def run_merge():
    print("--- clcond = 'merge', numslv=2, numrun=2, prmmax=2 ---")
    sv = build_synthetic_merge()
    cs = SfeCalcState()
    for cntrun in range(1, sv.numrun + 1):
        for prmcnt in range(1, sv.prmmax + 1):
            chmpot(sv, cs, prmcnt=prmcnt, cntrun=cntrun)
    print("chmpt =\n", sv.chmpt)

    ost = OutputState()
    wrtresl(sv, ost)  # exercises wrtmerge() and wrtcumu()
    assert np.all(np.isfinite(sv.chmpt))
    print("OK\n")


if __name__ == '__main__':
    run(False)
    run(True)
    run_merge()
