# -*- coding: utf-8 -*-
"""
Port of the `sysread` module (slvfe.F90): reading the energy-distribution
files, correlation matrices, weights, and average solute-solvent energies.

NOTE on binary correlation-matrix files (corsln.*, corref.*):
    These were written by Fortran as a single unformatted record
    containing an (ermax, ermax) array of the compiled `real` kind
    (float64, since the code is compiled with a double-precision
    `real`). We read them with scipy.io.FortranFile, which understands
    the record-length markers Fortran uses. If your Fortran build used a
    non-default record-marker size (e.g. compiled with
    -assume byterecl / an unusual RECL convention), adjust
    `read_fortran_matrix` accordingly.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.io import FortranFile

from .config import SysVars
from .exceptions import SlvfeError


def read_fortran_matrix(path: str, n: int) -> np.ndarray:
    """Read a single (n, n) unformatted Fortran record into a NumPy array.

    Equivalent of:
        open(unit=72, file=path, form="UNFORMATTED")
        read(72) cormat_temp   ! cormat_temp(n, n)
    """
    with FortranFile(path, 'r') as f:
        flat = f.read_reals(dtype=np.float64)
    # Fortran arrays are column-major ('F' order).
    return flat.reshape((n, n), order='F')


def defcond(sv: SysVars) -> None:
    """Port of `defcond`: determine ermax, numslv, mesh structure, and
    the group/inft parameter grid; read aveuv / weights / self-energy."""

    if sv.clcond not in ('basic', 'range', 'merge'):
        raise SlvfeError(' The clcond parameter is incorrect')

    if sv.clcond in ('basic', 'range'):
        sv.engfile[0] = input(" What is the energy distribution in solution?\n")
        if sv.slncor == 'yes':
            sv.engfile[1] = input(" What is the energy correlation in solution?\n")
        sv.engfile[2] = input(" What is the energy density for insertion?\n")
        sv.engfile[3] = input(" What is the energy correlation for insertion?\n")
        if sv.infchk == 'yes' and sv.meshread == 'yes':
            sv.engfile[4] = input(" Which file has the meshes for energy coordinates?\n")
        sv.maxsln = 1
        sv.maxref = 1
        sv.numrun = 1
        if sv.clcond == 'basic':
            sv.prmmax = 1
        if sv.clcond == 'range':
            sv.prmmax = sv.numprm
    else:  # merge
        sv.maxsln = sv.numsln
        sv.maxref = sv.numref
        sv.numrun = sv.numdiv
        sv.prmmax = sv.numprm

    if sv.clcond == 'merge':
        opnfile = Path(sv.solndirec) / f"{sv.slndnspf}.{sv.get_suffix(1, sv.suffix_of_engsln_is_tt)}"
    else:
        opnfile = Path(sv.engfile[0])

    # --- first pass: count bins (ermax) and species (numslv) ---
    with open(opnfile) as f:
        next(f)  # header
        ermax = 0
        numslv = 0
        k = 0
        for line in f:
            if not line.strip():
                continue
            parts = line.split()
            pti = int(float(parts[2]))
            if pti != k:
                numslv += 1
                k = pti
            ermax += 1
    sv.ermax = ermax
    sv.numslv = numslv

    sv.nummol = np.zeros(numslv, dtype=np.float64)
    sv.rduvmax = np.zeros(numslv, dtype=np.int64)
    sv.rduvcore = np.zeros(numslv, dtype=np.int64)
    sv.rdcrd = np.zeros(ermax, dtype=np.float64)
    sv.rddst = np.zeros(ermax, dtype=np.float64)
    sv.rddns = np.zeros(ermax, dtype=np.float64)
    if sv.slncor == 'yes':
        sv.rdslc = np.zeros((ermax, ermax), dtype=np.float64)
    sv.rdcor = np.zeros((ermax, ermax), dtype=np.float64)
    sv.rdspec = np.zeros(ermax, dtype=np.int64)
    sv.chmpt = np.zeros((sv.numslv + 1, sv.prmmax, sv.numrun), dtype=np.float64)
    sv.aveuv = np.zeros(numslv, dtype=np.float64)
    if sv.uvread != 'not' and sv.clcond == 'merge':
        sv.uvene = np.zeros((numslv, sv.maxsln), dtype=np.float64)
        sv.blockuv = np.zeros((numslv + 1, sv.numrun), dtype=np.float64)
    sv.svgrp = np.zeros(sv.prmmax, dtype=np.int64)
    sv.svinf = np.zeros(sv.prmmax, dtype=np.int64)
    sv.wgtsln = np.zeros(sv.maxsln, dtype=np.float64)
    sv.wgtref = np.zeros(sv.maxref, dtype=np.float64)

    # --- second pass: mesh type (linear/logarithmic) per species ---
    with open(opnfile) as f:
        next(f)
        k = -1  # sentinel: no valid (0-based) species id equals -1
        crdprev = 0.0
        crddif_prev = 0.0
        for iduv in range(ermax):
            parts = f.readline().split()
            crdnow = float(parts[1])
            pti = int(float(parts[2])) - 1  # 0-based species id
            if pti != k:
                sv.rduvmax[pti] = 1
                sv.rduvcore[pti] = 0
                k = pti
            else:
                if sv.rduvmax[pti] < 1:
                    raise SlvfeError("Bug in counting rduvmax")
                sv.rduvmax[pti] += 1
                crddif_now = crdnow - crdprev
                if sv.rduvmax[pti] > 2:
                    if abs(crddif_now - crddif_prev) < sv.tiny:
                        sv.rduvcore[pti] = 0     # linear mesh
                    else:
                        sv.rduvcore[pti] += 1    # logarithmic mesh
                crddif_prev = crddif_now
            crdprev = crdnow

    if sv.infchk == 'yes' and sv.meshread == 'yes':
        if sv.clcond == 'merge':
            meshfile = Path(sv.solndirec) / sv.engmeshfile
        else:
            meshfile = Path(sv.engfile[4])
        for pti in range(numslv):
            with open(meshfile) as f:
                for line in f:
                    parts = line.split()
                    k = int(parts[0]) - 1
                    if k == pti:
                        sv.rduvcore[pti] = int(parts[2])
                        break

    if int(sv.rduvmax.sum()) != ermax:
        raise SlvfeError(' The file format is incorrect')
    if ermax > sv.ermax_limit:
        raise SlvfeError(' The number of energy bins is too large')

    # --- group / inft parameter grid ---
    if sv.clcond == 'basic':
        group = int(input(" How many data are grouped into one?\n"))
        inft = 0
        if sv.infchk == 'yes':
            inft = int(input(" How many large-energy meshes are merged ? (in %)\n"))
        sv.svgrp[0] = group
        sv.svinf[0] = inft
        sv.temp = float(input(" What is the temperature in Kelvin?\n"))
    else:  # range, merge
        for prmcnt in range(sv.prmmax):        # 0-based; Fortran prmcnt is 1-based
            pc1 = prmcnt + 1                     # 1-based counter, matches Fortran cases
            if sv.infchk == 'yes':
                table = {1: (1, 0), 2: (1, 60), 3: (1, 80), 4: (1, 100),
                         5: (2, 0), 6: (3, 0), 7: (4, 0), 8: (5, 0),
                         9: (5, 60), 10: (5, 80), 11: (5, 100), 12: (8, 0)}
                if pc1 in table:
                    group, inft = table[pc1]
                else:
                    group, inft = 10 + (pc1 - 13) * 5, 0
            else:
                inft = 0
                if pc1 <= 10:
                    group = pc1
                else:
                    group = 10 + (pc1 - 10) * 2
            sv.svgrp[prmcnt] = group
            sv.svinf[prmcnt] = inft
        sv.temp = sv.inptemp
    sv.kT = sv.temp * 8.314510e-3 / 4.184  # kcal/mol

    # --- average solute-solvent energy ---
    if sv.uvread != 'not':
        if sv.clcond in ('basic', 'range'):
            vals = input(" What is average solute-solvent energy in solution?\n").split()
            sv.aveuv[:numslv] = [float(v) for v in vals[:numslv]]
        else:  # merge
            opnfile2 = Path(sv.solndirec) / sv.aveuvfile
            if opnfile2.exists():
                with open(opnfile2) as f:
                    for i in range(sv.maxsln):
                        parts = f.readline().split()
                        sv.uvene[:numslv, i] = [float(v) for v in parts[1:1 + numslv]]
            else:
                sv.uvread = 'not'
                print(f" Warning: Although the uvread parameter was set to "
                      f"'yes', it is changed into 'not' since the {opnfile2} "
                      f"file was not found")

    # --- weight files ---
    sv.wgtsln[:sv.maxsln] = 1.0
    if sv.clcond == 'merge' and sv.readwgtfl == 'yes':
        opnfile2 = Path(sv.solndirec) / sv.wgtslnfl
        with open(opnfile2) as f:
            for i in range(sv.maxsln):
                parts = f.readline().split()
                sv.wgtsln[i] = float(parts[1])
    sv.wgtsln[:sv.maxsln] /= sv.wgtsln[:sv.maxsln].sum()

    sv.wgtref[:sv.maxref] = 1.0
    if sv.clcond == 'merge' and sv.readwgtfl == 'yes':
        opnfile2 = Path(sv.refsdirec) / sv.wgtreffl
        with open(opnfile2) as f:
            for i in range(sv.maxref):
                parts = f.readline().split()
                sv.wgtref[i] = float(parts[1])
    sv.wgtref[:sv.maxref] /= sv.wgtref[:sv.maxref].sum()

    # --- solute self-energy ---
    if sv.slfslt == 'yes':
        if sv.clcond in ('basic', 'range'):
            sv.slfeng = float(input(" What is the solute self-energy?\n"))
        else:  # merge
            if sv.readwgtfl == 'not':
                raise SlvfeError("readwgtfl needs to be yes when slfslt is yes")
            sv.slfeng = 0.0
            opnfile2 = Path(sv.refsdirec) / sv.wgtreffl
            if not opnfile2.exists():
                raise SlvfeError(" weight_refs is absent although slfslt is set to yes")
            with open(opnfile2) as f:
                for i in range(sv.maxref):
                    line = f.readline()
                    parts = line.split()
                    try:
                        factor = float(parts[2])
                    except (IndexError, ValueError):
                        sv.slfslt = 'not'
                        print(" Warning: Although the slfslt parameter was "
                              "set to 'yes', it is changed into 'not'. Maybe "
                              "the MD was done without periodic boundary "
                              "condition, with PME employed for the isolated "
                              "solute, or with Coulombic interaction in its "
                              "bare form.")
                        break
                    sv.slfeng += sv.wgtref[i] * factor


@dataclass
class _ReadJob:
    """One of the four data sources `datread` combines: energy
    distribution / correlation matrix, for the solution or the
    reference solvent. Replaces the Fortran `cnt` flag (1=engsln,
    2=corsln, 3=engref, 4=corref) driving a single `do cnt = 1, 4` loop
    -- a pattern the original Fortran source itself flagged as
    "spaghetti" and in need of a rewrite.
    """
    role: str              # 'engsln' | 'corsln' | 'engref' | 'corref'
    engfile_index: int     # index into sv.engfile, for clcond in ('basic', 'range')
    directory: str         # sv.solndirec / sv.refsdirec, for clcond == 'merge'
    prefix: str            # sv.slndnspf / sv.slncorpf / sv.refdnspf / sv.refcorpf
    suffix_is_tt: bool
    weights: np.ndarray    # sv.wgtsln or sv.wgtref (mutated in place: renormalized)
    ecmin: int
    ecmax: int


def _job_filename(sv: SysVars, job: _ReadJob, i: int) -> Path:
    if sv.clcond in ('basic', 'range'):
        return Path(sv.engfile[job.engfile_index])
    suffnum = sv.get_suffix(i + 1, job.suffix_is_tt)  # file suffixes are 1-based
    return Path(job.directory) / f"{job.prefix}.{suffnum}"


def _read_1d_distribution(sv: SysVars, job: _ReadJob, opnfile: Path, i: int,
                           bin_check: np.ndarray) -> None:
    """Reads one `engsln`/`engref` file and accumulates it into
    `sv.rddst`/`sv.rddns` (weighted by `job.weights[i]`)."""
    is_primary = job.role == 'engsln'   # engsln defines rdcrd/bin_check; engref checks them
    target = sv.rddst if is_primary else sv.rddns
    with open(opnfile) as f:
        next(f)  # header
        k = -1
        m = -1
        for iduv in range(sv.ermax):
            parts = f.readline().split()
            leftbin = float(parts[0])
            crdnow = float(parts[1])
            pti0 = int(float(parts[2])) - 1
            factor = float(parts[3])
            if is_primary:
                sv.rdcrd[iduv] = crdnow
                bin_check[iduv] = leftbin
            elif bin_check[iduv] != leftbin:
                raise SlvfeError(
                    "Solution and reference system energy "
                    "coordinates are inconsitent"
                )
            if pti0 != k:
                k = pti0
                m += 1
            target[iduv] += job.weights[i] * factor
            sv.rdspec[iduv] = m


def datread(sv: SysVars, cntrun: int) -> None:
    """Port of `datread`. ``cntrun`` is 1-based, matching the Fortran call
    convention used by sfemain / main.py."""

    numslv = sv.numslv
    ermax = sv.ermax

    if sv.clcond in ('basic', 'range'):
        slnini, slnfin = 0, 0
        refini, reffin = 0, 0
    else:  # merge; all ranges below are 0-based [ini, fin] inclusive
        if sv.maxsln >= sv.numrun:
            k = sv.maxsln // sv.numrun
            slnini = (cntrun - 1) * k
            slnfin = cntrun * k - 1
        else:
            slnini = (cntrun - 1) % sv.maxsln
            slnfin = slnini
        if sv.refmerge == 'not':
            if sv.maxref >= sv.numrun:
                m = sv.maxref // sv.numrun
                refini = (cntrun - 1) * m
                reffin = cntrun * m - 1
            else:
                refini = (cntrun - 1) % sv.maxref
                reffin = refini
        else:
            refini, reffin = 0, sv.maxref - 1

    sv.rddst[:] = 0.0
    if sv.slncor == 'yes':
        sv.rdslc[:, :] = 0.0
    if cntrun == 1 or sv.refmerge == 'not':
        sv.rddns[:] = 0.0
        sv.rdcor[:, :] = 0.0

    bin_check = np.zeros(ermax)

    jobs = [
        _ReadJob('engsln', 0, sv.solndirec, sv.slndnspf,
                 sv.suffix_of_engsln_is_tt, sv.wgtsln, slnini, slnfin),
        _ReadJob('corsln', 1, sv.solndirec, sv.slncorpf,
                 sv.suffix_of_engsln_is_tt, sv.wgtsln, slnini, slnfin),
        _ReadJob('engref', 2, sv.refsdirec, sv.refdnspf,
                 sv.suffix_of_engref_is_tt, sv.wgtref, refini, reffin),
        _ReadJob('corref', 3, sv.refsdirec, sv.refcorpf,
                 sv.suffix_of_engref_is_tt, sv.wgtref, refini, reffin),
    ]

    for job in jobs:
        if job.role == 'corsln' and sv.slncor != 'yes':
            continue
        if job.role in ('engref', 'corref') and cntrun > 1 and sv.refmerge == 'yes':
            continue

        job.weights[job.ecmin:job.ecmax + 1] /= job.weights[job.ecmin:job.ecmax + 1].sum()

        for i in range(job.ecmin, job.ecmax + 1):
            opnfile = _job_filename(sv, job, i)
            if job.role in ('engsln', 'engref'):
                _read_1d_distribution(sv, job, opnfile, i, bin_check)
            else:  # 2-D correlation matrix (unformatted binary)
                cormat_temp = read_fortran_matrix(str(opnfile), ermax)
                target = sv.rdslc if job.role == 'corsln' else sv.rdcor
                target += job.weights[i] * cormat_temp

    if cntrun == 1:
        print()
    for pti in range(numslv):
        mask = sv.rdspec == pti
        factor = sv.rddst[mask].sum()
        ampl = sv.rddns[mask].sum()
        num_different = abs(factor - ampl) > sv.tiny
        if cntrun > 1:
            if round(factor) != round(sv.nummol[pti]):
                num_different = True
            if round(ampl) != round(sv.nummol[pti]):
                num_different = True
        if num_different:
            raise SlvfeError(f'  Incorrect normalization at {pti + 1}')
        if cntrun == 1:
            sv.nummol[pti] = round(factor)
            print(f'  Number of the {pti + 1:3d}-th solvent  = {int(round(sv.nummol[pti])):12d}')
    if cntrun == 1:
        print()

    if sv.uvread != 'not' and sv.clcond == 'merge':
        for pti in range(numslv):
            sv.aveuv[pti] = np.sum(sv.wgtsln[slnini:slnfin + 1]
                                    * sv.uvene[pti, slnini:slnfin + 1])
        sv.blockuv[1:numslv + 1, cntrun - 1] = sv.aveuv[:numslv]
        sv.blockuv[0, cntrun - 1] = sv.blockuv[1:numslv + 1, cntrun - 1].sum()
        if sv.slfslt == 'yes':
            sv.blockuv[0, cntrun - 1] += sv.slfeng
