# -*- coding: utf-8 -*-
"""Integration test for reader.py (defcond/datread), writing small
synthetic engsln/engref/weight files to a temp directory and checking
the resulting arrays. This is the only test that actually exercises
disk I/O in reader.py (test_smoke.py bypasses it entirely by setting
sv.rddst/rddns/... directly)."""
import tempfile
from pathlib import Path

import numpy as np

from slvfe.config import SysVars
from slvfe.reader import datread, defcond
from scipy.io import FortranFile


def _write_engfile(path, rows):
    with open(path, 'w') as f:
        f.write("# header\n")
        for leftbin, crd, pti, val in rows:
            f.write(f"{leftbin} {crd} {pti} {val}\n")


def _write_fortran_matrix(path, mat):
    with FortranFile(path, 'w') as f:
        f.write_record(np.asarray(mat, dtype=np.float64, order='F'))


def test_datread_merge_single_species():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        soln = tmp / 'soln'
        refs = tmp / 'refs'
        soln.mkdir()
        refs.mkdir()

        # 4 bins, single species (species id 1, 1-based as in the file format)
        _write_engfile(soln / 'engsln.01',
                       [(-2, -2.0, 1, 1.0), (-1, -1.0, 1, 2.0),
                        (0, 0.0, 1, 3.0), (1, 1.0, 1, 4.0)])
        _write_engfile(soln / 'engsln.02',
                       [(-2, -2.0, 1, 5.0), (-1, -1.0, 1, 6.0),
                        (0, 0.0, 1, 7.0), (1, 1.0, 1, 8.0)])
        _write_engfile(refs / 'engref.01',
                       [(-2, -2.0, 1, 4.0), (-1, -1.0, 1, 4.0),
                        (0, 0.0, 1, 5.0), (1, 1.0, 1, 5.0)])

        with open(soln / 'weight_soln', 'w') as f:
            f.write("1 1.0\n2 1.0\n")
        with open(refs / 'weight_refs', 'w') as f:
            f.write("1 1.0\n")

        corref_mat = np.arange(16, dtype=np.float64).reshape(4, 4)
        corref_mat = corref_mat + corref_mat.T  # symmetric, for realism
        _write_fortran_matrix(refs / 'corref.01', corref_mat)

        sv = SysVars()
        sv.clcond = 'merge'
        sv.numslv = 1
        sv.ermax = 4
        sv.solndirec = str(soln)
        sv.refsdirec = str(refs)
        sv.slndnspf = 'engsln'
        sv.refdnspf = 'engref'
        sv.slncorpf = 'corsln'
        sv.refcorpf = 'corref'
        sv.slncor = False
        sv.uvread = False
        sv.refmerge = True
        sv.maxsln = 2
        sv.maxref = 1
        sv.numrun = 1
        sv.suffix_of_engsln_is_tt = False
        sv.suffix_of_engref_is_tt = False
        sv.tiny = 1.0e-8
        sv.wgtsln = np.array([1.0, 1.0])
        sv.wgtref = np.array([1.0])
        sv.rdcrd = np.zeros(4)
        sv.rddst = np.zeros(4)
        sv.rddns = np.zeros(4)
        sv.rdcor = np.zeros((4, 4))
        sv.rdspec = np.zeros(4, dtype=np.int64)
        sv.nummol = np.zeros(1)

        datread(sv, cntrun=1)

        # rddst = average of the two weighted engsln files (equal weights)
        np.testing.assert_allclose(sv.rddst, [3.0, 4.0, 5.0, 6.0])
        np.testing.assert_allclose(sv.rddns, [4.0, 4.0, 5.0, 5.0])
        np.testing.assert_allclose(sv.rdcrd, [-2.0, -1.0, 0.0, 1.0])
        np.testing.assert_allclose(sv.rdcor, corref_mat)  # single ref file, weight 1.0
        assert list(sv.rdspec) == [0, 0, 0, 0]
        # nummol(species 1) set from the normalization check (sum of rddst)
        assert sv.nummol[0] == 18  # 3+4+5+6
        print("test_datread_merge_single_species OK")


def test_defcond_merge_two_species():
    """Exercises `defcond` end-to-end (bin/species counting, mesh-type
    detection, the group/inft grid, and weight-file reading), which
    `test_datread_merge_single_species` above does not touch (it sets
    up `sv` by hand and calls `datread` directly)."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        soln = tmp / 'soln'
        refs = tmp / 'refs'
        soln.mkdir()
        refs.mkdir()

        # 2 species, 4 (linearly spaced) bins each -> ermax = 8
        rows = (
            [(-2, -1.5, 1, 1.0), (-1, -0.5, 1, 2.0),
             (0, 0.5, 1, 3.0), (1, 1.5, 1, 4.0)]
            + [(-2, -2.0, 2, 1.0), (-1, -1.0, 2, 2.0),
               (0, 0.0, 2, 3.0), (1, 1.0, 2, 4.0)]
        )
        _write_engfile(soln / 'engsln.01', rows)

        with open(soln / 'weight_soln', 'w') as f:
            f.write("1 1.0\n")
        with open(refs / 'weight_refs', 'w') as f:
            f.write("1 1.0\n")

        sv = SysVars()
        sv.clcond = 'merge'
        sv.solndirec = str(soln)
        sv.refsdirec = str(refs)
        sv.slndnspf = 'engsln'
        sv.slncor = False
        sv.uvread = False
        sv.slfslt = False
        sv.infchk = False
        sv.meshread = False
        sv.readwgtfl = True
        sv.refmerge = True
        sv.suffix_of_engsln_is_tt = False
        sv.suffix_of_engref_is_tt = False
        sv.numsln = 1
        sv.numref = 1
        sv.numdiv = 1
        sv.numprm = 3
        sv.inptemp = 310.0
        sv.pickgr = 1

        defcond(sv)

        assert sv.ermax == 8
        assert sv.numslv == 2
        assert list(sv.rduvmax) == [4, 4]
        assert list(sv.rduvcore) == [0, 0]   # both meshes are linear
        assert list(sv.svgrp) == [1, 2, 3]   # infchk is False: group == pc1 (<=10)
        assert list(sv.svinf) == [0, 0, 0]
        assert sv.temp == 310.0
        assert abs(sv.kT - 310.0 * 8.314510e-3 / 4.184) < 1e-12
        np.testing.assert_allclose(sv.wgtsln, [1.0])   # single file -> weight 1.0
        np.testing.assert_allclose(sv.wgtref, [1.0])
        print("test_defcond_merge_two_species OK")


if __name__ == '__main__':
    test_datread_merge_single_species()
    test_defcond_merge_two_species()
    print("ALL OK")
