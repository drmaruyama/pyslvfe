# -*- coding: utf-8 -*-
"""Integration test for uvcorrect.py (the LJ long-range correction),
which previously had no test coverage at all (ljlrc is False in every
other test). Writes small synthetic SltInfo/MolPrm1/parameters_er
files to a temp directory."""
import math
import tempfile
from pathlib import Path

import numpy as np

from slvfe.config import SysVars
from slvfe.sfecalc import SfeCalcState
from slvfe import uvcorrect
from slvfe.uvcorrect import LJState, ljcorrect, _get_ljtable, _set_keyparam, _calc_ljlrc, _enelj


def _write_molfile_old_format(path, rows):
    """`rows`: list of (m, atmtype, charge, eps, sigma)."""
    with open(path, 'w') as f:
        for row in rows:
            f.write(' '.join(str(v) for v in row) + '\n')


def test_get_ljtable_and_calc_ljlrc():
    with tempfile.TemporaryDirectory() as tmp:
        refs = Path(tmp) / 'refs'
        refs.mkdir()

        # ljformat = 0 (LJFMT_EPS_cal_SGM_nm): no unit conversion, so the
        # eps/sigma columns below are used as-is -- easiest to hand-check.
        _write_molfile_old_format(refs / 'SltInfo', [(1, 'CT', 0.0, 0.10, 3.5)])
        _write_molfile_old_format(refs / 'MolPrm1', [(1, 'OW', 0.0, 0.15, 3.2)])

        with open(refs / 'parameters_er', 'w') as f:
            f.write("&ene_param\n"
                    "  ljformat = 0\n"
                    "  ljswitch = 0\n"
                    "  cmbrule = 0\n"
                    "  lwljcut = 8.0\n"
                    "  upljcut = 10.0\n"
                    "/\n")

        sv = SysVars()
        sv.numslv = 1
        sv.refsdirec = str(refs)
        sv.nummol = np.array([10.0])
        sv.avevolume = 20000.0   # > volm_min, and > 0 so _set_keyparam doesn't prompt

        st = LJState()
        _set_keyparam(sv, st)
        assert st.ljformat == 0
        assert st.lwljcut == 8.0
        assert st.upljcut == 10.0

        _get_ljtable(sv, st)

        # one LJ type per distinct (eps, sigma) pair -> 2 types here
        assert list(st.ptsite) == [1, 1]
        assert st.ljtype_max == 2
        solute_type = st.ljtype[0, 0]
        solvent_type = st.ljtype[0, 1]
        assert solute_type != solvent_type

        expected_sigma = (3.5 + 3.2) / 2.0   # arithmetic combining rule
        expected_eps = math.sqrt(0.10 * 0.15)   # geometric combining rule
        np.testing.assert_allclose(
            st.ljlensq_mat[solute_type, solvent_type], expected_sigma ** 2)
        np.testing.assert_allclose(
            st.ljene_mat[solute_type, solvent_type], expected_eps)

        # _calc_ljlrc should just be density * the single-pair enelj() integral
        correction = _calc_ljlrc(sv, st, 0)
        dens = sv.nummol[0] / sv.avevolume
        expected_integral = _enelj(st, expected_eps, expected_sigma ** 2)
        np.testing.assert_allclose(correction, dens * expected_integral)
        assert math.isfinite(correction)
        print("test_get_ljtable_and_calc_ljlrc OK "
              f"(correction={correction:.6g} kcal/mol)")


def test_ljcorrect_updates_aveuv_once():
    """`ljcorrect` should only recompute the correction on the first
    call (`st.first_time`), and just re-apply the cached value on
    subsequent calls -- and each `SfeCalcState` should get its own
    `LJState`, not a stale one from a previous, unrelated run (see the
    `lj_state` docstring in sfecalc.py)."""
    with tempfile.TemporaryDirectory() as tmp:
        refs = Path(tmp) / 'refs'
        refs.mkdir()
        _write_molfile_old_format(refs / 'SltInfo', [(1, 'CT', 0.0, 0.10, 3.5)])
        _write_molfile_old_format(refs / 'MolPrm1', [(1, 'OW', 0.0, 0.15, 3.2)])
        with open(refs / 'parameters_er', 'w') as f:
            f.write("&ene_param\n  ljformat = 0\n  lwljcut = 8.0\n"
                    "  upljcut = 10.0\n/\n")

        sv = SysVars()
        sv.numslv = 1
        sv.refsdirec = str(refs)
        sv.nummol = np.array([10.0])
        sv.avevolume = 20000.0
        sv.aveuv = np.array([1.0])
        sv.uvread = False   # so ljcorrect doesn't also touch sv.blockuv

        cs = SfeCalcState()
        ljcorrect(sv, cs.lj_state, cntrun=1)
        first_correction = cs.lj_state.ljcorr.copy()
        assert not cs.lj_state.first_time
        np.testing.assert_allclose(sv.aveuv, 1.0 + first_correction)

        # second call: aveuv accumulates the *same* cached correction again
        ljcorrect(sv, cs.lj_state, cntrun=1)
        np.testing.assert_allclose(sv.aveuv, 1.0 + 2 * first_correction)

        # a second, independent SfeCalcState must not see the first one's
        # cached LJState
        cs2 = SfeCalcState()
        assert cs2.lj_state.first_time
        print("test_ljcorrect_updates_aveuv_once OK")


if __name__ == '__main__':
    test_get_ljtable_and_calc_ljlrc()
    test_ljcorrect_updates_aveuv_once()
    print("ALL OK")
