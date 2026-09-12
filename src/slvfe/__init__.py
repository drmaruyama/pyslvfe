# -*- coding: utf-8 -*-
"""
slvfe: Python port of ERmod's slvfe program (solvation free-energy
solver from the energy representation method).

Ported from the Fortran sources `slvfe.F90`, `sfemain.F90`, and
`sfecorrect.F90` in https://github.com/drmaruyama/ermod-openacc
(GPL-2.0-or-later; see LICENSE / NOTICE.md at the repository root).

The console command installed by this package is `pyslvfe` (not
`slvfe`), so it does not shadow the original Fortran `slvfe` binary on
$PATH.
"""
__version__ = "0.1.0"
