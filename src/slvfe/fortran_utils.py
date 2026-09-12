# -*- coding: utf-8 -*-
"""Small helpers to replicate Fortran intrinsic semantics exactly."""
from __future__ import annotations

import math


def nint(x: float) -> int:
    """Fortran NINT(): round to nearest integer, ties away from zero.

    (Python's built-in round() uses banker's rounding, which differs from
    Fortran's NINT on .5 ties, so we need this explicit replacement.)
    """
    if x >= 0:
        return math.floor(x + 0.5)
    return -math.floor(-x + 0.5)
