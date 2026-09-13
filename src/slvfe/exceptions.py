# -*- coding: utf-8 -*-
"""Exception types used throughout this package."""
from __future__ import annotations


class SlvfeError(Exception):
    """Raised for user-facing / data errors: bad or inconsistent input
    files, unsupported parameter combinations, numerical failures the
    code can't recover from, and so on.

    This replaces the many ``raise SystemExit(...)`` calls that were a
    direct translation of the Fortran originals' ``stop 'message'``
    statements. ``SystemExit`` is a ``BaseException``, not an
    ``Exception`` -- code that uses this package as a library and
    writes ``except Exception:`` would never catch a raised
    ``SystemExit``; it would just terminate the whole process instead.
    ``SlvfeError`` is a normal, catchable exception. The CLI entry
    point (``main.main()``) catches it at the top level and prints a
    clean one-line error message instead of a Python traceback, so
    command-line behavior is unchanged from before this refactor.
    """
