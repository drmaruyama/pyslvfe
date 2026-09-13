# pyslvfe — Python port of ERmod's `slvfe`

A Python port of `ermod-openacc/slvfe` (`slvfe.F90`, `sfemain.F90`,
`sfecorrect.F90`). The GPU-dependent solvers (cuSolverDn) have been
replaced with CPU SciPy/LAPACK calls.

## Requirements

- Python >= 3.8
- `numpy`, `scipy` (installed automatically by `pip install`, or via
  `pip install numpy scipy` if you're running the script directly
  without installing the package)

No external namelist-parsing package is required — see
["Namelist parsing"](#namelist-parsing-parameters_fe--parameters_er)
below.

## Installation

```bash
git clone <this repository>
cd pyslvfe
pip install .          # or `pip install -e .` for an editable install
```

This installs the `pyslvfe` command (see below). The console command is
named `pyslvfe`, **not** `slvfe`, so it doesn't shadow the original
Fortran `slvfe` binary on `$PATH`.

If you'd rather not install anything, you can also just install the
two dependencies and run the top-level script directly (see "Running
it without installing" below):

```bash
pip install numpy scipy
```

## Usage

```bash
cd <directory containing parameters_fe, soln/, refs/>
pyslvfe
```

As in the original Fortran program, when `clcond = 'basic'` or
`'range'` (in `parameters_fe`), you will be prompted for a few values
via standard input. `clcond = 'merge'` is the main mode intended for
automated/batch processing and does not prompt for input.

### Running it without installing

```bash
cd <directory containing parameters_fe, soln/, refs/>
python3 /path/to/slvfe.py
```

`slvfe.py` at the repository root is a thin wrapper that runs the exact
same code as the `pyslvfe` command, without requiring `pip install`
first.

### Other ways to run it

```bash
python -m slvfe          # equivalent to the pyslvfe / slvfe.py above
```

```python
# using it as a library from your own script, after `pip install`:
from slvfe.config import SysVars
from slvfe.solver import posv_wrap, syevr_wrap
```

## Namelist parsing (`parameters_fe` / `parameters_er`)

Parsing is handled by the small, self-contained parser in
`src/slvfe/namelist_parser.py` — no external namelist package is
required. It covers what these two files actually use: quoted strings,
logicals, integers/reals (including Fortran's `d`/`D` exponent marker),
simple comma-separated arrays, the `n*value` repeat shorthand, and `!`
comments. It does not implement the full Fortran namelist standard
(e.g. `arr(2:4) = ...` slice assignment or the legacy
`$group ... $end` delimiter style), but this is not needed by
`parameters_fe` / `parameters_er`. See `tests/test_namelist_parser.py`
for example input/output.

## On precision

The original Fortran code was compiled with a flag (e.g. `-r8`) that
promotes the default `real` kind to double precision. NumPy's default
floating-point dtype is already `float64` (double precision), so this
is reproduced automatically without any special handling.

## cuSolverDn → CPU solver mapping

| Fortran (`slvfe.F90`) | cuSolverDn API used | Replacement |
|---|---|---|
| `posv_wrap` | `cusolverDnDpotrf` + `cusolverDnDpotrs` | `scipy.linalg.cho_factor` + `cho_solve` (Cholesky-based linear solve) |
| `syevr_wrap` | `cusolverDnDsyevd` | `scipy.linalg.eigh` (symmetric eigenvalue/eigenvector decomposition) |

Both are exactly the LAPACK algorithms (`DPOTRF`/`DPOTRS`, `DSYEVD`)
that cuSolverDn wraps on the GPU, so there is no GPU-specific behavior
to reproduce. The meaning of `info` (0 = success, non-zero = failure →
caller falls back to the EVD-based solver) is preserved as-is.

## Running the tests

```bash
pip install -e ".[dev]"
pytest
```

or, without pytest, each test file can be run directly:

```bash
python3 tests/test_smoke.py
python3 tests/test_reader.py
python3 tests/test_namelist_parser.py
```

## Repository layout

```
pyslvfe/
├── pyproject.toml       # `pip install .` metadata; installs the `pyslvfe` command
├── slvfe.py             # thin wrapper: run without installing (see "Usage" above)
├── src/slvfe/           # the actual package (import slvfe / python -m slvfe)
│   ├── __init__.py
│   ├── __main__.py       # supports `python -m slvfe`
│   ├── main.py           # entry point logic (`def main()`), port of `program sfemain`
│   ├── config.py         # port of `sysvars` (sfemain.F90)
│   ├── reader.py         # port of `sysread` (slvfe.F90): defcond, datread
│   ├── sfecalc.py        # port of `sfecalc` (slvfe.F90): the numerical core
│   ├── solver.py         # cuSolverDn → SciPy/LAPACK replacement (posv_wrap, syevr_wrap)
│   ├── uvcorrect.py      # port of `uvcorrect` (sfecorrect.F90): LJ long-range correction
│   ├── output.py         # port of `opwrite` (slvfe.F90): result printing
│   ├── namelist_parser.py  # self-contained Fortran namelist parser
│   ├── fortran_utils.py    # Fortran-intrinsic compatibility helpers (e.g. NINT)
│   └── exceptions.py       # SlvfeError (see "Error handling" above)
└── tests/
    ├── test_smoke.py            # synthetic-data sanity check (no real input files needed)
    ├── test_reader.py           # integration test for reader.py (writes temp input files)
    └── test_namelist_parser.py  # unit tests for namelist_parser.py
```

| File | Original Fortran module | Contents |
|---|---|---|
| `src/slvfe/solver.py` | `posv_wrap`, `syevr_wrap` (sfecalc) | **The cuSolverDn → SciPy replacement** |
| `src/slvfe/config.py` | `sysvars` (sfemain.F90) | Reads the `parameters_fe` namelist; internally split into `Config` (scalars) + `RunData` (arrays), exposed as a single flat `SysVars` facade (see docstring in the file) |
| `src/slvfe/reader.py` | `sysread` (slvfe.F90) | `defcond`, `datread` (reading input files) |
| `src/slvfe/sfecalc.py` | `sfecalc` (slvfe.F90) | Numerical core of the chemical-potential calculation (`chmpot`, `getslncv`, `getinscv`, etc.) |
| `src/slvfe/uvcorrect.py` | `uvcorrect` (sfecorrect.F90) | Lennard-Jones long-range correction (used only when `ljlrc = 'yes'`) |
| `src/slvfe/output.py` | `opwrite` (slvfe.F90) | Result output (`wrtresl`, `wrtmerge`, `wrtcumu`) |
| `src/slvfe/main.py` | `program sfemain` | Entry point logic |
| `src/slvfe/fortran_utils.py` | — | Compatibility helpers for Fortran intrinsics (e.g. `NINT`) |
| `src/slvfe/namelist_parser.py` | — | Self-contained Fortran namelist parser (no external dependency) |
| `src/slvfe/exceptions.py` | — | `SlvfeError`, a normal (catchable) exception used for all user-facing/data errors — see "Error handling" below |

## Error handling

All user-facing/data errors (bad or inconsistent input files,
unsupported parameter combinations, numerical failures, ...) are
raised as `slvfe.exceptions.SlvfeError`, a normal, catchable
`Exception` subclass. If you use this package as a library (rather
than through the `pyslvfe` command), catch `SlvfeError` (or
`Exception`) around calls into it. The CLI entry point (`main()` in
`src/slvfe/main.py`) catches `SlvfeError` itself and prints a one-line
`Error: ...` message to stderr with exit code 1, instead of a Python
traceback.

## Porting notes / things worth double-checking

Since no real input data was available while porting, runtime
verification was limited to a synthetic-data sanity check
(`tests/test_smoke.py`). **Before using this in production, run it
against the same input directories as the existing Fortran version and
confirm the output (especially the "Total solvation free energy"
value) matches.**

Points that deserve particular attention:

1. **Reading the binary correlation-matrix files (`corsln.*`,
   `corref.*`)** (`reader.py: read_fortran_matrix`)
   These are read via `scipy.io.FortranFile`, which understands
   Fortran's unformatted record-length markers. If the Fortran program
   that originally wrote these files (outside this repository, e.g. the
   `mkedmp`/`corr`-equivalent program) used non-standard record markers
   (rare), adjust `read_fortran_matrix` accordingly.

2. **Parsing the `parameters_fe` / `parameters_er` namelists**
   `namelist_parser.py` is a minimal, purpose-built parser (see
   ["Namelist parsing"](#namelist-parsing-parameters_fe--parameters_er)
   above for what it covers). If your namelist files use syntax it
   doesn't handle (e.g. `arr(2:4) = ...` slice assignment), either
   extend the parser or verify the parsed values with
   `namelist_parser.read_namelist(path)`.

3. **Output formatting (column widths, etc.)**
   `output.py` reproduces the Fortran `write` format specifiers
   (`f12.4`, etc.) using Python f-strings, but the exact column widths
   may not match byte-for-byte in every case (the numeric values
   themselves are the same). This is fine for a human-readable log, but
   if downstream tooling parses this stdout output strictly, adjust the
   formatting as needed.

4. **Interactive input in `basic` / `range` mode**
   As in the original Fortran, prompts are shown via `input()`. If you
   want to automate this, replace the relevant parts of
   `reader.defcond` with command-line arguments or a config file
   (`clcond = 'merge'` is the main mode intended for automated/batch
   processing).

5. **Performance of `sfecalc.py`'s main loops**
   The core loops are translated to match the Fortran code index-for-
   index (aside from the 1-based → 0-based conversion), prioritizing
   correctness over performance. For very large `gemax` (thousands or
   more), some of the plain Python loops inside `getslncv`/`getinscv`
   could become a bottleneck. The mesh-grouping step and the
   correlation-matrix reduction (`edscr`, `ecorr`) in `chmpot` are
   already vectorized with `numpy`/`scipy.sparse`; if performance
   becomes an issue elsewhere, profile first and then vectorize.

   One specific, known-and-accepted cost: `SysVars` (`config.py`)
   forwards attribute access to an internal `Config`/`RunData` pair via
   `__getattr__`/`__setattr__`, which is slower than a plain dataclass
   field lookup. This is only worth caring about inside per-bin Python
   loops over large `gemax` (e.g. `chmpot`'s main loop, which reads
   `sv.kT` per bin). If profiling ever shows this mattering, read the
   hot values into local variables once before the loop rather than
   reverting the `Config`/`RunData` split.

## Recommended validation procedure

1. Run both the existing Fortran binary and this Python port against
   the same `soln/refs/parameters_fe` inputs, and compare the numbers
   in stdout (e.g. "Total solvation free energy").
2. Confirm the results match with both `invmtrx = 'evd'` explicitly set
   and the default `'reg'`, to exercise both the `posv_wrap` and
   `syevr_wrap` code paths.
3. Also separately verify the `slncor = 'yes'` case (which exercises
   the `edscr`/`sdrcv` code path).

## License

This is a derivative work of ERmod (`https://github.com/drmaruyama/ermod-openacc`),
which is GPL-2.0-or-later. If you distribute this port, include a
`LICENSE` file with the GPL text (e.g. from
`https://www.gnu.org/licenses/old-licenses/gpl-2.0.txt`) and a short
note pointing back to the original project.
