# pyslvfe — Python port of ERmod's `slvfe`

A Python port of `ermod-openacc/slvfe` (`slvfe.F90`, `sfemain.F90`,
`sfecorrect.F90`). The GPU-dependent solvers (cuSolverDn) have been
replaced with CPU SciPy/LAPACK calls.

## File layout

| File | Original Fortran module | Contents |
|---|---|---|
| `solver.py` | `posv_wrap`, `syevr_wrap` (sfecalc) | **The cuSolverDn → SciPy replacement** |
| `config.py` | `sysvars` (sfemain.F90) | Reads the `parameters_fe` namelist; internally split into `Config` (scalars) + `RunData` (arrays), exposed as a single flat `SysVars` facade (see docstring in the file) |
| `reader.py` | `sysread` (slvfe.F90) | `defcond`, `datread` (reading input files) |
| `sfecalc.py` | `sfecalc` (slvfe.F90) | Numerical core of the chemical-potential calculation (`chmpot`, `getslncv`, `getinscv`, etc.) |
| `uvcorrect.py` | `uvcorrect` (sfecorrect.F90) | Lennard-Jones long-range correction (used only when `ljlrc = 'yes'`) |
| `output.py` | `opwrite` (slvfe.F90) | Result output (`wrtresl`, `wrtmerge`, `wrtcumu`) |
| `main.py` | `program sfemain` | Entry point |
| `fortran_utils.py` | — | Compatibility helpers for Fortran intrinsics (e.g. `NINT`) |
| `namelist_parser.py` | — | Self-contained Fortran namelist parser (no external dependency) |
| `test_smoke.py` | — | Sanity check with synthetic data (no real input files needed) |
| `test_reader.py` | — | Integration test for `reader.py` (writes small synthetic input files to a temp dir) |
| `test_namelist_parser.py` | — | Unit tests for `namelist_parser.py` |

## Dependencies

```
pip install numpy scipy
```

- `numpy` / `scipy`: numerical linear algebra (Cholesky factorization,
  symmetric eigendecomposition)

Namelist parsing (`parameters_fe`, `parameters_er`) is handled by the
small, self-contained parser in `namelist_parser.py` — no external
namelist package is required. It covers what these two files actually
use: quoted strings, logicals, integers/reals (including Fortran's
`d`/`D` exponent marker), simple comma-separated arrays, the `n*value`
repeat shorthand, and `!` comments. It does not implement the full
Fortran namelist standard (e.g. `arr(2:4) = ...` slice assignment or
the legacy `$group ... $end` delimiter style), but this is not needed
by `parameters_fe` / `parameters_er`. See `test_namelist_parser.py` for
example input/output.

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

## Running it

```bash
cd <directory containing parameters_fe, soln/, refs/>
python3 /PATH/TO/slvfe.py
```

As in the original Fortran program, when `clcond = 'basic'` or
`'range'`, you will be prompted for a few values via standard input
(`input()` calls).

## Porting notes / things worth double-checking

Since no real input data was available while porting, runtime
verification was limited to a synthetic-data sanity check
(`test_smoke.py`). **Before using this in production, run it against
the same input directories as the existing Fortran version and confirm
the output (especially the "Total solvation free energy" value)
matches.**

Points that deserve particular attention:

1. **Reading the binary correlation-matrix files (`corsln.*`,
   `corref.*`)** (`reader.py: read_fortran_matrix`)
   These are read via `scipy.io.FortranFile`, which understands
   Fortran's unformatted record-length markers. If the Fortran program
   that originally wrote these files (outside this repository, e.g. the
   `mkedmp`/`corr`-equivalent program) used non-standard record markers
   (rare), adjust `read_fortran_matrix` accordingly.

2. **Parsing the `parameters_fe` / `parameters_er` namelists**
   `namelist_parser.py` is a minimal, purpose-built parser (see the
   "Dependencies" section above for what it covers). If your namelist
   files use syntax it doesn't handle (e.g. `arr(2:4) = ...` slice
   assignment), either extend the parser or verify the parsed values
   with `namelist_parser.read_namelist(path)`.

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
