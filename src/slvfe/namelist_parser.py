# -*- coding: utf-8 -*-
"""
A small, dependency-free parser for Fortran namelist files, replacing
the external `f90nml` package.

This is NOT a full implementation of the Fortran namelist standard --
it covers what `parameters_fe` (namelist `fevars`) and `parameters_er`
(namelist `ene_param`) actually use in this codebase:

    - one or more `&groupname  key = value, key2 = value2, ... /` blocks
    - scalar values: quoted strings ('...' or "..."), logicals
      (.true./.false./.t./.f./t/f, case-insensitive), integers, and
      reals (including Fortran's 'd'/'D' double-precision exponent
      marker)
    - simple comma-separated lists for array-valued keys
      (e.g. `hostspec = 1, 2, 3`)
    - the `n*value` repeat-count shorthand (e.g. `5*0.0`)
    - `!` comments (correctly ignored inside quoted strings)

Not supported (and not needed by this codebase): array subscript /
slice assignment (`arr(2:4) = ...`), the older `$group ... $end`
namelist delimiter style, and derived-type component assignment.

Usage mirrors the small subset of the `f90nml` API used elsewhere in
this package:

    nml = read_namelist('parameters_fe')
    if 'fevars' in nml:
        for key, value in nml['fevars'].items():
            ...
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Union

NamelistDict = Dict[str, Dict[str, Any]]

_TOKEN_RE = re.compile(
    r"'(?:[^']|'')*'"      # single-quoted string ('' = escaped quote)
    r'|"(?:[^"]|"")*"'      # double-quoted string
    r'|&\w+'                # group start, e.g. &fevars
    r'|[^\s,=/]+'           # bare token (number, logical, bare word, name)
    r'|[=,/]'               # punctuation
)

_REPEAT_RE = re.compile(r'^(\d+)\*(.*)$')


def _strip_comment(line: str) -> str:
    """Remove a trailing '!' comment, ignoring '!' inside quotes."""
    in_squote = False
    in_dquote = False
    for i, ch in enumerate(line):
        if ch == "'" and not in_dquote:
            in_squote = not in_squote
        elif ch == '"' and not in_squote:
            in_dquote = not in_dquote
        elif ch == '!' and not in_squote and not in_dquote:
            return line[:i]
    return line


def _parse_scalar(tok: str) -> Any:
    if (len(tok) >= 2 and tok[0] == tok[-1] and tok[0] in ("'", '"')):
        inner = tok[1:-1]
        return inner.replace(tok[0] * 2, tok[0])  # unescape doubled quote
    low = tok.lower()
    if low in ('.true.', '.t.', 't'):
        return True
    if low in ('.false.', '.f.', 'f'):
        return False
    try:
        return int(tok)
    except ValueError:
        pass
    try:
        return float(tok.replace('d', 'e').replace('D', 'E'))
    except ValueError:
        pass
    return tok  # fall back to a bare (unquoted) string


def _parse_value_token(tok: str) -> List[Any]:
    """Returns a list because of the `n*value` repeat shorthand."""
    m = _REPEAT_RE.match(tok)
    if m and not (tok[0] in ("'", '"')):
        count = int(m.group(1))
        rest = m.group(2)
        return [_parse_scalar(rest)] * count
    return [_parse_scalar(tok)]


def _strip_subscript(name: str) -> str:
    """`hostspec(3)` -> `hostspec` (subscript/slice info is discarded;
    not needed by anything this codebase reads from a namelist)."""
    paren = name.find('(')
    return name[:paren] if paren != -1 else name


def parse_namelist_text(text: str) -> NamelistDict:
    lines = [_strip_comment(line) for line in text.splitlines()]
    blob = ' '.join(lines)
    tokens = _TOKEN_RE.findall(blob)

    result: NamelistDict = {}
    group_name = None
    current: Dict[str, Any] = {}
    pending_key = None
    pending_vals: List[Any] = []

    def finalize_key() -> None:
        nonlocal pending_key, pending_vals
        if pending_key is not None:
            base = _strip_subscript(pending_key).lower()
            if len(pending_vals) == 1:
                current[base] = pending_vals[0]
            else:
                current[base] = list(pending_vals)
        pending_key = None
        pending_vals = []

    i = 0
    n = len(tokens)
    while i < n:
        tok = tokens[i]
        if tok.startswith('&'):
            group_name = tok[1:].lower()
            current = {}
            pending_key = None
            pending_vals = []
            i += 1
            continue

        if group_name is None:
            i += 1
            continue

        if tok == '/':
            finalize_key()
            result[group_name] = current
            group_name = None
            i += 1
            continue

        if tok == ',':
            i += 1
            continue

        if tok == '=':
            # stray '=' with no preceding key token; ignore defensively
            i += 1
            continue

        if i + 1 < n and tokens[i + 1] == '=':
            finalize_key()
            pending_key = tok
            i += 2
            continue

        pending_vals.extend(_parse_value_token(tok))
        i += 1

    # tolerate a missing trailing '/' on the last group
    if group_name is not None:
        finalize_key()
        result[group_name] = current

    return result


def read_namelist(path: Union[str, Path]) -> NamelistDict:
    text = Path(path).read_text()
    return parse_namelist_text(text)
