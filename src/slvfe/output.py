# -*- coding: utf-8 -*-
"""Port of the `opwrite` module (slvfe.F90): printing the results."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .config import SysVars


@dataclass
class OutputState:
    grref: int = 0                 # 0-based index into svgrp/svinf/chmpt (prmcnt-1)
    fe_stat_error: float = 0.0
    mshdif: Optional[np.ndarray] = None   # indexed [group] directly (1-based group #)


def wrtresl(sv: SysVars, ost: OutputState) -> None:
    if sv.slfslt == 'yes':
        print(f"  Self-energy of the solute   =   {sv.slfeng:12.4f}  kcal/mol")

    if sv.clcond in ('basic', 'range'):
        print()
        if sv.numslv > 1 and sv.uvread == 'not':
            print('  Solute-solvent energy       =   '
                  + ''.join(f'{v:12.4f}' for v in sv.aveuv))
        totuv = sv.aveuv.sum()
        if sv.slfslt == 'yes':
            totuv += sv.slfeng
        print(f"  Total solvation energy      =   {totuv:12.4f}  kcal/mol")

    if sv.clcond == 'basic':
        if sv.numslv > 1:
            print('  Solvation free energy       =   '
                  + ''.join(f'{sv.chmpt[pti + 1, 0, 0]:12.4f}' for pti in range(sv.numslv)))
        print(f"  Total solvation free energy =   {sv.chmpt[0, 0, 0]:12.4f}  kcal/mol")

    if sv.clcond in ('range', 'merge'):
        ost.grref = 0
        for prmcnt in range(sv.prmmax):
            if int(sv.svgrp[prmcnt]) == sv.pickgr:
                ost.grref = prmcnt
                break
        ost.mshdif = -np.ones(sv.msemax + 1, dtype=np.float64)  # index directly by group #

    if sv.clcond == 'range':
        k_range = [0] if sv.numslv == 1 else list(range(0, sv.numslv + 1))
        for pti in k_range:
            print()
            if sv.infchk == 'yes':
                if pti == 0:
                    print(" group  inft  solvation free energy difference")
                else:
                    print(f"               {pti:3d}-th component difference")
            else:
                if pti == 0:
                    print(" group    solvation free energy   difference")
                else:
                    print(f"           {pti:3d}-th component       difference")
            for prmcnt in range(sv.prmmax):
                group = int(sv.svgrp[prmcnt])
                inft = int(sv.svinf[prmcnt])
                valcp = sv.chmpt[pti, prmcnt, 0]
                differ = valcp - sv.chmpt[pti, ost.grref, 0]
                if sv.infchk == 'yes':
                    print(f"{group:4d}{inft:7d}{valcp:17.5f}{differ:18.5f}")
                else:
                    print(f"{group:4d}{valcp:20.5f}{differ:18.5f}")
                if (pti == 0 and inft == 0
                        and sv.msemin <= group <= sv.msemax):
                    ost.mshdif[group] = abs(differ)

    if sv.clcond == 'merge':
        wrtmerge(sv, ost)

    if sv.clcond in ('range', 'merge'):
        valid = ost.mshdif[sv.msemin:sv.msemax + 1]
        valid = valid[valid > sv.zero]
        mesh_error = float(valid.max()) if valid.size else 0.0
        if sv.write_mesherror == 'not':
            pass
        elif sv.write_mesherror == 'yes':
            print()
            print(f" Mesh error is {mesh_error:8.3f} kcal/mol")
        elif mesh_error < sv.mesherr:
            pass
        elif sv.clcond == 'merge' and mesh_error > ost.fe_stat_error:
            print()
            print(f" Warning: mesh error is {mesh_error:8.3f} kcal/mol and is "
                  f"larger than the 95% error of {ost.fe_stat_error:8.3f} kcal/mol")
        elif sv.mesherr > sv.zero and mesh_error > sv.mesherr:
            print()
            print(f" Warning: mesh error is {mesh_error:8.3f} kcal/mol and is "
                  f"larger than the threshold value of {sv.mesherr:8.3f} kcal/mol")


def wrtmerge(sv: SysVars, ost: OutputState) -> None:
    numslv = sv.numslv
    numrun = sv.numrun

    if sv.uvread != 'not':
        wrtdata = sv.blockuv[:numslv + 1, :numrun].copy()
        print()
        print()
        print(" cumulative average & 95% error for solvation energy")
        wrtcumu(sv, ost, wrtdata)
        print()

    pti_range = [0] if numslv == 1 else list(range(0, numslv + 1))
    for pti in pti_range:
        avcp0 = sv.chmpt[pti, ost.grref, :numrun].mean()
        for prmcnt in range(sv.prmmax):
            group = int(sv.svgrp[prmcnt])
            inft = int(sv.svinf[prmcnt])
            vals = sv.chmpt[pti, prmcnt, :numrun]
            avecp = vals.mean()
            stdcp = 0.0
            if numrun > 1:
                var = (vals ** 2).mean() - avecp ** 2
                stdcp = math.sqrt(numrun / (numrun - 1.0)) * math.sqrt(max(var, 0.0))
                stdcp = 2.0 * stdcp / math.sqrt(numrun)

            if prmcnt == 0:
                print()
                if pti == 0:
                    if sv.infchk == 'yes':
                        if numrun == 1:
                            print(" group  inft  solvation free energy     difference")
                        else:
                            print(" group  inft  solvation free energy     error          difference")
                    else:
                        if numrun == 1:
                            print(" group    solvation free energy     difference")
                        else:
                            print(" group    solvation free energy     error          difference")
                if numslv > 1:
                    if pti == 0:
                        print("  total solvation free energy")
                    else:
                        print(f"  contribution from {pti:2d}-th solvent component")

            if sv.infchk == 'yes':
                if numrun == 1:
                    print(f"{group:4d}{inft:7d}{avecp:17.5f}{(avecp - avcp0):21.5f}")
                else:
                    print(f"{group:4d}{inft:7d}{avecp:17.5f}{stdcp:18.5f}{(avecp - avcp0):18.5f}")
            else:
                if numrun == 1:
                    print(f"{group:4d}{avecp:20.5f}{(avecp - avcp0):21.5f}")
                else:
                    print(f"{group:4d}{avecp:20.5f}{stdcp:18.5f}{(avecp - avcp0):18.5f}")

            if (pti == 0 and inft == 0 and sv.msemin <= group <= sv.msemax):
                ost.mshdif[group] = abs(avecp - avcp0)

    if numrun > 1:
        print()
        print()
        for pti in pti_range:
            for prmcnt in range(sv.prmmax):
                group = int(sv.svgrp[prmcnt])
                inft = int(sv.svinf[prmcnt])
                showcp = sv.chmpt[pti, prmcnt, :numrun]
                if prmcnt == 0:
                    if sv.infchk == 'yes':
                        if numslv == 1:
                            print(" group  inft   Estimated free energy (kcal/mol)")
                        else:
                            if pti == 0:
                                print(" group  inft   Estimated free energy: total (kcal/mol)")
                            else:
                                print()
                                print(f" group  inft   Estimated free energy:{pti:3d}"
                                      f"-th solvent contribution (kcal/mol)")
                    else:
                        if numslv == 1:
                            print(" group   Estimated free energy (kcal/mol)")
                        else:
                            if pti == 0:
                                print(" group   Estimated free energy: total (kcal/mol)")
                            else:
                                print()
                                print(f" group   Estimated free energy:{pti:3d}"
                                      f"-th solvent contribution (kcal/mol)")

                _print_five_per_line(showcp, group, inft if sv.infchk == 'yes' else None)

    wrtdata = sv.chmpt[:numslv + 1, ost.grref, :numrun]
    print()
    print()
    print(" cumulative average & 95% error for solvation free energy")
    ost.fe_stat_error = wrtcumu(sv, ost, wrtdata)


def _print_five_per_line(values: np.ndarray, group: int, inft) -> None:
    n = len(values)
    if inft is not None:
        prefix = f"{group:4d}{inft:7d}"
        cont_prefix = "           "
    else:
        prefix = f"{group:4d}  "
        cont_prefix = "      "
    for start in range(0, n, 5):
        chunk = values[start:start + 5]
        p = prefix if start == 0 else cont_prefix
        print(p + ''.join(f'{v:13.4f}' for v in chunk))


def wrtcumu(sv: SysVars, ost: OutputState, wrtdata: np.ndarray) -> float:
    """``wrtdata`` has shape (numslv+1, numrun). Returns the 95% error of
    the *last* column (used as ``fe_stat_error`` by the caller when
    called for the solvation free energy)."""
    numslv = sv.numslv
    numrun = sv.numrun
    runcp = np.zeros(numslv + 1, dtype=np.float64)
    runer = np.zeros(numslv + 1, dtype=np.float64)
    stat_error = 0.0

    if numslv >= 2:
        header = "              total             1st component         2nd component"
        if numslv >= 3:
            header += "         3rd component"
        for pti in range(4, numslv + 1):
            if pti < 10:
                header += f"         {pti}th component"
            else:
                header += f"        {pti}th component"
        print(header)

    for cntrun in range(1, numrun + 1):
        recnt = float(cntrun)
        wrtcp = np.zeros(2 * numslv + 2, dtype=np.float64)
        for pti in range(numslv + 1):
            slvfe = wrtdata[pti, cntrun - 1]
            runcp[pti] += slvfe
            runer[pti] += slvfe ** 2
            avecp = runcp[pti] / recnt
            wrtcp[2 * pti] = avecp
            if cntrun > 1:
                factor = runer[pti] / recnt - avecp ** 2
                if factor <= sv.zero:
                    wrtcp[2 * pti + 1] = 0.0
                else:
                    wrtcp[2 * pti + 1] = (2.0 / math.sqrt(recnt)) * math.sqrt(
                        recnt / (recnt - 1.0)) * math.sqrt(factor)

        if cntrun == 1:
            vals = [wrtcp[2 * pti] for pti in range(numslv + 1)]
            if numslv == 1:
                print(f"{cntrun:3d}{vals[0]:11.4f}")
            else:
                print(f"{cntrun:3d}{vals[0]:11.4f}" + ''.join(f'{v:22.4f}' for v in vals[1:]))
        else:
            if numslv == 1:
                print(f"{cntrun:3d}{wrtcp[0]:11.4f}{wrtcp[1]:11.4f}")
            else:
                print(f"{cntrun:3d}" + ''.join(f'{v:11.4f}' for v in wrtcp))
            if cntrun == numrun:
                stat_error = wrtcp[1]

    return stat_error
