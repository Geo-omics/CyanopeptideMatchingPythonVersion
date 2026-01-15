# massql_utils.py
from typing import List, Tuple, Dict, Iterable, Optional
import os
import math
import pandas as pd
from massql import msql_engine, msql_fileloading

# -------------------------------------------------------------------------------------------
# Load a single file
def load_file(input_file: str):
    """Load MS1/MS2 dataframes once for speed."""
    ms1_df, ms2_df = msql_fileloading.load_data(input_file)
    return ms1_df, ms2_df

# Load multiple files
def load_files(input_files: List[str]) -> Dict[str, object]:
    """Load MS1/MS2 dataframes from multiple files.
    Returns both per-file results and merged results (with source_file column).
    """
    ms1_list, ms2_list = [], []
    per_file = {}

    for f in input_files:
        ms1_df, ms2_df = load_file(f)
        ms1_df = ms1_df.copy()
        ms2_df = ms2_df.copy()
        ms1_df["source_file"] = f
        ms2_df["source_file"] = f
        per_file[f] = (ms1_df, ms2_df)
        ms1_list.append(ms1_df)
        ms2_list.append(ms2_df)

    return {
        "per_file": per_file,
        "merged": (pd.concat(ms1_list), pd.concat(ms2_list))
    }

# -------------------------------------------------------------------------------------------
# Build a MassQL query string
def build_massql_query(
    ions_mz: Iterable[float],
    tol_mz: float = 0.01,
    polarity: Optional[str] = None,
    rt_window: Optional[Tuple[float, float]] = None,
) -> str:
    clauses = []
    if polarity:
        clauses.append(f"POLARITY={polarity}")
    if rt_window:
        rtmin, rtmax = rt_window
        clauses.append(f"RTMIN={rtmin} AND RTMAX={rtmax}")
    for m in ions_mz:
        clauses.append(f"MS2PROD={m}:TOLERANCEMZ={tol_mz}")
    where = " AND ".join(clauses)
    return f"QUERY scaninfo(MS2DATA) WHERE {where}"

# -------------------------------------------------------------------------------------------
# Run across files (individual ions)
def run_across_files_individual(
    input_files: Iterable[str],
    ions: Iterable[float],
    tol_mz: float = 0.01,
    polarity: Optional[str] = None,
    rt_window: Optional[Tuple[float,float]] = None,
) -> pd.DataFrame:
    rows = []
    for f in input_files:
        ms1_df, ms2_df = load_file(f)
        for ion in ions:
            q = build_massql_query([ion], tol_mz=tol_mz, polarity=polarity, rt_window=rt_window)
            df = msql_engine.process_query(q, f, ms1_df=ms1_df, ms2_df=ms2_df)
            if df is not None and not df.empty:
                df = df.copy()
                df["ion"] = float(ion)
                df["source_file"] = os.path.basename(f)
                rows.append(df)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

# Run across files (ion combinations)
def run_across_files_combinations(
    input_files: Iterable[str],
    combos: Iterable[Tuple[float, ...]],
    tol_mz: float = 0.01,
    polarity: Optional[str] = None,
    rt_window: Optional[Tuple[float,float]] = None,
) -> pd.DataFrame:
    rows = []
    for f in input_files:
        ms1_df, ms2_df = load_file(f)
        for combo in combos:
            q = build_massql_query(combo, tol_mz=tol_mz, polarity=polarity, rt_window=rt_window)
            df = msql_engine.process_query(q, f, ms1_df=ms1_df, ms2_df=ms2_df)
            if df is not None and not df.empty:
                df = df.copy()
                df["combo_label"] = "+".join(f"{m:.4f}" for m in combo)
                df["combo_size"] = len(combo)
                df["source_file"] = os.path.basename(f)
                rows.append(df)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

# -------------------------------------------------------------------------------------------
# Add friendly labels for MP ions
def add_MP_labels(df: pd.DataFrame, ion_to_MP: Dict[float, str], label_col: str = "MP") -> pd.DataFrame:
    out = df.copy()
    if "ion" in out.columns:
        out[label_col] = out["ion"].map(lambda x: ion_to_MP.get(float(x), f"{x:.4f}"))
    elif "combo_label" in out.columns:
        def label_combo(s: str) -> str:
            parts = s.split("+")
            return "+".join([ion_to_MP.get(float(p), p) for p in parts])
        out[label_col] = out["combo_label"].map(label_combo)
    return out

#add  friendly labels for MC ions
def add_MC_labels(df: pd.DataFrame, ion_to_MC: Dict[float, str], label_col: str = "MC") -> pd.DataFrame:
    out = df.copy()
    if "ion" in out.columns:
        out[label_col] = out["ion"].map(lambda x: ion_to_MC.get(float(x), f"{x:.4f}"))
    elif "combo_label" in out.columns:
        def label_combo(s: str) -> str:
            parts = s.split("+")
            return "+".join([ion_to_MC.get(float(p), p) for p in parts])
        out[label_col] = out["combo_label"].map(label_combo)
    return out

#add friendly labels for AR
def add_AR_labels(df: pd.DataFrame, ion_to_AR: Dict[float, str], label_col: str = "AR") -> pd.DataFrame:
    out = df.copy()
    if "ion" in out.columns:
        out[label_col] = out["ion"].map(lambda x: ion_to_AR.get(float(x), f"{x:.4f}"))
    elif "combo_label" in out.columns:
        def label_combo(s: str) -> str:
            parts = s.split("+")
            return "+".join([ion_to_AR.get(float(p), p) for p in parts])
        out[label_col] = out["combo_label"].map(label_combo)
    return out

#add friendly labels for AB
def add_AB_labels(df: pd.DataFrame, ion_to_AB: Dict[float, str], label_col: str = "AB") -> pd.DataFrame:
    out = df.copy()
    if "ion" in out.columns:
        out[label_col] = out["ion"].map(lambda x: ion_to_AB.get(float(x), f"{x:.4f}"))
    elif "combo_label" in out.columns:
        def label_combo(s: str) -> str:
            parts = s.split("+")
            return "+".join([ion_to_AB.get(float(p), p) for p in parts])
        out[label_col] = out["combo_label"].map(label_combo)
    return out

#add friendly labels for MV
def add_MV_labels(df: pd.DataFrame, ion_to_MV: Dict[float, str], label_col: str = "MV") -> pd.DataFrame:
    out = df.copy()
    if "ion" in out.columns:
        out[label_col] = out["ion"].map(lambda x: ion_to_MV.get(float(x), f"{x:.4f}"))
    elif "combo_label" in out.columns:
        def label_combo(s: str) -> str:
            parts = s.split("+")
            return "+".join([ion_to_MV.get(float(p), p) for p in parts])
        out[label_col] = out["combo_label"].map(label_combo)
    return out

#add friendly labels for AG
def add_AG_labels(df: pd.DataFrame, ion_to_AG: Dict[float, str], label_col: str = "AG") -> pd.DataFrame:
    out = df.copy()
    if "ion" in out.columns:
        out[label_col] = out["ion"].map(lambda x: ion_to_AG.get(float(x), f"{x:.4f}"))
    elif "combo_label" in out.columns:
        def label_combo(s: str) -> str:
            parts = s.split("+")
            return "+".join([ion_to_AG.get(float(p), p) for p in parts])
        out[label_col] = out["combo_label"].map(label_combo)
    return out

#add friendly labels for MG
def add_MG_labels(df: pd.DataFrame, ion_to_MG: Dict[float, str], label_col: str = "MG") -> pd.DataFrame:
    out = df.copy()
    if "ion" in out.columns:
        out[label_col] = out["ion"].map(lambda x: ion_to_MG.get(float(x), f"{x:.4f}"))
    elif "combo_label" in out.columns:
        def label_combo(s: str) -> str:
            parts = s.split("+")
            return "+".join([ion_to_MG.get(float(p), p) for p in parts])
        out[label_col] = out["combo_label"].map(label_combo)
    return out


# massql_utils.py 01/04/2025 (adding reference compound to this search as well)

import os
import numpy as np
import pandas as pd
from typing import Iterable, Tuple, Optional, Dict, List

def ms1_xic_auc_from_ms1df(
    ms1_df: pd.DataFrame,
    target_mz: float,
    tol_mz: float = 0.01,
    rt_window: Optional[Tuple[float, float]] = None,
    polarity: Optional[str] = None,   # "POSITIVE"/"NEGATIVE"/None; your ms1_df uses 1/-1 typically
    intensity_col: str = "i",
    mz_col: str = "mz",
    rt_col: str = "rt",
) -> float:
    """
    Compute MS1 XIC AUC by filtering MS1 points near target_mz (±tol_mz),
    optionally within an RT window, then integrating intensity vs RT using trapz.
    """
    df = ms1_df

    # Polarity filter if requested
    if polarity is not None:
        # your ms1_df shows polarity as numeric (1). If you pass "POSITIVE"/"NEGATIVE", map it.
        pol_map = {"POSITIVE": 1, "NEGATIVE": -1}
        pol_val = pol_map.get(polarity, polarity)  # allow passing 1/-1 directly
        if "polarity" in df.columns:
            df = df[df["polarity"] == pol_val]

    # m/z filter
    df = df[(df[mz_col] >= target_mz - tol_mz) & (df[mz_col] <= target_mz + tol_mz)]

    # RT filter
    if rt_window is not None:
        rtmin, rtmax = rt_window
        df = df[(df[rt_col] >= rtmin) & (df[rt_col] <= rtmax)]

    if df.empty:
        return 0.0

    # If multiple points per scan, sum intensities per (rt) to make a clean XIC trace
    trace = (
        df.groupby(rt_col, as_index=False)[intensity_col]
        .sum()
        .sort_values(rt_col)
    )

    x = trace[rt_col].to_numpy(dtype=float)
    y = trace[intensity_col].to_numpy(dtype=float)

    if len(x) < 2:
        return 0.0

    return float(np.trapz(y, x))


def reference_auc_per_file(
    input_files: Iterable[str],
    ref_mz: float,
    tol_mz: float = 0.01,
    rt_window: Optional[Tuple[float, float]] = None,
    polarity: Optional[str] = None,
) -> Tuple[Dict[str, float], List[str]]:
    """
    Compute MS1 AUC of a reference compound in each file.
    Returns (ref_auc_dict, missing_files_list) using basenames.
    """
    ref_auc: Dict[str, float] = {}
    missing: List[str] = []

    for f in input_files:
        ms1_df, ms2_df = load_file(f)  # uses your existing loader
        auc = ms1_xic_auc_from_ms1df(
            ms1_df,
            target_mz=ref_mz,
            tol_mz=tol_mz,
            rt_window=rt_window,
            polarity=polarity,
        )
        base = os.path.basename(f)
        if auc <= 0:
            missing.append(base)
        else:
            ref_auc[base] = auc

    return ref_auc, missing


def ms1_auc_table_for_targets(
    input_files: Iterable[str],
    target_mzs: Iterable[float],
    tol_mz: float = 0.01,
    rt_window: Optional[Tuple[float, float]] = None,
    polarity: Optional[str] = None,
    label_map: Optional[Dict[float, str]] = None,
) -> pd.DataFrame:
    """
    For each file and each target m/z, compute MS1 AUC.
    Returns a tidy table: file, target_mz, label, ms1_auc.
    """
    rows = []
    for f in input_files:
        ms1_df, ms2_df = load_file(f)
        base = os.path.basename(f)
        for mz in target_mzs:
            auc = ms1_xic_auc_from_ms1df(
                ms1_df,
                target_mz=float(mz),
                tol_mz=tol_mz,
                rt_window=rt_window,
                polarity=polarity,
            )
            rows.append({
                "source_file": base,
                "target_mz": float(mz),
                "label": (label_map.get(float(mz)) if label_map else None),
                "ms1_auc": auc,
            })
    return pd.DataFrame(rows)


def add_auc_over_reference(df: pd.DataFrame, ref_auc: Dict[str, float]) -> pd.DataFrame:
    """
    Adds ref_auc and auc_over_ref columns to a table that has source_file and ms1_auc.
    """
    out = df.copy()
    out["ref_auc"] = out["source_file"].map(ref_auc)
    out["auc_over_ref"] = out["ms1_auc"] / out["ref_auc"]
    return out
