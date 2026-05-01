# qc_helpers.py
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd
import re
import shutil

# -----------------------------
# Config
# -----------------------------
@dataclass
class QCConfig:
    # Metadata (optional)
    metadata_path: Optional[str] = None     # e.g., "/path/to/metadata.csv"
    blank_col: str = "sample_type"
    blank_value: str = "blank"
    batch_col: str = "batch"

    # Reference (optional)
    ref_mz: Optional[float] = None          # None disables reference normalization
    ref_tol: float = 0.01
    ref_rt_window: Optional[Tuple[float, float]] = None  # (rt_min, rt_max) in minutes

    # Blank filter (feature-level, applies to downstream only)
    apply_blank_filter: bool = True
    blank_ratio_thresh: float = 3.0         # keep if median(sample)/median(blank) >= thresh
    blank_stat: str = "median"              # "median" or "mean"

    # Batch correction (applies to downstream only)
    apply_batch_correction: bool = True     # median scaling across batches
    batch_use_nonblank_only: bool = True

    # Columns used in your pipeline
    file_col: str = "source_file"           # in perfile_summary_auc
    ms1_points_file_col: str = "source_file"


# -----------------------------
# Small utilities
# -----------------------------
def _stat(x: pd.Series, how: str) -> float:
    x = pd.to_numeric(x, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if x.empty:
        return 0.0
    return float(x.median() if how == "median" else x.mean())


def load_metadata(cfg: QCConfig) -> Optional[pd.DataFrame]:
    if cfg.metadata_path is None or str(cfg.metadata_path).strip() == "":
        print("[INFO] metadata_path=None — blanks + batch correction will be skipped.")
        return None
    p = Path(cfg.metadata_path)
    if not p.exists():
        print(f"[WARN] metadata not found: {p.resolve()} — blanks + batch correction will be skipped.")
        return None
    meta = pd.read_csv(p)
    # normalize filename column if present
    if "filename" in meta.columns:
        meta["filename"] = meta["filename"].astype(str).str.strip().apply(lambda x: Path(x).name)
    # allow source_file too
    if "source_file" in meta.columns:
        meta["source_file"] = meta["source_file"].astype(str).str.strip().apply(lambda x: Path(x).name)
    return meta


def load_ms1_points(ms1_points_file: str) -> pd.DataFrame:
    p = Path(ms1_points_file)

    if not p.exists():
        raise FileNotFoundError(f"[ERROR] MS1 points file not found: {p.resolve()}")

    print("[INFO] Using MS1 points file:", p.resolve())
    ms1_points = pd.read_csv(p)

    # normalize basenames
    if "source_file" in ms1_points.columns:
        ms1_points["source_file"] = ms1_points["source_file"].astype(str).apply(lambda x: Path(x).name)
    elif "filename" in ms1_points.columns:
        ms1_points["filename"] = ms1_points["filename"].astype(str).apply(lambda x: Path(x).name)
        ms1_points.rename(columns={"filename": "source_file"}, inplace=True)
    else:
        raise ValueError("[ERROR] ms1_points CSV must have 'source_file' (or 'filename').")

    # numeric safety
    for c in ["precmz", "rt", "i"]:
        if c in ms1_points.columns:
            ms1_points[c] = pd.to_numeric(ms1_points[c], errors="coerce")

    return ms1_points

# -----------------------------
# Reference AUC
# -----------------------------
def compute_ref_auc_df(
    ms1_points: pd.DataFrame,
    *,
    ref_mz: Optional[float],
    ref_tol: float,
    ref_rt_window: Optional[Tuple[float, float]],
    polarity: Optional[int] = None,
) -> pd.DataFrame:
    """
    Returns columns: source_file, ref_ms1_auc
    If ref_mz is None => ref_ms1_auc=1.0 (no normalization)
    """
    files = ms1_points["source_file"].astype(str).unique()

    if ref_mz is None or ref_rt_window is None:
        print("[INFO] Reference normalization OFF (REF_MZ=None or ref_rt_window=None).")
        return pd.DataFrame({"source_file": files, "ref_ms1_auc": 1.0})

    rt_min, rt_max = ref_rt_window
    ref_pts = ms1_points.loc[
        (ms1_points["precmz"].between(ref_mz - ref_tol, ref_mz + ref_tol)) &
        (ms1_points["rt"].between(rt_min, rt_max)),
        ["source_file", "rt", "i"] + (["polarity"] if "polarity" in ms1_points.columns else [])
    ].copy()

    if polarity is not None and "polarity" in ref_pts.columns:
        ref_pts = ref_pts[ref_pts["polarity"] == polarity]

    trace = (
        ref_pts.groupby(["source_file", "rt"], as_index=False)["i"]
        .sum()
        .sort_values(["source_file", "rt"])
    )

    ref_auc = {}
    for fname, sub in trace.groupby("source_file", sort=False):
        ref_auc[fname] = 0.0 if len(sub) < 2 else float(np.trapezoid(sub["i"].to_numpy(), sub["rt"].to_numpy()))

    # ensure all files exist
    for f in files:
        ref_auc.setdefault(f, 0.0)

    z = [k for k, v in ref_auc.items() if v == 0.0]
    if z:
        print("[WARN] Reference AUC==0 for these file(s):")
        for f in z[:20]:
            print("  -", f)
        if len(z) > 20:
            print(f"  ... and {len(z)-20} more")

    return pd.DataFrame([{"source_file": k, "ref_ms1_auc": v} for k, v in ref_auc.items()])


def add_reference_normalization(
    perfile_summary_auc: pd.DataFrame,
    ref_auc_df: pd.DataFrame,
    *,
    ref_mz: Optional[float],
) -> pd.DataFrame:
    """
    Adds ms1_auc_over_ref.
    If ref_mz is None => ms1_auc_over_ref = ms1_auc (so downstream always has a quant column)
    """
    out = perfile_summary_auc.merge(ref_auc_df, on="source_file", how="left")

    if ref_mz is None:
        out["ms1_auc_over_ref"] = out["ms1_auc"]
        print("[INFO] REF_MZ=None — using ms1_auc as ms1_auc_over_ref.")
        return out

    out["ms1_auc_over_ref"] = np.where(
        (out["ref_ms1_auc"].notna()) & (out["ref_ms1_auc"] > 0),
        out["ms1_auc"] / out["ref_ms1_auc"],
        np.nan,
    )
    return out


# -----------------------------
# Batch correction (downstream-only)
# -----------------------------

def apply_batch_correction_median_scaling(
    perfile_df: pd.DataFrame,
    metadata: Optional[pd.DataFrame],
    cfg: QCConfig,
    *,
    intensity_col: str = "ms1_auc_over_ref",
) -> pd.DataFrame:
    """
    Median scaling across batches on intensity_col.
    Factors computed using non-blank files only if cfg.batch_use_nonblank_only=True.
    """
    print("[DEBUG] entered apply_batch_correction_median_scaling")
    print("[DEBUG] perfile_df columns:", perfile_df.columns.tolist())
    print("[DEBUG] has batch?", "batch" in perfile_df.columns)
    if (not cfg.apply_batch_correction) or (metadata is None):
        return perfile_df

    # accept either filename or source_file in metadata
    key = "source_file" if "source_file" in metadata.columns else ("filename" if "filename" in metadata.columns else None)
    if key is None or cfg.batch_col not in metadata.columns:
        print(f"[INFO] Metadata missing '{key}' or '{cfg.batch_col}' — batch correction skipped.")
        return perfile_df

    if intensity_col not in perfile_df.columns:
        print(f"[INFO] intensity_col='{intensity_col}' missing — batch correction skipped.")
        return perfile_df

    meta = metadata.copy()
    meta[key] = meta[key].astype(str).str.strip().apply(lambda x: Path(x).name)

    df = perfile_df.copy()
    df["source_file"] = df["source_file"].astype(str).str.strip().apply(lambda x: Path(x).name)

    for col in [cfg.batch_col, cfg.blank_col]:
        if col in df.columns:
            print(f"[DEBUG] dropping existing '{col}' before metadata merge")
            df = df.drop(columns=[col])

    df = df.merge(meta[[key, cfg.batch_col] + ([cfg.blank_col] if cfg.blank_col in meta.columns else [])],
                  left_on="source_file", right_on=key, how="left")

    if df[cfg.batch_col].isna().all():
        print(f"[INFO] No batch labels found in '{cfg.batch_col}' — batch correction skipped.")
        return perfile_df

    use = df
    if cfg.batch_use_nonblank_only and (cfg.blank_col in df.columns):
        is_blank = df[cfg.blank_col].astype(str).str.lower().eq(str(cfg.blank_value).lower())
        use = df.loc[~is_blank].copy()

    vals = pd.to_numeric(use[intensity_col], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if vals.empty:
        print("[INFO] No valid intensities for batch factor computation — batch correction skipped.")
        return perfile_df

    global_med = float(vals.median())
    batch_meds = use.groupby(cfg.batch_col)[intensity_col].median()

    factors = {}
    for b, m in batch_meds.items():
        m = float(m) if pd.notna(m) else 0.0
        factors[b] = (global_med / m) if m > 0 else 1.0

    print("[INFO] Batch correction factors (median scaling) on", intensity_col)
    for b, f in factors.items():
        print(f"  batch={b}: factor={f:.4f}")

    df[intensity_col] = pd.to_numeric(df[intensity_col], errors="coerce") * df[cfg.batch_col].map(factors).fillna(1.0)

    # return same schema as input (drop join helper cols)
    out = perfile_df.copy()
    out[intensity_col] = df[intensity_col].to_numpy()
    return out


# -----------------------------
# Blank filtering (downstream-only)
# -----------------------------
def blank_filter_perfile_table(
    perfile_df: pd.DataFrame,
    metadata: Optional[pd.DataFrame],
    cfg: QCConfig,
    *,
    intensity_col: str = "ms1_auc_over_ref",
    feature_cols: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Returns (filtered_perfile_df, keep_table_features, removed_table_features)

    keep rule per feature (across files):
      keep if blank_stat <= 0 OR sample_stat / blank_stat >= cfg.blank_ratio_thresh
    """
    if (not cfg.apply_blank_filter) or (metadata is None):
        print("[INFO] Blank filter skipped (disabled or no metadata).")
        return perfile_df, pd.DataFrame(), pd.DataFrame()

    key = "source_file" if "source_file" in metadata.columns else ("filename" if "filename" in metadata.columns else None)
    if key is None or cfg.blank_col not in metadata.columns:
        print(f"[INFO] Metadata missing '{key}' or '{cfg.blank_col}' — blank filter skipped.")
        return perfile_df, pd.DataFrame(), pd.DataFrame()

    if intensity_col not in perfile_df.columns:
        print(f"[INFO] intensity_col='{intensity_col}' missing — blank filter skipped.")
        return perfile_df, pd.DataFrame(), pd.DataFrame()

    df = perfile_df.copy()
    df["source_file"] = df["source_file"].astype(str).str.strip().apply(lambda x: Path(x).name)

    meta = metadata.copy()
    meta[key] = meta[key].astype(str).str.strip().apply(lambda x: Path(x).name)

    df = df.merge(meta[[key, cfg.blank_col]], left_on="source_file", right_on=key, how="left")
    is_blank = df[cfg.blank_col].astype(str).str.lower().eq(str(cfg.blank_value).lower())

    # choose feature columns
    if feature_cols is None:
        feature_cols = []
        for c in ["merged_precmz", "rt_median", "charge"]:
            if c in df.columns:
                feature_cols.append(c)
        if not feature_cols:
            raise ValueError("[ERROR] No default feature columns found (need merged_precmz/rt_median/charge).")

    # universe of features present
    universe_table = _make_universe_table(df, feature_cols)

    keep_rows = []
    for feat, g in df.groupby(feature_cols, sort=False):
        g_blank = pd.to_numeric(
            g.loc[is_blank.loc[g.index], intensity_col], errors="coerce"
        ).replace([np.inf, -np.inf], np.nan).dropna()

        g_samp = pd.to_numeric(
            g.loc[~is_blank.loc[g.index], intensity_col], errors="coerce"
        ).replace([np.inf, -np.inf], np.nan).dropna()

        has_blank = not g_blank.empty

        if not has_blank:
            keep = True
        else:
            b = _stat(g_blank, cfg.blank_stat)
            s = _stat(g_samp, cfg.blank_stat)
            keep = (b <= 0) or ((s / b) >= cfg.blank_ratio_thresh)
        if keep:
            if not isinstance(feat, tuple):
                feat = (feat,)
            keep_rows.append(dict(zip(feature_cols, feat)))

    keep_table = pd.DataFrame(keep_rows).drop_duplicates()
    removed_table = compute_removed_table(universe_table, keep_table, feature_cols)

    if keep_table.empty:
        print("[WARN] Blank filter kept 0 features (check metadata blank labels / thresholds).")
        return perfile_df.iloc[0:0].copy(), keep_table, removed_table

    before = len(perfile_df)
    filtered = df.merge(keep_table, on=feature_cols, how="inner")
    after = len(filtered)

    # drop join helper columns from merge
    drop_cols = [key, cfg.blank_col]
    filtered = filtered.drop(columns=[c for c in drop_cols if c in filtered.columns], errors="ignore")

    print(
        f"[INFO] Blank filter applied on '{intensity_col}' — rows {before} -> {after}, "
        f"kept features={len(keep_table)}, removed features={len(removed_table)}"
    )
    return filtered, keep_table, removed_table


def save_qc_audit(
    out_dir: str,
    *,
    perfile_raw: pd.DataFrame,
    perfile_clean: Optional[pd.DataFrame] = None,
    keep_table: Optional[pd.DataFrame] = None,
    removed_table: Optional[pd.DataFrame] = None,
    tag: str = "CLASS",
):
    d = Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    raw_path = d / f"{tag}_perfile_RAW_{ts}.csv"
    perfile_raw.to_csv(raw_path, index=False)
    print("[DONE] wrote:", raw_path)

    if perfile_clean is not None:
        clean_path = d / f"{tag}_perfile_CLEAN_{ts}.csv"
        perfile_clean.to_csv(clean_path, index=False)
        print("[DONE] wrote:", clean_path)

    if keep_table is not None and (not keep_table.empty):
        keep_path = d / f"{tag}_keep_features_{ts}.csv"
        keep_table.to_csv(keep_path, index=False)
        print("[DONE] wrote:", keep_path)

    if removed_table is not None and (not removed_table.empty):
        rem_path = d / f"{tag}_blank_removed_features_{ts}.csv"
        removed_table.to_csv(rem_path, index=False)
        print("[DONE] wrote:", rem_path)

# ------------------------------------------------------------
#drop blank files from CLEAN table (row-level)
# ------------------------------------------------------------
def drop_blank_rows(perfile_df, metadata, cfg):
    if metadata is None or perfile_df is None or perfile_df.empty:
        return perfile_df

    key = "source_file" if "source_file" in metadata.columns else ("filename" if "filename" in metadata.columns else None)
    if key is None or cfg.blank_col not in metadata.columns:
        print("[INFO] Cannot drop blank rows (metadata missing filename/source_file or blank_col).")
        return perfile_df

    meta = metadata.copy()
    meta[key] = meta[key].astype(str).str.strip().apply(lambda x: Path(x).name)

    df = perfile_df.copy()
    df["source_file"] = df["source_file"].astype(str).str.strip().apply(lambda x: Path(x).name)

    df = df.merge(meta[[key, cfg.blank_col]], left_on="source_file", right_on=key, how="left")
    is_blank = df[cfg.blank_col].astype(str).str.lower().eq(str(cfg.blank_value).lower())

    before = len(df)
    df = df.loc[~is_blank].drop(columns=[key, cfg.blank_col], errors="ignore")
    after = len(df)

    print(f"[INFO] Dropped blank rows from CLEAN: {before} -> {after}")
    return df



def blank_filter_perfile_table_by_batch(
    perfile_df: pd.DataFrame,
    metadata: Optional[pd.DataFrame],
    cfg: QCConfig,
    *,
    intensity_col: str = "ms1_auc_over_ref",
    feature_cols: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Batch-aware blank filtering.
    Filters rows (feature,file) within each batch using blanks from that same batch.

    Keep rule per (feature,batch):
      keep if blank_stat <= 0 OR sample_stat / blank_stat >= cfg.blank_ratio_thresh

    Returns: (filtered_perfile_df, keep_table, removed_table)
      - keep_table includes batch + feature id cols
      - removed_table includes batch + feature id cols
    """
    if (not cfg.apply_blank_filter) or (metadata is None):
        print("[INFO] Blank filter skipped (disabled or no metadata).")
        return perfile_df, pd.DataFrame(), pd.DataFrame()

    key = "source_file" if "source_file" in metadata.columns else ("filename" if "filename" in metadata.columns else None)
    if key is None or cfg.blank_col not in metadata.columns or cfg.batch_col not in metadata.columns:
        print(f"[INFO] Metadata missing '{key}' or '{cfg.blank_col}' or '{cfg.batch_col}' — batch-aware blank filter skipped.")
        return perfile_df, pd.DataFrame(), pd.DataFrame()

    if intensity_col not in perfile_df.columns:
        print(f"[INFO] intensity_col='{intensity_col}' missing — blank filter skipped.")
        return perfile_df, pd.DataFrame(), pd.DataFrame()

    df = perfile_df.copy()
    df["source_file"] = df["source_file"].astype(str).str.strip().apply(lambda x: Path(x).name)

    meta = metadata.copy()
    meta[key] = meta[key].astype(str).str.strip().apply(lambda x: Path(x).name)

    df = df.merge(meta[[key, cfg.blank_col, cfg.batch_col]], left_on="source_file", right_on=key, how="left")
    is_blank = df[cfg.blank_col].astype(str).str.lower().eq(str(cfg.blank_value).lower())

    if feature_cols is None:
        feature_cols = [c for c in ["merged_precmz", "rt_median", "charge"] if c in df.columns]
        if not feature_cols:
            raise ValueError("[ERROR] No default feature columns found (need merged_precmz/rt_median/charge).")

    group_cols = feature_cols + [cfg.batch_col]

    # universe of feature×batch groups present
    universe_table = _make_universe_table(df, group_cols)

    keep_rows = []
    for feat_batch, g in df.groupby(group_cols, sort=False):
        g_blank = pd.to_numeric(
            g.loc[is_blank.loc[g.index], intensity_col], errors="coerce"
        ).replace([np.inf, -np.inf], np.nan).dropna()

        g_samp = pd.to_numeric(
            g.loc[~is_blank.loc[g.index], intensity_col], errors="coerce"
        ).replace([np.inf, -np.inf], np.nan).dropna()

        has_blank = not g_blank.empty

        if not has_blank:
            keep = True
        else:
            b = _stat(g_blank, cfg.blank_stat)
            s = _stat(g_samp, cfg.blank_stat)
            keep = (b <= 0) or ((s / b) >= cfg.blank_ratio_thresh)

        if keep:
            if not isinstance(feat_batch, tuple):
                feat_batch = (feat_batch,)
            keep_rows.append(dict(zip(group_cols, feat_batch)))

    keep_table = pd.DataFrame(keep_rows).drop_duplicates()
    removed_table = compute_removed_table(universe_table, keep_table, group_cols)

    if keep_table.empty:
        print("[WARN] Batch-aware blank filter kept 0 feature×batch groups.")
        return perfile_df.iloc[0:0].copy(), keep_table, removed_table

    before = len(df)
    filtered = df.merge(keep_table, on=group_cols, how="inner")
    after = len(filtered)

    filtered = filtered.drop(
        columns=[c for c in [key, cfg.blank_col, cfg.batch_col] if c in filtered.columns],
        errors="ignore"
)
    print(
        f"[INFO] Batch-aware blank filter applied — rows {before} -> {after}, "
        f"kept feature×batch={len(keep_table)}, removed feature×batch={len(removed_table)}"
    )
    return filtered, keep_table, removed_table

def _make_universe_table(
    df: pd.DataFrame,
    cols: List[str],
) -> pd.DataFrame:
    """
    Return unique combinations of cols from df as a table.
    """
    if not cols:
        raise ValueError("[ERROR] cols cannot be empty for universe table.")
    return df[cols].drop_duplicates().reset_index(drop=True)


def compute_removed_table(
    universe_table: pd.DataFrame,
    keep_table: pd.DataFrame,
    id_cols: List[str],
) -> pd.DataFrame:
    """
    removed = universe - keep, by id_cols.
    """
    if universe_table is None or universe_table.empty:
        return pd.DataFrame(columns=id_cols)

    if keep_table is None or keep_table.empty:
        # If nothing kept, everything in universe was removed
        return universe_table[id_cols].drop_duplicates().reset_index(drop=True)

    u = universe_table[id_cols].drop_duplicates()
    k = keep_table[id_cols].drop_duplicates()

    removed = (
        u.merge(k, on=id_cols, how="left", indicator=True)
         .query("_merge == 'left_only'")
         .drop(columns=["_merge"])
         .reset_index(drop=True)
    )
    return removed




def flatten_subfolders(run_dir: str | Path, remove_empty: bool = True):
    """
    Merge files from timestamped sibling folders into their base folder.

    Handles names like:
      Adduct_and_summary_outputs_26-04-12_15-53-43
      Adduct_and_summary_outputs_2026-04-12_15-53-43
      CyanoMetDB_matches_out_2026-04-12_15-53-43_RAW
      CyanoMetDB_matches_out_2026-04-12_15-53-43_CLEAN
    """
    run_dir = Path(run_dir)

    if not run_dir.exists():
        print(f"[INFO] merge step skipped; run_dir not found: {run_dir}")
        return

    folders = [p for p in run_dir.iterdir() if p.is_dir()]

    for dup_dir in folders:
        name = dup_dir.name

        m = re.match(
            r"^(.*)_(\d{2,4}-\d{2}-\d{2})_(\d{2}-\d{2}-\d{2})(?:_(RAW|CLEAN))?$",
            name,
        )
        if not m:
            continue

        base_name = m.group(1)
        base_dir = run_dir / base_name

        if not base_dir.exists() or base_dir == dup_dir:
            continue

        print(f"[INFO] Merging:")
        print(f"       from: {dup_dir}")
        print(f"         to: {base_dir}")

        for item in dup_dir.iterdir():
            dest = base_dir / item.name

            if dest.exists():
                stem = dest.stem
                suffix = dest.suffix
                k = 1
                while True:
                    alt = base_dir / f"{stem}_dup{k}{suffix}"
                    if not alt.exists():
                        dest = alt
                        break
                    k += 1

            shutil.move(str(item), str(dest))
            print(f"       moved: {item.name} -> {dest.name}")

        if remove_empty:
            try:
                dup_dir.rmdir()
                print(f"       removed empty folder: {dup_dir}")
            except OSError:
                print(f"       folder not empty, kept: {dup_dir}")