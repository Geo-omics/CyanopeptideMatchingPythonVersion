# summary_builder.py
import math
import pandas as pd
from pathlib import Path
import numpy as np


def compress_scan_ids(ids, compress=True, force_text=True):
    """Return comma-separated scan ids; optionally compress consecutive ranges."""
    ids = sorted({int(x) for x in ids if pd.notna(x)})
    if not ids:
        out = ""
    elif not compress:
        out = ",".join(str(x) for x in ids)
    else:
        ranges = []
        start = prev = ids[0]
        for s in ids[1:]:
            if s == prev + 1:
                prev = s
            else:
                ranges.append(f"{start}" if start == prev else f"{start}-{prev}")
                start = prev = s
        ranges.append(f"{start}" if start == prev else f"{start}-{prev}")
        out = ",".join(ranges)

    if force_text:
        # Leading apostrophe keeps it as text in Excel
        return "'" + out if out and not out.startswith("'") else out
    return out

def make_scan_level_has_table(
    hits: pd.DataFrame,
    *,
    label_col: str,
    expected_labels: list[str],
) -> pd.DataFrame:
    df = hits.copy()

    key_cols = [c for c in ["source_file", "scan", "ms1scan", "rt", "precmz", "charge"] if c in df.columns]

    # one row per scan, one column per class-specific label
    out = (
        df.assign(value=True)
          .pivot_table(
              index=key_cols,
              columns=label_col,
              values="value",
              aggfunc="max",
              fill_value=False,
          )
          .reset_index()
    )

    out.columns.name = None

    # rename label columns to has_<label>
    rename_map = {lab: f"has_{lab}" for lab in expected_labels if lab in out.columns}
    out = out.rename(columns=rename_map)

    # ensure ALL expected class-specific has_* columns exist
    expected_has_cols = [f"has_{lab}" for lab in expected_labels]
    for c in expected_has_cols:
        if c not in out.columns:
            out[c] = False

    # order columns
    front = key_cols
    rest = [c for c in expected_has_cols if c in out.columns]
    out = out[front + rest]

    return out

def _decimals_from_tol(tol: float) -> int:
    """Infer number of decimal places to round to (e.g., 0.01 -> 2)."""
    if tol <= 0:
        return 2
    return max(0, int(round(-math.log10(tol))))


def assign_feature_clusters(
    mz: pd.Series,
    rt: pd.Series,
    charge: pd.Series,
    mz_tol: float,
    rt_tol: float,
) -> pd.Series:
    """
    Cluster features using BOTH m/z and RT tolerance.
    A new cluster starts if either mz OR RT difference is too large.
    """
    df = pd.DataFrame({
        "mz": mz,
        "rt": rt,
        "charge": charge,
    })

    clusters = pd.Series(index=df.index, dtype=int)
    current = 0

    for ch, sub in df.groupby("charge", sort=False):
        sub = sub.sort_values(["mz", "rt"])

        last_mz = None
        last_rt = None

        for idx, row in sub.iterrows():
            mz_val = row["mz"]
            rt_val = row["rt"]

            if (
                last_mz is None
                or abs(mz_val - last_mz) > mz_tol
                or abs(rt_val - last_rt) > rt_tol
            ):
                current += 1

            clusters.loc[idx] = current
            last_mz = mz_val
            last_rt = rt_val

    return clusters


def make_summary_ind(
    df: pd.DataFrame,
    merge_tol_mz: float = 0.01,
    merge_tol_rt: float = 0.2,
    *,
    compress_scans: bool = False,
    force_text: bool = True,
    ion_to_label: dict | None = None,
) -> pd.DataFrame:
    """
    Summarize hits by precursor m/z within TRUE tolerance (abs delta <= merge_tol_mz)
    AND charge. Keeps track of files and scans that contributed to each precursor.

    Notes:
    - This uses "chaining" tolerance clustering within each charge.
    - Presence/absence flags are OR'ed across all rows in the merged feature.
    """
    if df.empty:
        print("Input dataframe is empty, returning unchanged.")
        return df

    df = df.copy()

    # If no mapping provided, use an empty dict so .get(...) still works
    if ion_to_label is None:
        ion_to_label = {}

    # Build ion_label
    if "ion" in df.columns:

        def _map_label(x):
            if pd.isna(x):
                return ""
            try:
                return ion_to_label.get(float(x), f"{float(x):.4f}")
            except Exception:
                return str(x)

        df["ion_label"] = df["ion"].map(_map_label)

    elif "ion_label" not in df.columns:
        # Ultimate fallback: use precursor m/z as label
        df["ion_label"] = df["precmz"].map(
            lambda x: f"{float(x):.4f}" if pd.notna(x) else ""
        )

    # ---- TRUE tolerance clustering on precursor m/z, separately within each charge ----
    df["_feature_cluster"] = assign_feature_clusters(
        mz=df["precmz"],
        rt=df["rt"],
        charge=df["charge"],
        mz_tol=merge_tol_mz,
        rt_tol=merge_tol_rt,
    )

    # Group by cluster + charge
    grouped = df.groupby(["_feature_cluster", "charge"], as_index=False)

    def collect_info(sub: pd.DataFrame) -> pd.Series:
        return pd.Series(
            {
                "merged_precmz": sub["precmz"].mean(),
                "rt_min": sub["rt"].min(),
                "rt_median": sub["rt"].median(),
                "rt_max": sub["rt"].max(),
                "n_scans": sub["scan"].nunique(),
                "scan_ids": compress_scan_ids(
                    sub["scan"].unique(),
                    compress=compress_scans,
                    force_text=force_text,
                ),
                "ms1_scan_ids": compress_scan_ids(
                    sub["ms1scan"].unique(),
                    compress=compress_scans,
                    force_text=force_text,
                ),
                "files": ",".join(sorted(sub["source_file"].unique())),
            }
        )

    summary = grouped.apply(collect_info).reset_index(drop=True)

    # Presence/absence flags per ion_label PER (cluster, charge)
    presence = (
        df.groupby(["_feature_cluster", "charge", "ion_label"])["scan"]
        .size()
        .unstack(fill_value=0)
        .astype(bool)
        .reset_index()
    )

    # Rename columns to has_<ion_label>
    base_cols = ["_feature_cluster", "charge"]
    new_cols = base_cols + [
        f"has_{c}" for c in presence.columns if c not in base_cols
    ]
    presence.columns = new_cols

    # Merge presence flags back; drop helper col
    summary = (
        summary.merge(presence, on=["_feature_cluster", "charge"], how="left")
        .drop(columns=["_feature_cluster"])
    )

    print(f"make_summary_ind: {len(summary)} merged precursors from {len(df)} rows")
    return summary



# --- MS1 AUC from MS1-point-table CSV---
# Uses your extracted MS1 point table:
#   columns: source_file, scan, rt, precmz, i, mslevel, polarity
# And your matches table:
#   columns: merged_precmz, rt_min, rt_max, files (comma-separated mzML basenames)
#
# Output: per-file exploded matches + ms1_auc per row (and optional pooled sums)


# ---------------- helpers ----------------
def read_table(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)

def explode_matches_to_per_file(matches: pd.DataFrame) -> pd.DataFrame:
    out = matches.copy()

    # normalize filenames list (ensure basenames)
    if "files" in out.columns:
        out["files"] = out["files"].astype(str).fillna("")
        out["files"] = out["files"].apply(
            lambda s: ",".join([Path(x.strip()).name for x in s.split(",") if x.strip()])
        )

    # explode into one file per row
    out["source_file"] = out["files"].astype(str).apply(lambda s: [x.strip() for x in s.split(",") if x.strip()])
    out = out.explode("source_file", ignore_index=True)

    return out


# ---------------- MS1 AUC from points table ----------------
def add_ms1_auc_from_points(
    perfile_df: pd.DataFrame,
    ms1_points_df: pd.DataFrame,
    *,
    tol_mz: float = 0.01,
    rt_pad: float = 0.25,
    polarity: int | None = None,          # 1 / -1 / None (no filter)
    intensity_col: str = "i",             # "i" or "i_norm" or whatever you want to integrate
    mz_col: str = "precmz",               # in your MS1 table we stored m/z in "precmz"
    rt_col: str = "rt",
    file_col: str = "source_file",
) -> pd.DataFrame:
    """
    Input must already be exploded so each row has one source_file.
    Requires columns in perfile_df: merged_precmz, rt_min, rt_max, source_file.
    Requires columns in ms1_points_df: source_file, rt, precmz(or mz_col), i(or intensity_col), mslevel.
    """

    out = perfile_df.copy()

    # numeric safety
    for c in ["merged_precmz", "rt_min", "rt_max"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    # --- prepare MS1 points ---
    pts = ms1_points_df.copy()

    # keep only MS1
    if "mslevel" in pts.columns:
        pts = pts[pts["mslevel"] == 1]

    # optional polarity filter
    if polarity is not None and "polarity" in pts.columns:
        pts = pts[pts["polarity"] == polarity]

    # sanity checks
    required_cols = {file_col, rt_col, mz_col, intensity_col}
    missing = required_cols - set(pts.columns)
    if missing:
        raise ValueError(f"ms1_points_df missing columns: {sorted(missing)}. Have: {list(pts.columns)}")

    # ensure numeric
    pts[rt_col] = pd.to_numeric(pts[rt_col], errors="coerce")
    pts[mz_col] = pd.to_numeric(pts[mz_col], errors="coerce")
    pts[intensity_col] = pd.to_numeric(pts[intensity_col], errors="coerce")

    # drop junk
    pts = pts.dropna(subset=[file_col, rt_col, mz_col, intensity_col])

    # index by file for fast subsetting
    # (groupby lookup is much faster than repeatedly filtering whole df)
    pts_by_file = {k: v for k, v in pts.groupby(file_col, sort=False)}

    aucs = []
    xic_n_points = []   # helpful debug: how many RT points used
    xic_n_rows = []     # how many raw points used

    for _, row in out.iterrows():
        mz0 = row["merged_precmz"]
        rtmin = row["rt_min"]
        rtmax = row["rt_max"]
        fname = str(row[file_col])

        if pd.isna(mz0) or pd.isna(rtmin) or pd.isna(rtmax) or not fname.strip():
            aucs.append(np.nan)
            xic_n_points.append(0)
            xic_n_rows.append(0)
            continue

        mz0 = float(mz0)
        rtmin = float(rtmin)
        rtmax = float(rtmax)

        # pad tiny windows
        if (rtmax - rtmin) < 0.10:
            rtmin -= rt_pad
            rtmax += rt_pad

        sub = pts_by_file.get(fname)
        if sub is None or sub.empty:
            aucs.append(0.0)
            xic_n_points.append(0)
            xic_n_rows.append(0)
            continue

        # filter to mz/rt window
        win = sub[
            sub[mz_col].between(mz0 - tol_mz, mz0 + tol_mz)
            & sub[rt_col].between(rtmin, rtmax)
        ]

        if win.empty:
            aucs.append(0.0)
            xic_n_points.append(0)
            xic_n_rows.append(0)
            continue

        # build XIC trace: sum intensity per RT and integrate
        trace = (
            win.groupby(rt_col, as_index=False)[intensity_col]
               .sum()
               .sort_values(rt_col)
        )

        x = trace[rt_col].to_numpy(dtype=float)
        y = trace[intensity_col].to_numpy(dtype=float)

        # IMPORTANT: need at least 2 RT points for trapezoid area
        if len(x) < 2:
            aucs.append(0.0)
        else:
            aucs.append(float(np.trapezoid(y, x)))

        xic_n_points.append(int(len(x)))
        xic_n_rows.append(int(len(win)))

    out["ms1_auc"] = aucs
    out["xic_n_rt_points"] = xic_n_points
    out["xic_n_raw_points"] = xic_n_rows

    return out


# ---------------- pooled sum back to match-level ----------------
def pool_auc_back_to_matches(perfile_with_auc: pd.DataFrame) -> pd.DataFrame:
    """
    If you want the old 'sum across files' version:
      - group by the match identity columns and sum ms1_auc across files.
    You can adjust the group keys depending on what uniquely identifies a match.
    """
    # reasonable default keys for your matches table:
    keys = ["merged_precmz", "rt_min", "rt_median", "rt_max", "scan_ids", "ms1_scan_ids", "files"]
    keys = [k for k in keys if k in perfile_with_auc.columns]

    pooled = (
        perfile_with_auc
        .groupby(keys, as_index=False)["ms1_auc"]
        .sum()
        .rename(columns={"ms1_auc": "ms1_auc_sum_across_files"})
    )
    return pooled

#extract date time from the MS1 points generated
import re
from datetime import datetime
from pathlib import Path
def extract_created_time(path: Path) -> datetime:
    """
    Extract datetime from:
    ms1_points_YYYY-MM-DD_HH-MM-SS.csv
    """
    match = re.search(
        r"ms1_points_(\d{4}-\d{2}-\d{2})_(\d{2}-\d{2}-\d{2})",
        path.name,
    )
    if not match:
        raise ValueError(f"[ERROR] Invalid MS1 filename format: {path.name}")

    date_part, time_part = match.groups()
    timestamp_str = f"{date_part} {time_part.replace('-', ':')}"
    return datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
