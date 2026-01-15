# summary_builder.py
import math
import pandas as pd


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


def _decimals_from_tol(tol: float) -> int:
    """Infer number of decimal places to round to (e.g., 0.01 -> 2)."""
    if tol <= 0:
        return 2
    return max(0, int(round(-math.log10(tol))))


def assign_mz_clusters(mz: pd.Series, tol: float) -> pd.Series:
    """
    Assign cluster IDs to m/z values using a simple sorted "chaining" rule:
    start a new cluster when the gap from the previous value exceeds tol.
    """
    order = mz.sort_values().index
    clusters = pd.Series(index=mz.index, dtype=int)

    current = 0
    last_mz = None

    for i in order:
        val = mz.loc[i]
        if last_mz is None or abs(val - last_mz) > tol:
            current += 1
        clusters.loc[i] = current
        last_mz = val

    return clusters


def make_summary_ind(
    df: pd.DataFrame,
    merge_tol_mz: float = 0.01,
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
    df["_mz_cluster"] = (
        df.groupby("charge", group_keys=False)["precmz"]
        .apply(lambda s: assign_mz_clusters(s, merge_tol_mz))
    )

    # Group by cluster + charge
    grouped = df.groupby(["_mz_cluster", "charge"], as_index=False)

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
        df.groupby(["_mz_cluster", "charge", "ion_label"])["scan"]
        .size()
        .unstack(fill_value=0)
        .astype(bool)
        .reset_index()
    )

    # Rename columns to has_<ion_label>
    base_cols = ["_mz_cluster", "charge"]
    new_cols = base_cols + [
        f"has_{c}" for c in presence.columns if c not in base_cols
    ]
    presence.columns = new_cols

    # Merge presence flags back; drop helper col
    summary = (
        summary.merge(presence, on=["_mz_cluster", "charge"], how="left")
        .drop(columns=["_mz_cluster"])
    )

    print(f"make_summary_ind: {len(summary)} merged precursors from {len(df)} rows")
    return summary



def make_summary_combo(df: pd.DataFrame, merge_tol_mz: float = 0.01) -> pd.DataFrame:
    """
    Summarize hits by precursor m/z within tolerance AND ion_label identity.
    Keeps track of files and scans that contributed to each precursor.
    """

    if df.empty:
        print("Input dataframe is empty, returning unchanged.")
        return df

    # Round precursors to a bin so "close enough" precursors merge
    df = df.copy()
    df["_mz_bin"] = df["precmz"].round(2)  # adjust precision if needed

    # Group by m/z bin + ion identity + charge
    grouped = df.groupby(["_mz_bin", "combo_label", "charge"], as_index=False)

    def collect_info(sub):
        return pd.Series({
            "merged_precmz": sub["precmz"].mean(),
            "rt_min": sub["rt"].min(),
            "rt_median": sub["rt"].median(),
            "rt_max": sub["rt"].max(),
            "n_scans": sub["scan"].nunique(),
            "scan_ids": ",".join(map(str, sorted(sub["scan"].unique()))),
            "files": ",".join(sorted(sub["source_file"].unique())),
        })

    summary_combo = grouped.apply(collect_info).reset_index(drop=True)

    # Add presence/absence flags per combo_label
    presence = (
        df.groupby(["_mz_bin", "combo_label"])["scan"]
          .size().unstack(fill_value=0).astype(bool).reset_index()
    )
    presence.columns = ["_mz_bin"] + [f"has_{c}" for c in presence.columns if c != "_mz_bin"]

    # Merge presence flags back (safe, no cluster_id needed)
    summary_combo_merge = summary_combo.merge(presence, on="_mz_bin", how="left")

    # Drop helper col
    summary_combo_merge = summary_combo_merge.drop(columns=["_mz_bin"])

    print(f"make_summary_combo: {len(summary_combo_merge)} merged precursors from {len(df)} rows")
    print(summary_combo_merge.head())

    return summary_combo_merge

#01/04/2025: added this to calculate auc of each metabolite
#01/04/2025: MS1 AUC per metabolite + per-file normalization + multi-file pooled sum
# --- MS1 AUC from MS1-point-table CSV (no mzML reread) ---
# Uses your extracted MS1 point table:
#   columns: source_file, scan, rt, precmz, i, mslevel, polarity
# And your matches table:
#   columns: merged_precmz, rt_min, rt_max, files (comma-separated mzML basenames)
#
# Output: per-file exploded matches + ms1_auc per row (and optional pooled sums)

from pathlib import Path
import numpy as np
import pandas as pd


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
            aucs.append(float(np.trapz(y, x)))

        xic_n_points.append(int(len(x)))
        xic_n_rows.append(int(len(win)))

    out["ms1_auc"] = aucs
    out["xic_n_rt_points"] = xic_n_points
    out["xic_n_raw_points"] = xic_n_rows

    return out


# ---------------- optional: pooled sum back to match-level ----------------
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

