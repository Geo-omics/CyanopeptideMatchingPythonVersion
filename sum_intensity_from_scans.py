
"""
sum_intensity_from_scans.py

- Find latest indiv_merged_summary_*.csv and individual_hits_*.csv
- Explode scan_ids into individual scans
- Join MS2 intensity ("i") from hits file
- Sum intensity per MS1 feature
- Save output with timestamp
- Provides an importable function: sum_intensities()
"""

import pandas as pd
import numpy as np
import glob
import os
from datetime import datetime


# -----------------------------------------------------
# 1. Find newest file
# -----------------------------------------------------
def find_latest(pattern):
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"No files found for pattern: {pattern}")
    latest_file = max(files, key=os.path.getmtime)
    print(f"Using most recent file: {latest_file}")
    return latest_file


# -----------------------------------------------------
# 2. Split scan_ids into integer list
# -----------------------------------------------------
def split_scan_ids(s):
    if pd.isna(s):
        return []
    s = str(s).lstrip("'")
    parts = [p.strip() for p in s.split(",")]
    out = []
    for p in parts:
        if p.replace(".", "", 1).isdigit():
            out.append(int(float(p)))
    return out


# -----------------------------------------------------
# 3. Main function (IMPORTABLE)
# -----------------------------------------------------
def sum_intensities(
    run_dir_pattern="Adduct_and_summary_outputs_*",
    summary_pattern="indiv_merged_summary_*.csv",
    hits_pattern="individual_hits_*.csv",
):
    """
    Find the newest Adduct_and_summary__outputs_* folder, then:
      - inside that folder: the newest indiv_merged_summary_*.csv
      - in the *current directory*: the newest individual_hits_*.csv

    Compute summed intensities per summary row and write a new CSV.
    Returns (out_df, outfile_path).
    """
    # 1) newest Adduct_and_summary__outputs_* directory
    run_dir = find_latest(run_dir_pattern)   # this will be a directory path
    print(f"Using newest adduct run dir: {run_dir}")

    # 2) newest summary within that directory
    summary_file = find_latest(os.path.join(run_dir, summary_pattern))

    # 3) newest hits file in the same directory as Adduct_and_summary_outputs_* folders
    #    (i.e. current working directory)
    hits_file = find_latest(hits_pattern)

    summary = pd.read_csv(summary_file)
    hits = pd.read_csv(hits_file)

    # assign unique row id
    summary["_row_id"] = np.arange(len(summary))

    # explode scan_ids
    exploded = summary[["_row_id", "scan_ids"]].copy()
    exploded["scan"] = exploded["scan_ids"].apply(split_scan_ids)
    exploded = exploded.explode("scan", ignore_index=True)
    exploded = exploded.dropna(subset=["scan"])

    # attach intensity
    exploded = exploded.merge(hits[["scan", "i"]], on="scan", how="left")

    # sum per row
    i_sum = (
        exploded.groupby("_row_id", as_index=False)["i"]
        .sum(min_count=1)
        .rename(columns={"i": "i_sum"})
    )
    i_sum["i_sum"] = i_sum["i_sum"].fillna(0)

    # output merged dataset
    out = summary.merge(i_sum, on="_row_id", how="left").drop(columns="_row_id")
    out["log_i_sum"] = out["i_sum"].apply(lambda x: np.log(x) if x > 0 else np.nan)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    outfile = os.path.join(
        run_dir,
        f"indiv_merged_summary_with_intensities_{timestamp}.csv"
    )
    out.to_csv(outfile, index=False)

    print(f"\nSaved: {outfile}\n")
    print(out[["merged_precmz", "n_scans", "scan_ids", "i_sum"]].head(10))

    return out, outfile
