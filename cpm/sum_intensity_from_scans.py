
"""
sum_intensity_from_scans.py

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
import pandas as pd
import numpy as np
import os
from datetime import datetime


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


def sum_intensities(summary_file, hits_file, output_dir=None):
    if not os.path.isfile(summary_file):
        raise FileNotFoundError(f"Summary file not found: {summary_file}")

    if not os.path.isfile(hits_file):
        raise FileNotFoundError(f"Hits file not found: {hits_file}")

    if output_dir is None:
        output_dir = os.path.dirname(summary_file)

    print(f"Using summary file: {summary_file}")
    print(f"Using hits file: {hits_file}")
    print(f"Output directory: {output_dir}")

    summary = pd.read_csv(summary_file)
    hits = pd.read_csv(hits_file)

    summary["_row_id"] = np.arange(len(summary))

    exploded = summary[["_row_id", "scan_ids"]].copy()
    exploded["scan"] = exploded["scan_ids"].apply(split_scan_ids)
    exploded = exploded.explode("scan", ignore_index=True)
    exploded = exploded.dropna(subset=["scan"])

    exploded = exploded.merge(hits[["scan", "i"]], on="scan", how="left")

    i_sum = (
        exploded.groupby("_row_id", as_index=False)["i"]
        .sum(min_count=1)
        .rename(columns={"i": "i_sum"})
    )
    i_sum["i_sum"] = i_sum["i_sum"].fillna(0)

    out = summary.merge(i_sum, on="_row_id", how="left").drop(columns="_row_id")
    out["log_i_sum"] = out["i_sum"].apply(lambda x: np.log10(x) if x > 0 else np.nan)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    outfile = os.path.join(
        output_dir,
        f"indiv_merged_summary_with_intensities_{timestamp}.csv"
    )
    out.to_csv(outfile, index=False)

    print(f"\nSaved: {outfile}\n")
    print(out[["merged_precmz", "n_scans", "scan_ids", "i_sum"]].head(10))

    return out, outfile

