#rt_histograms.py
#!/usr/bin/env python3
"""
rt_histograms.py

Make per-file and merged RT histograms by diagnostic ion, save figures into a
timestamped folder, and export an Excel with labeled ions.

Requirements:
    - pandas
    - matplotlib
    - numpy

Usage:
    python rt_histograms.py input.csv --out-dir plots --label-json ion_labels.json

    Where input.csv should contain at least the columns:
        - rt           (float; retention time in minutes)
        - ion          (float or str; diagnostic ion)
        - source_file  (str; path or name of the source file)
        - (optional) precmz (float; precursor m/z)  # not used for plotting, but kept in Excel

The optional --label-json should be a JSON file mapping numeric ion values to labels, 
If omitted, ions will be formatted numerically to 4 decimals when a mapping isn't found.
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from typing import Dict, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _load_labels(path: str | None) -> Dict[float, str]:
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    labels = {}
    for k, v in raw.items():
        try:
            labels[float(k)] = str(v)
        except Exception:
            # ignore non-numeric keys
            pass
    return labels


def _safe_basename(p: str) -> str:
    base = os.path.splitext(os.path.basename(str(p)))[0]
    # replace weird characters for filenames
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in base)


def _make_output_folder(root: str) -> str:
    ts_folder = datetime.now().strftime("%y-%m-%d_%H-%M-%S")
    out_dir = os.path.join(root, f"rt_hist_{ts_folder}")
    os.makedirs(out_dir, exist_ok=True)
    return out_dir, ts_folder


def _label_ion(x: Any, mapping: Dict[float, str]) -> str:
    # try numeric lookup into mapping; fallback to numeric formatting/string
    try:
        fx = float(x)
        return mapping.get(fx, f"{fx:.4f}")
    except Exception:
        return str(x)


def plot_rt_histograms(ind_hits_l: pd.DataFrame, labels: Dict[float, str], out_dir_root: str) -> str:
    """Create per-file and merged RT histograms and save into a timestamped folder."""
    out_dir, ts_folder = _make_output_folder(out_dir_root)

    # add ion_label column for Excel and legends
    ind_hits_l = ind_hits_l.copy()
    ind_hits_l["ion_label"] = ind_hits_l["ion"].map(lambda x: _label_ion(x, labels))

    # ---------------- Preview by file (printed) ----------------
    grouped = ind_hits_l.groupby("source_file")
    for f, sub in grouped:
        print(f"\nPreview for file: {os.path.basename(str(f))}")
        print(sub.head())

    # Timestamp for filenames (matches requested format)
    ts_file = ts_folder  # already "%y-%m-%d_%H-%M-%S"

    # ---------------- Plot RT histograms per file ----------------
    for f, sub in grouped:
        plt.figure(figsize=(10, 6))
        for ion, ion_sub in sub.groupby("ion"):
            lbl = _label_ion(ion, labels)
            plt.hist(ion_sub["rt"], bins=50, alpha=0.5, label=lbl)
        plt.xlabel("Retention time (min)")
        plt.ylabel("Count of scans")
        plt.title(f"RT distributions by diagnostic ions of cyanopeptide\nFile: {os.path.basename(str(f))}")
        plt.legend()
        plt.tight_layout()

        safe_name = _safe_basename(f)
        out_fig = os.path.join(out_dir, f"RT_distribution_{safe_name}_{ts_file}.png")
        plt.savefig(out_fig, dpi=300, bbox_inches="tight", facecolor="white")
        print(f" Saved per-file figure: {os.path.abspath(out_fig)}")
        plt.close()

    # ---------------- Plot merged (all files) ----------------
    plt.figure(figsize=(10, 6))
    for ion, ion_sub in ind_hits_l.groupby("ion"):
        lbl = _label_ion(ion, labels)
        plt.hist(ion_sub["rt"], bins=50, alpha=0.5, label=lbl)
    plt.xlabel("Retention time (min)")
    plt.ylabel("Count of scans")
    plt.title("RT distributions by diagnostic ions of cyanopeptide (Merged across all files)")
    plt.legend()
    plt.tight_layout()

    out_fig_merged = os.path.join(out_dir, f"RT_distribution_all_files_{ts_file}.png")
    plt.savefig(out_fig_merged, dpi=300, bbox_inches="tight", facecolor="white")
    print(f" Saved merged figure: {os.path.abspath(out_fig_merged)}")
    plt.close()

    # ---------------- Save Excel ----------------
    excel_name = f"rt_distribution_scan_precmz_i_labeled_ions_{ts_file}.xlsx"
    excel_path = os.path.join(out_dir, excel_name)

    # Pick Excel engine safely (xlsxwriter preferred)
    try:
        import xlsxwriter  # noqa: F401
        engine = "xlsxwriter"
    except ImportError:
        print("xlsxwriter not installed — falling back to openpyxl.")
        engine = "openpyxl"

    with pd.ExcelWriter(excel_path, engine=engine) as xw:
        ind_hits_l.to_excel(xw, index=False, sheet_name="ind_hits_labeled")
        # optional: add a quick pivot of counts by file/ion
        pivot = (
            ind_hits_l.assign(count=1)
            .pivot_table(index="source_file", columns="ion_label", values="count", aggfunc="sum", fill_value=0)
        )
        pivot.to_excel(xw, sheet_name="counts_by_file_ion")

    print(f"Saved Excel: {os.path.abspath(excel_path)}")


    return out_dir


def _build_cli():
    ap = argparse.ArgumentParser(description="Plot RT histograms per file and merged; save figures and Excel with timestamp.")
    ap.add_argument("input_csv", help="Path to CSV containing columns: rt, ion, source_file (and optionally precmz)")
    ap.add_argument("--out-dir", default=".", help="Root folder to write outputs (default: current dir)")
    ap.add_argument("--label-json", default=None, help="Optional JSON mapping of ion (as string/number) -> label, e.g. {'18.0': 'NH4+'}")
    return ap


def main(argv=None) -> int:
    parser = _build_cli()
    args = parser.parse_args(argv)

    if not os.path.exists(args.input_csv):
        parser.error(f"Input CSV not found: {args.input_csv}")

    df = pd.read_csv(args.input_csv)

    required = ["rt", "ion", "source_file"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        parser.error(f"Input CSV is missing required columns: {missing}")

    labels = _load_labels(args.label_json)
    out_dir = plot_rt_histograms(df, labels, args.out_dir)
    print(f"All outputs written to: {os.path.abspath(out_dir)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


import os
import glob
import pandas as pd

def load_latest_hits(ind_pattern="individual_hits_*.csv"):
    """
    Automatically find and load the most recent individual  hit CSVs.

    Parameters
    ----------
    ind_pattern : str, default 'individual_hits_*.csv'
        Glob pattern for individual hits CSVs.
    

    Returns
    -------
    (ind_hits_l) : tuple of DataFrames
        Loaded DataFrames for individual  hits.
    """

    def _latest_file(pattern):
        files = glob.glob(pattern)
        if not files:
            print(f" No files found matching pattern: {pattern}")
            return None
        latest = max(files, key=os.path.getctime)
        print(f"Using latest file: {latest}")
        return latest

    ind_path = _latest_file(ind_pattern)
    #combo_path = _latest_file(combo_pattern)

    ind_hits_l = pd.read_csv(ind_path) if ind_path else pd.DataFrame()
    #combo_hits_l = pd.read_csv(combo_path) if combo_path else pd.DataFrame()

    print(f" Loaded {len(ind_hits_l)} individual hits")
    return ind_hits_l
