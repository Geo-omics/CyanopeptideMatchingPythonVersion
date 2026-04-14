#!/usr/bin/env python3
"""
cyanopeptide_counts_plots.py

Plot bar charts of diagnostic ion counts (individual cyanopeptide labels)
by source_file, with skipping if inputs are empty or missing.

Features:
- Works as a module (import functions and pass DataFrames)
- Timestamped filenames in the format %y-%m-%d_%H-%M-%S (e.g., 25-11-11_13-28-51)
- Saves figures to an output directory you choose (default: current directory)

Requirements:
    pandas, matplotlib

Example (Python):
    from mp_counts_plots import plot_indiv_counts, plot_combo_counts
    plot_indiv_counts(ind_hits_l, out_dir="plots")
    plot_combo_counts(combo_hits_l, out_dir="plots")

"""
from __future__ import annotations

import argparse
import os
from datetime import datetime
from typing import Optional, Tuple

import pandas as pd
import matplotlib.pyplot as plt


def _timestamp() -> str:
    """Return timestamp 'yy-mm-dd_HH-MM-SS' like '25-11-11_13-28-51'."""
    return datetime.now().strftime("%y-%m-%d_%H-%M-%S")


def _safe_outdir(out_dir: str) -> str:
    """Append a timestamp to the output folder name and create it."""
    from datetime import datetime
    import os

    ts = datetime.now().strftime("%y-%m-%d_%H-%M-%S")
    out_dir_ts = f"{out_dir}_{ts}"
    os.makedirs(out_dir_ts, exist_ok=True)
    return out_dir_ts



def _check_df(df: Optional[pd.DataFrame], label_col: str) -> Tuple[bool, str]:
    """Validate df presence and required label column; return (ok, message)."""
    if df is None or df.empty:
        return False, "No data found (empty DataFrame)."
    if label_col not in df.columns:
        return False, f"No '{label_col}' column found."
    if "source_file" not in df.columns:
        return False, "No 'source_file' column found."
    return True, ""


def plot_indiv_counts(
    ind_hits_l: Optional[pd.DataFrame],
    *,
    label_col: Optional[str] = None,
    out_dir: str = ".",
    show: bool = False,
) -> Tuple[Optional[str], Optional[str]]:

    if ind_hits_l is None or ind_hits_l.empty:
        print("No data provided.")
        return None, None

    # Auto-detect any column containing "CyanopeptideClass_"
    if label_col is None:
        matches = [
            c for c in ind_hits_l.columns
            if "cyanopeptideclass_" in c.lower()
        ]

        if not matches:
            raise ValueError("No column containing 'CyanopeptideClass_' found.")

        if len(matches) > 1:
            print(f"Multiple matches found, using first: {matches}")

        label_col = matches[0]
        print(f"Detected label column: {label_col}")

    # ---- Your plotting logic continues here ----

    """
    Plot bar charts for individual diagnostic ions by file.

    Returns (bar_path, stacked_path); paths may be None if skipped.
    """
    ok, msg = _check_df(ind_hits_l, label_col)
    if not ok:
        print(f"No individual diagnostic ions found — {msg}")
        return None, None

    out_dir = _safe_outdir(out_dir)
    ts = _timestamp()

    df = ind_hits_l.copy()

    # Group: counts per file and label
    counts = (
        df.groupby(["source_file", label_col])
          .size().rename("hits").reset_index()
    )

    # Sort and preview top counts
    preview = counts.sort_values(["source_file", "hits"], ascending=[True, False]).head(20)
    print("Top counts (individual):")
    print(preview)

    # Pivot for plotting
    pivot = counts.pivot(index=label_col, columns="source_file", values="hits").fillna(0)

    # 1) Unstacked bars (side-by-side)
    ax = pivot.plot(kind="bar", figsize=(12, 6))
    ax.set_ylabel("Hits")
    ax.set_title("Counts by diagnostic ion and file (individual)")
    plt.tight_layout()
    bar_path = os.path.join(out_dir, f"Diagnostic_ion_distribution_individual_{ts}.png")
    plt.savefig(bar_path, dpi=300, bbox_inches="tight", facecolor="white")
    print(f" Saved individual (unstacked) figure: {os.path.abspath(bar_path)}")
    if show:
        plt.show()
    else:
        plt.close()

    # 2) Stacked bars
    ax = pivot.plot(kind="bar", stacked=True, figsize=(12, 6))
    ax.set_ylabel("Hits")
    ax.set_title("Counts by Cyanopeptide diagnostic ion (files stacked) — individual")
    plt.tight_layout()
    stacked_path = os.path.join(out_dir, f"Diagnostic_ion_distribution_individual_stacked_{ts}.png")
    plt.savefig(stacked_path, dpi=300, bbox_inches="tight", facecolor="white")
    print(f" Saved individual (stacked) figure: {os.path.abspath(stacked_path)}")
    if show:
        plt.show()
    else:
        plt.close()

    return bar_path, stacked_path


def plot_combo_counts(
    combo_hits_l: Optional[pd.DataFrame],
    *,
    label_col: str = "MP_combo",
    out_dir: str = ".",
    show: bool = False,
) -> Tuple[Optional[str], Optional[str]]:
    """
    Plot bar charts for combo diagnostic ions (MP_combo) by file.

    Returns (bar_path, stacked_path); paths may be None if skipped.
    """
    ok, msg = _check_df(combo_hits_l, label_col)
    if not ok:
        print(f"No combo diagnostic ions found — {msg}")
        return None, None

    out_dir = _safe_outdir(out_dir)
    ts = _timestamp()

    df = combo_hits_l.copy()

    counts = (
        df.groupby(["source_file", label_col])
          .size().rename("hits").reset_index()
    )

    preview = counts.sort_values(["source_file", "hits"], ascending=[True, False]).head(20)
    print("Top counts (combo):")
    print(preview)

    pivot = counts.pivot(index=label_col, columns="source_file", values="hits").fillna(0)

    # 1) Unstacked bars
    ax = pivot.plot(kind="bar", figsize=(12, 6))
    ax.set_ylabel("Hits")
    ax.set_title("Counts by cyanopeptide combo/diagnostic ion and file (combo)")
    plt.tight_layout()
    bar_path = os.path.join(out_dir, f"Diagnostic_ion_distribution_combo_{ts}.png")
    plt.savefig(bar_path, dpi=300, bbox_inches="tight", facecolor="white")
    print(f" Saved combo (unstacked) figure: {os.path.abspath(bar_path)}")
    if show:
        plt.show()
    else:
        plt.close()

    # 2) Stacked bars
    ax = pivot.plot(kind="bar", stacked=True, figsize=(12, 6))
    ax.set_ylabel("Hits")
    ax.set_title("Counts by cyanopeptide combo/diagnostic ion (files stacked) — combo")
    plt.tight_layout()
    stacked_path = os.path.join(out_dir, f"Diagnostic_ion_distribution_combo_stacked_{ts}.png")
    plt.savefig(stacked_path, dpi=300, bbox_inches="tight", facecolor="white")
    print(f" Saved combo (stacked) figure: {os.path.abspath(stacked_path)}")
    if show:
        plt.show()
    else:
        plt.close()

    return bar_path, stacked_path
