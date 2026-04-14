#indiv_combo_dot_plot.py
"""
indiv_combo_dot_plots.py

  1) Individual diagnostic-ion presence per precursor (indiv summary)
  2) Combo diagnostic-ion presence per precursor (combo summary)


As a module:
  from indiv_combo_plots import plot_indiv_scatter, plot_combo_scatter
  fig, ax, path = plot_indiv_scatter(indiv_df, out_dir="plots")
  fig, ax, path = plot_combo_scatter(combo_df, out_dir="plots")
"""
from __future__ import annotations

import os
from datetime import datetime
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# -------------------------
# Individual (per-ion flags)
# -------------------------
def plot_indiv_scatter(
    indiv_summary: pd.DataFrame,
    *,
    scan_col: str = "n_scans",
    rt_col: str = "rt_median",
    mz_col: str = "merged_precmz",
    source_file_col: str = "files",
    palette: str = "viridis",
    title: str = "Indiv. Diagnostic ions detected in precursor m/z",
    out_dir: str = ".",
    fmt: str = "png",
    show: bool = False,
) -> Tuple[plt.Figure, plt.Axes, Optional[str]]:
    """
    Build and plot a scatter of diagnostic-ion presence (has_*) vs precursor m/z,
    scaling dot sizes by scan count. Returns (fig, ax, saved_path).
    """


    if indiv_summary is None or indiv_summary.empty:
        print("indiv_summary is empty — skipping plot.")
        return None, None, None

    # Required columns
    id_vars = [mz_col, rt_col, scan_col, source_file_col]
    missing = [c for c in id_vars if c not in indiv_summary.columns]
    if missing:
        raise KeyError(f"indiv_summary missing columns: {missing}")

    # has_* columns
    has_cols = [c for c in indiv_summary.columns if str(c).startswith("has_")]
    if not has_cols:
        raise ValueError("No 'has_*' columns found in indiv_summary")

    # melt
    melted = indiv_summary.melt(
        id_vars=id_vars,
        value_vars=has_cols,
        var_name="frag_name",
        value_name="present",
    )

    # keep True rows and clean names
    melted = melted[melted["present"] == True].copy()
    if melted.empty:
        raise ValueError("No True flags found after melting indiv_summary")
    melted["frag_name"] = melted["frag_name"].str.replace(r"^has_", "", regex=True)

    # scale sizes by scan count
    max_scans = pd.to_numeric(melted[scan_col], errors="coerce").fillna(0).max()
    if max_scans <= 0:
        sizes = np.full(len(melted), 20.0)
    else:
        sizes = 20 + 50 * (melted[scan_col] / max_scans)

    # plot
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.scatterplot(
        data=melted,
        x=mz_col,
        y="frag_name",
        size=sizes,
        sizes=(20, 300),
        hue=source_file_col,
        palette=palette,
        legend="brief",
        edgecolor="k",
        ax=ax
    )

    ax.set_xlabel("Precursor m/z")
    ax.set_ylabel("Diagnostic ion")
    ax.set_title(title)
    fig.tight_layout()

    # save with timestamp
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_fig = os.path.join(out_dir, f"indiv_diagnostic_ions_{ts}.{fmt}")
    fig.savefig(out_fig, dpi=300, bbox_inches="tight", facecolor="white")
    saved_path = os.path.abspath(out_fig)
    print(f"Saved figure: {saved_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig, ax, saved_path


# -------------------------
# Combo (combined flags)
# -------------------------
def plot_combo_scatter(
    combo_summary: pd.DataFrame,
    *,
    scan_col: str = "n_scans",
    rt_col: str = "rt_median",
    mz_col: str = "merged_precmz",
    source_file_col: str = "files",
    palette: str = "viridis",
    title: str = "Combo Diagnostic ions detected in precursor m/z",
    out_dir: str = ".",
    fmt: str = "png",
    show: bool = False,
) -> Tuple[plt.Figure, plt.Axes, Optional[str]]:
    """
    Build and plot a scatter of combo 'has_*' flags vs precursor m/z,
    scaling dot sizes by scan count. Returns (fig, ax, saved_path).
    """
    if combo_summary is None or combo_summary.empty:
        print("combo_summary is empty — skipping plot.")
        return None, None, None


    needed = [scan_col, rt_col, mz_col, source_file_col]
    missing = [c for c in needed if c not in combo_summary.columns]
    if missing:
        raise KeyError(f"combo_summary missing required columns: {missing}")

    has_cols = [c for c in combo_summary.columns if str(c).startswith("has_")]
    if not has_cols:
        raise ValueError("No 'has_*' columns found in combo_summary")

    melted = combo_summary.melt(
        id_vars=[mz_col, rt_col, scan_col, source_file_col],
        value_vars=has_cols,
        var_name="combo_label",
        value_name="present"
    )

    melted = melted[melted["present"] == True].copy()
    if melted.empty:
        raise ValueError("No True flags found after melting combo_summary")

    melted["combo_label"] = melted["combo_label"].str.replace(r"^has_", "", regex=True)

    max_scans = pd.to_numeric(melted[scan_col], errors="coerce").fillna(0).max()
    if max_scans <= 0:
        sizes = np.full(len(melted), 20.0)
    else:
        sizes = 20 + 50 * (melted[scan_col] / max_scans)

    melted[mz_col] = pd.to_numeric(melted[mz_col], errors="coerce")
    if melted[mz_col].dropna().empty:
        raise ValueError("No valid numeric precursor m/z values to plot.")

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.scatterplot(
        data=melted,
        x=mz_col, y="combo_label",
        size=sizes, sizes=(20, 300),
        hue=source_file_col, palette=palette, legend="brief", edgecolor="k", ax=ax
    )
    ax.set_xlabel("Precursor m/z")
    ax.set_ylabel("Diagnostic ion")
    ax.set_title(title)
    fig.tight_layout()

    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_fig = os.path.join(out_dir, f"combo_diagnostic_ions_{ts}.{fmt}")
    fig.savefig(out_fig, dpi=300, bbox_inches="tight", facecolor="white")
    saved_path = os.path.abspath(out_fig)
    print(f"Saved figure: {saved_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig, ax, saved_path


