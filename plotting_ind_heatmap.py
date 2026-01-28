# plotting_ind_heatmap.py
import os
import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd


def plot_heatmaps(
    df: pd.DataFrame,
    ion_to_label: dict,
    BIN_WIDTH: float = 0.1,
    merged: bool = True,
    per_file: bool = True,
    save: bool = False,
    out_dir: str = ".",
    fmt: str = "png",
):
    """
    Plot merged and/or per-file heatmaps of counts by diagnostic ion vs precursor m/z.
    Optionally save each figure with a timestamped filename.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with columns ['ion', 'precmz', 'source_file'].
    ion_to_label : dict
        Mapping from ion m/z (float) to human-readable labels.
    BIN_WIDTH : float, optional
        Width of precursor m/z bins.
    merged : bool, default True
        Whether to plot a merged heatmap across all files.
    per_file : bool, default True
        Whether to plot per-file heatmaps.
    save : bool, default False
        Whether to save the generated plots.
    out_dir : str, default "."
        Directory to save plots in (created if missing).
    fmt : str, default "png"
        File format for saving (e.g., 'png', 'pdf', 'svg').
    """
    df = df.copy()

    # Ensure readable ion labels
    if "ion_label" not in df.columns:
        df["ion_label"] = df["ion"].map(
            lambda x: ion_to_label.get(float(x), f"{float(x):.4f}")
        )

    # Bin precursor m/z
    bin_start = np.floor(df["precmz"] / BIN_WIDTH) * BIN_WIDTH
    df["precmz_bin"] = (bin_start + BIN_WIDTH / 2.0).round(4)

    # Ensure output directory exists if saving
    if save and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    def _plot_heat(sub, title, which, tag):
        heat = (
            sub.groupby(["ion_label", "precmz_bin"])
            .size()
            .rename("count")
            .reset_index()
        )
        mat = (
            heat.pivot(index="ion_label", columns="precmz_bin", values="count")
            .fillna(0)
            .sort_index(axis=1)
        )

        values = mat.values
        vmax = np.max(values)
        raw_bounds = [0, 1, 5, 10, vmax + 1]
        bounds = [b for i, b in enumerate(raw_bounds) if i == 0 or b > raw_bounds[i - 1]]

        norm = mcolors.BoundaryNorm(bounds, ncolors=len(bounds) - 1)
        cmap = mcolors.ListedColormap(["white", "lightblue", "royalblue", "navy"])

        print(f"Plotting heatmap: {which}")

        plt.figure(figsize=(max(10, 0.5 * len(mat.columns)), max(4, 0.5 * len(mat))))
        im = plt.imshow(values, aspect="auto", cmap=cmap, norm=norm)

        cbar = plt.colorbar(im, ticks=bounds)
        cbar.set_label("Count bins")

        plt.yticks(range(len(mat.index)), mat.index)
        plt.xticks(
            range(len(mat.columns)),
            [f"{c:.4f}" for c in mat.columns],
            rotation=90,
        )

        plt.xlabel(f"Precursor m/z (binned, width={BIN_WIDTH} Da)")
        plt.ylabel("Diagnostic ion")
        plt.title(title)

        plt.tight_layout()

        # Save step with timestamp
        if save:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_tag = tag.replace(" ", "_").replace(":", "_")
            filename = f"heatmap_{safe_tag}_{timestamp}.{fmt}"
            out_path = os.path.join(out_dir, filename)
            plt.savefig(out_path, dpi=300)
            print(f"Saved figure → {out_path}")

        plt.show()

    # Plot merged
    if merged:
        _plot_heat(
            df,
            "Counts by diagnostic ion vs precursor m/z\n(Merged across all files)",
            "Merged",
            "merged",
        )

    # Plot per-file
    if per_file and "source_file" in df.columns:
        for f, sub in df.groupby("source_file"):
            tag = os.path.basename(f)
            _plot_heat(
                sub,
                f"Counts by diagnostic ion vs precursor m/z\nFile: {os.path.basename(f)}",
                f"File: {os.path.basename(f)}",
                tag,
            )



