# ms2_tilemap_intensities.py

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -----------------------------------------------------
# Helper: find newest path matching a glob pattern
# -----------------------------------------------------
def find_latest(pattern, what="item"):
    items = glob.glob(pattern)
    if not items:
        raise FileNotFoundError(f"No {what} found for pattern: {pattern}")
    latest = max(items, key=os.path.getmtime)
    print(f"Using most recent {what}: {latest}")
    return latest


# -----------------------------------------------------
# Load latest indiv_merged_summary_with_intensities_* in latest adduct_outputs_*
# -----------------------------------------------------
def load_latest_summary_with_intensities(
    run_dir_pattern="adduct_outputs_*",
    summary_basename_pattern="indiv_merged_summary_with_intensities_*.csv",
):
    """
    Find the newest adduct_outputs_* directory, then inside it the newest
    indiv_merged_summary_with_intensities_*.csv, and load it as a DataFrame.
    """
    # newest adduct_outputs_* directory
    run_dir = find_latest(run_dir_pattern, what="adduct_outputs directory")

    # newest indiv_merged_summary_with_intensities_* inside that directory
    summary_pattern = os.path.join(run_dir, summary_basename_pattern)
    summary_file = find_latest(summary_pattern, what="summary_with_intensities file")

    df = pd.read_csv(summary_file)
    return df, run_dir, summary_file


# -----------------------------------------------------
# Tile map of has_* columns shaded by log_i_sum
# -----------------------------------------------------
def plot_has_tilemap_from_latest(ion_to_label: dict | None = None):
    """
    Make a tile map where:
      - rows = precursor m/z (merged_precmz)
      - columns = has_* columns in the latest indiv_merged_summary_with_intensities_*.csv
      - cell color = normalized log_i_sum (light blue = low, dark blue = high)
      - cells where has_* is False are white

    If ion_to_label is provided (mapping m/z -> label), it will be used to make
    prettier x-axis labels when has_* columns are numeric (e.g., has_184.06).
    """
    df, run_dir, summary_file = load_latest_summary_with_intensities()

    print(f"\nLoaded {summary_file}")

    # detect has_* columns (has_A, has_B, has_..., etc.)
    has_cols = [c for c in df.columns if "has_" in c]
    if not has_cols:
        raise ValueError("No columns containing 'has_' were found in the dataframe.")
    print(f"Using has_* columns: {has_cols}")

    # precursor m/z column: adjust if your column name is different
    if "merged_precmz" not in df.columns:
        raise ValueError("Column 'merged_precmz' not found (needed for MS1 axis).")
    mz_vals = df["merged_precmz"]

    # Ensure we have log_i_sum; if not, compute from i_sum
    if "log_i_sum" not in df.columns:
        if "i_sum" not in df.columns:
            raise ValueError("Neither 'log_i_sum' nor 'i_sum' found in dataframe.")
        df["log_i_sum"] = df["i_sum"].apply(lambda x: np.log(x) if x > 0 else np.nan)

    # Normalize log_i_sum for shading
    log_vals = df["log_i_sum"].copy()
    valid = log_vals.notna()

    if valid.any():
        min_val = log_vals[valid].min()
        max_val = log_vals[valid].max()
        
        if max_val > min_val:
            raw_norm = (log_vals - min_val) / (max_val - min_val)
        else:
            raw_norm = pd.Series(1.0, index=df.index)  # all equal
        
        # Rescale TRUE values into [0.2, 1.0]
        norm_log = 0.2 + 0.8 * raw_norm
    else:
        # no intensities
        norm_log = pd.Series(np.nan, index=df.index)

    # Build matrix:
    n_rows = len(df)
    n_cols = len(has_cols)
    M = np.zeros((n_rows, n_cols), dtype=float)

    for j, col in enumerate(has_cols):
        col_bool = df[col].astype(bool)
        M[col_bool.values, j] = norm_log[col_bool].fillna(0).values

    # Mask zeros (False) so they appear white
    M_masked = np.ma.masked_where(M == 0, M)

    fig, ax = plt.subplots(figsize=(12, 8))

    # --- BLUE COLORMAP (light = low, dark = high) ---
    cmap = plt.cm.Blues
    cmap = cmap.copy()
    cmap.set_bad("white")  # hidden cells become white

    im = ax.imshow(M_masked, aspect="auto", cmap=cmap, interpolation="nearest")

    ax.set_title("has_* tile map (blue shading by log_i_sum)\nrows = precursor m/z")
    ax.set_xlabel("has_* columns")
    ax.set_ylabel("precursor m/z (MS1 feature)")

    # X: has_* columns → optionally map to pretty labels using ion_to_label
    ax.set_xticks(range(n_cols))
    if ion_to_label is not None:
        display_labels = []
        for col in has_cols:
            base = col.replace("has_", "", 1)
            label = col  # default
            try:
                mz = float(base)
                label = ion_to_label.get(mz, col)
            except Exception:
                pass
            display_labels.append(label)
    else:
        display_labels = has_cols

    ax.set_xticklabels(display_labels, rotation=90)

    # Y: precursor m/z labels
    if n_rows > 50:
        step = max(1, n_rows // 50)
        y_indices = np.arange(0, n_rows, step)
    else:
        y_indices = np.arange(n_rows)

    ax.set_yticks(y_indices)
    ax.set_yticklabels(np.round(mz_vals.iloc[y_indices], 4))

    # Colorbar: blue scale
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("normalized log_i_sum\n(light = low, dark = high)")

    plt.tight_layout()
    plt.show()

    # --- Save here ---
    run_tag = os.path.basename(run_dir)
    filename = f"Cyanopeptide_detection_intensity_heatmap_{run_tag}.png"
    out_path = os.path.join(run_dir, filename)
    fig.savefig(out_path, dpi=300)
    print(f"Saved figure → {out_path}")


    return df, run_dir, summary_file

