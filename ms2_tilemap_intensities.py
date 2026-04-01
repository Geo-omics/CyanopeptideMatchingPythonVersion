# ms2_tilemap_intensities.py

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_has_tilemap(summary_file, output_dir=None, ion_to_label=None):
    """
    Make a tile map where:
      - rows = precursor m/z (merged_precmz)
      - columns = has_* columns in the provided summary file
      - cell color = normalized log_i_sum
      - cells where has_* is False are white
    """
    if not os.path.isfile(summary_file):
        raise FileNotFoundError(f"Summary file not found: {summary_file}")

    if output_dir is None:
        output_dir = os.path.dirname(summary_file)

    print(f"\nLoaded summary file: {summary_file}")
    print(f"Output directory: {output_dir}")

    df = pd.read_csv(summary_file)

    has_cols = [c for c in df.columns if c.startswith("has_")]
    if not has_cols:
        raise ValueError("No columns starting with 'has_' were found in the dataframe.")
    print(f"Using has_* columns: {has_cols}")

    if "merged_precmz" not in df.columns:
        raise ValueError("Column 'merged_precmz' not found.")

    mz_vals = df["merged_precmz"]

    if "log_i_sum" not in df.columns:
        if "i_sum" not in df.columns:
            raise ValueError("Neither 'log_i_sum' nor 'i_sum' found in dataframe.")
        df["log_i_sum"] = df["i_sum"].apply(lambda x: np.log10(x) if x > 0 else np.nan)

    log_vals = df["log_i_sum"].copy()
    valid = log_vals.notna()

    if valid.any():
        min_val = log_vals[valid].min()
        max_val = log_vals[valid].max()

        if max_val > min_val:
            raw_norm = (log_vals - min_val) / (max_val - min_val)
        else:
            raw_norm = pd.Series(1.0, index=df.index)

        norm_log = 0.2 + 0.8 * raw_norm
    else:
        norm_log = pd.Series(np.nan, index=df.index)

    n_rows = len(df)
    n_cols = len(has_cols)
    M = np.zeros((n_rows, n_cols), dtype=float)

    for j, col in enumerate(has_cols):
        col_bool = df[col].fillna(False).astype(bool)
        M[col_bool.values, j] = norm_log[col_bool].fillna(0).values

    M_masked = np.ma.masked_where(M == 0, M)

    fig, ax = plt.subplots(figsize=(12, 8))

    cmap = plt.cm.Blues.copy()
    cmap.set_bad("white")

    im = ax.imshow(M_masked, aspect="auto", cmap=cmap, interpolation="nearest")

    ax.set_title("has_* tile map (blue shading by log_i_sum)\nrows = precursor m/z")
    ax.set_xlabel("has_* columns")
    ax.set_ylabel("precursor m/z (MS1 feature)")

    ax.set_xticks(range(n_cols))
    if ion_to_label is not None:
        display_labels = []
        for col in has_cols:
            base = col.replace("has_", "", 1)
            label = col
            try:
                mz = float(base)
                label = ion_to_label.get(mz, col)
            except Exception:
                pass
            display_labels.append(label)
    else:
        display_labels = has_cols

    ax.set_xticklabels(display_labels, rotation=90)

    if n_rows > 50:
        step = max(1, n_rows // 50)
        y_indices = np.arange(0, n_rows, step)
    else:
        y_indices = np.arange(n_rows)

    ax.set_yticks(y_indices)
    ax.set_yticklabels(np.round(mz_vals.iloc[y_indices], 4))

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("normalized log_i_sum\n(light = low, dark = high)")

    plt.tight_layout()

    base_name = os.path.splitext(os.path.basename(summary_file))[0]
    out_path = os.path.join(output_dir, f"{base_name}_heatmap.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved figure → {out_path}")

    plt.show()

    return df, output_dir, summary_file, out_path
