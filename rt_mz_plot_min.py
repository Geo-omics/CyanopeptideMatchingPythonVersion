# rt_mz_plot_min.py
import os, colorsys, numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import matplotlib.patches as mpatches



def make_cmap(base_hue: float):
    """Return a list of colors for a given base hue (0–1)."""
    def cmap(levels):
        return [
            mcolors.to_hex(colorsys.hls_to_rgb(base_hue, 0.3 + 0.6*l, 0.8))
            for l in levels
        ]
    return cmap

def plot_precursor_rt(df, ion_to_label=None, ax=None, save=False, out_dir="."):
    fig, ax = (plt.subplots(figsize=(10,6)) if ax is None else (ax.figure, ax))
    markers = ["o","^","s","D","x","*"]
    files = df["source_file"].unique().tolist()
    for i, src in enumerate(files):
        sf = df[df["source_file"] == src]
        ions = sf["ion"].unique().tolist()
        colors = make_cmap(i/len(files))(np.linspace(0,1,max(len(ions),1)))
        for ion, color in zip(ions, colors):
            sub = sf[sf["ion"] == ion]
            label = f"{ion_to_label.get(float(ion), ion) if ion_to_label else ion} — {os.path.basename(str(src))}"
            ax.scatter(sub["rt"], sub["precmz"], s=20, alpha=0.6,
                       marker=markers[i % len(markers)], color=color, label=label)
    ax.set_xlabel("Retention time (min)")
    ax.set_ylabel("Precursor m/z")
    ax.set_title("Precursor m/z vs RT — per ion & file")
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()

    if save:
        import datetime
        ts = datetime.datetime.now().strftime("%y-%m-%d_%H-%M-%S")
        out_path = os.path.join(out_dir, f"Precursor_rt_plot_{ts}.png")
        fig.savefig(out_path, dpi=300)
        print(f"Saved figure → {out_path}")

    return fig, ax


def plot_per_file_legend(df, save=False, out_dir="."):
    if df.empty:
        print("DataFrame is empty; nothing to plot.")
        return
    plt.figure(figsize=(10,6))
    markers = ["o","^","s","D","x","*"]
    files = df["source_file"].unique().tolist()
    legend_handles = []
    for i, src in enumerate(files):
        sub_file = df[df["source_file"] == src]
        unique_ions = sub_file["ion"].unique()
        hue = i / len(files)
        cmap = make_cmap(hue)
        colors = cmap(np.linspace(0,1,len(unique_ions)))
        for ion, color in zip(unique_ions, colors):
            sub = sub_file[sub_file["ion"] == ion]
            plt.scatter(sub["rt"], sub["precmz"], s=20, alpha=0.6,
                        marker=markers[i % len(markers)], color=color)
        mid_color = cmap([0.5])[0]
        patch = mpatches.Patch(color=mid_color, label=os.path.basename(src))
        legend_handles.append(patch)
    plt.xlabel("Retention time (min)")
    plt.ylabel("Precursor m/z")
    plt.title("Precursor m/z vs RT — per ion & file")
    plt.legend(handles=legend_handles, ncol=2, fontsize=8, title="Files")
    plt.tight_layout()

    if save:
        import datetime
        ts = datetime.datetime.now().strftime("%y-%m-%d_%H-%M-%S")
        out_path = os.path.join(out_dir, f"RT_precursor_per_file_legend_{ts}.png")
        plt.savefig(out_path, dpi=300)
        print(f"Saved figure → {out_path}")

    plt.show()


