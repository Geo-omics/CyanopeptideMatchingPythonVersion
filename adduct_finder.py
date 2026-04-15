# adduct_finder.py
#Adduct / neutral-loss linker for merged precursor tables.
#detects adducts and losses by linking precursor features with m/z mass shifts
#builds network of related ions and plots to visualize precursor families 

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple, List

import numpy as np
import pandas as pd

try:
    import networkx as nx
except ImportError:  # helpful at runtime if missing
    raise ImportError("Please `pip install networkx` to use adduct_finder.")

import matplotlib.pyplot as plt


PROTON = 1.007276466

# Default set (positive mode). You can pass your own dict to override/extend.
DEFAULT_DELTAS: Dict[str, float] = {
    "+H2O":              18.010565,                          # add water
    "-H2O":             -18.010565,                          # lose water
    "+NH4 (vs H)":       18.033823 - PROTON,                 # NH4 vs H
    "+Na (vs H)":        22.989218 - PROTON,                 # Na vs H
    "+K (vs H)":         38.963158 - PROTON,                 # K vs H
    "-SO3":             -79.956815,                          # lose SO3
    "+Na + H2O":             39.99,                          #add water and Na

}

@dataclass
class AdductConfig:
    deltas: Dict[str, float] = None          # name -> neutral mass difference (Da)
    ppm_tol: float = 10.0                    # ppm window on expected Δm/z
    mz_tol: float = 0.01                     # absolute Da fallback
    polarity: str = "POSITIVE"               # "POSITIVE" or "NEGATIVE"
    max_charge_assumed: int = 1              # used if charge column missing

    def __post_init__(self):
        if self.deltas is None:
            self.deltas = dict(DEFAULT_DELTAS)
        # For NEGATIVE mode, most users still calculate deltas in neutral-mass space,
        # so we typically do NOT invert signs here. Change if your chemistry requires it.


def _expected_dmz(neutral_delta_da: float, z: float) -> float:
    # Convert neutral-mass delta to m/z difference at charge z
    z = max(float(z), 1.0)
    return neutral_delta_da / z


def _ppm(err: float, ref: float) -> float:
    return (abs(err) / abs(ref)) * 1e6 if ref != 0 else math.inf

def detect_adducts(
    df: pd.DataFrame,
    mz_col: str = "merged_precmz",
    charge_col: Optional[str] = "rep_charge",
    rt_col: Optional[str] = "rt_median",
    cfg: Optional[AdductConfig] = None,
) -> Tuple[nx.Graph, pd.DataFrame, pd.DataFrame]:
    """
    Build a graph linking nodes (precursors) whose m/z differences match configured deltas.

    Returns
    -------
    (G, edges_best_df, edges_all_df)
        G: networkx.Graph with node attributes {mz, z, rt}
        edges_best_df: one "best" edge (lowest ppm) per (u, v)
        edges_all_df: all matching edges (u, v, possibly multiple delta_name per pair)
    """
    if cfg is None:
        cfg = AdductConfig()

    if mz_col not in df.columns:
        raise ValueError(f"df must contain column '{mz_col}'")

    d = df.copy().sort_values(mz_col).reset_index(drop=True)
    if charge_col not in d.columns or charge_col is None:
        d["__z__"] = cfg.max_charge_assumed
        cz = "__z__"
    else:
        cz = charge_col
        d[cz] = d[cz].fillna(cfg.max_charge_assumed)

    # Node labels are rounded string m/z for readability; keep full-precision in attrs
    d["__node__"] = d[mz_col].round(4).astype(str)

    # --------- Build nodes ----------
    G = nx.Graph()
    for _, row in d.iterrows():
        attrs = {
            "mz": float(row[mz_col]),
            "z": int(row[cz]) if not pd.isna(row[cz]) else cfg.max_charge_assumed,
        }
        if rt_col and rt_col in d.columns:
            try:
                attrs["rt"] = float(row[rt_col])
            except Exception:
                pass
        G.add_node(row["__node__"], **attrs)

    # --------- Pairwise tests ----------
    edges: List[Tuple[str, str, Dict]] = []
    mz_vals = d[mz_col].values
    z_vals = d[cz].astype(float).values
    nodes = d["__node__"].values


    for i in range(len(d)):
        for j in range(i + 1, len(d)):
            dmz = mz_vals[j] - mz_vals[i]

            z_i = z_vals[i] if z_vals[i] else cfg.max_charge_assumed
            z_j = z_vals[j] if z_vals[j] else cfg.max_charge_assumed

            # Enforce same-charge linking
            if z_i != z_j:
                continue

            for name, dmass in cfg.deltas.items():
                exp_dmz = _expected_dmz(dmass, z_i)
                tol = max(cfg.mz_tol, abs(exp_dmz) * cfg.ppm_tol / 1e6)
                if abs(dmz - exp_dmz) <= tol:
                    Mi = z_i * mz_vals[i] - z_i * PROTON
                    Mj_est = z_i * mz_vals[j] - z_i * PROTON
                    ppm_err = _ppm(Mj_est - (Mi + dmass), Mi + dmass)
                    edges.append(
                        (
                            nodes[i],
                            nodes[j],
                            {
                                "label": f"{name} ({dmass:+.4f})",
                                "delta_name": name,
                                "delta_da": float(dmass),
                                "ppm": float(ppm_err),
                                "dmz": float(dmz),
                                "expected_dmz": float(exp_dmz),
                            },
                        )
                    )
                    # NOTE: no break → allow multiple deltas per pair

    # --------- ALL matches table ----------
    if edges:
        edges_all_df = pd.DataFrame(
            [
                (u, v, d["delta_name"], d["delta_da"], d["ppm"], d["dmz"], d["expected_dmz"])
                for u, v, d in edges
            ],
            columns=["u", "v", "delta_name", "delta_da", "ppm", "dmz", "expected_dmz"],
        ).sort_values(["u", "v", "ppm", "delta_name"], ignore_index=True)
    else:
        edges_all_df = pd.DataFrame(
            columns=["u", "v", "delta_name", "delta_da", "ppm", "dmz", "expected_dmz"]
        )

    # --------- BEST edge per (u, v) for the graph ----------
    for u, v, data in edges:
        if G.has_edge(u, v):
            if data["ppm"] < G[u][v].get("ppm", 1e9):
                G[u][v].update(data)
        else:
            G.add_edge(u, v, **data)

    edges_best_df = pd.DataFrame(
        [
            (u, v, d.get("delta_name"), d.get("delta_da"), d.get("ppm"), d.get("dmz"), d.get("expected_dmz"))
            for u, v, d in G.edges(data=True)
        ],
        columns=["u", "v", "delta_name", "delta_da", "ppm", "dmz", "expected_dmz"],
    ).sort_values(["ppm", "delta_name"], ignore_index=True)

    return G, edges_best_df, edges_all_df



def plot_graph(
    G: nx.Graph,
    title: str = "Delta-mass forest",
    seed: int = 42,
    family_gap: float = 4.0,
    figsize: Optional[Tuple[float, float]] = None,
):
    """
    Simple multi-component layout & plot, coloring by component.
    """
    comps = [sorted(list(c), key=lambda n: G.nodes[n].get("mz", np.inf)) for c in nx.connected_components(G)]
    roots = [c[0] for c in comps if c]

    # Pack components left-to-right
    pos = {}
    x_offset = 0.0
    for comp in comps:
        sub = G.subgraph(comp)
        sub_pos = nx.spring_layout(sub, k=0.6, seed=seed)
        xs = [sub_pos[n][0] for n in sub] or [0.0]
        mn, mx = min(xs), max(xs)
        for n in sub:
            pos[n] = (sub_pos[n][0] - mn + x_offset, sub_pos[n][1])
        x_offset += (mx - mn + family_gap)

    if figsize is None:
        figsize = (max(10, 2 * len(comps)), 8)

    plt.figure(figsize=figsize)
    colors = plt.cm.tab20(np.linspace(0, 1, max(1, len(comps))))
    comp_color = {}
    for ci, comp in enumerate(comps):
        for n in comp:
            comp_color[n] = colors[ci % len(colors)]

    nx.draw_networkx_edges(G, pos, alpha=0.6)

    node_sizes = [900 if n in roots else 450 for n in G.nodes]
    node_colors = [comp_color.get(n, colors[0]) for n in G.nodes]
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, edgecolors="k", linewidths=0.5)

    labels = {}
    for n, data in G.nodes(data=True):
        rt_part = f"\nRT {data['rt']:.2f}m" if "rt" in data and isinstance(data["rt"], (int, float)) else ""
        labels[n] = f"{n} (z={data.get('z', 1)}){rt_part}"
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=8)

    edge_labels = {(u, v): d.get("label", "") for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)

    plt.axis("off")
    plt.title(title)
    plt.tight_layout()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
  
