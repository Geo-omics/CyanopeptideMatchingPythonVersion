# adduct_pipeline.py
"""
Helpers to:
  - summarize (via your make_summary_ind)
  - detect adducts (via af.detect_adducts)
  - color/draw graphs
  - save CSVs and PNGs with timestamped names (yy-mm-dd_HH-MM-SS)
  - export cleaned edges to Excel (renames 'expected' -> 'expected_dmz' and reorders columns)

NOTE: This module purposely does NOT filter. Do any m/z or RT filtering in your notebook
before calling these functions (pass a pre-filtered DataFrame).
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Optional, Tuple

import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx


# =========================
# Coloring / drawing helpers
# =========================
NODE_COLOR_MAP = {
    "+H2O": "skyblue",
    "+K (vs H)": "red",
    "-H2O": "grey",
    "+Na (vs H)": "orange",
    "+H": "green",
    "+NH4 (vs H)": "pink",
}
DEFAULT_NODE_COLOR = "gray"


def node_colors_from_edges(
    G: nx.Graph,
    color_map: dict = NODE_COLOR_MAP,
    default: str = DEFAULT_NODE_COLOR,
) -> list[str]:
    """Color each node by the first matching edge 'delta_name' it participates in."""
    colors = []
    for n in G.nodes:
        rels = [G[n][nbr].get("delta_name", "") for nbr in G.neighbors(n)]
        color = default
        for r in rels:
            if r in color_map:
                color = color_map[r]
                break
        colors.append(color)
    return colors


def draw_colored_graph(
    G: nx.Graph,
    title: str,
    k: float = 0.8,
    out_png: Optional[str] = None,
) -> None:
    """NetworkX drawing with node colors + edge labels. Saves PNG if out_png is provided."""
    if G.number_of_nodes() == 0:
        print(f"{title}: graph is empty; nothing to draw.")
        return

    # Skip drawing very large graphs
    MAX_NODES_TO_DRAW = 400
    if G.number_of_nodes() > MAX_NODES_TO_DRAW:
        print(
            f"{title}: graph has {G.number_of_nodes()} nodes (>{MAX_NODES_TO_DRAW}); "
            "skipping plot."
        )
        return

    pos = nx.spring_layout(G, seed=20, k=k)
    plt.figure(figsize=(12, 10))
    nx.draw(
        G,
        pos,
        with_labels=True,
        node_size=500,
        font_size=9,
        node_color=node_colors_from_edges(G),
        edge_color="black",
    )
    edge_labels = nx.get_edge_attributes(G, "delta_name")
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)
    plt.title(title)

    if out_png:
        os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
        plt.tight_layout()
        plt.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"Saved graph → {os.path.abspath(out_png)}")

    plt.show()
    plt.close()



# =========================
# Timestamp / paths helpers
# =========================
def _ts() -> str:
    """Timestamp like 'yy-mm-dd_HH-MM-SS'."""
    return datetime.now().strftime("%y-%m-%d_%H-%M-%S")


def _safe_outdir(base_out_dir: str, ts: Optional[str] = None) -> tuple[str, str]:
    """
    Create a timestamped output folder and return (out_dir_ts, ts).

    If ts is provided, reuse it. Otherwise generate a new one.
    """
    if ts is None:
        ts = _ts()
    out_dir_ts = f"{base_out_dir}_{ts}"
    os.makedirs(out_dir_ts, exist_ok=True)
    return out_dir_ts, ts


def _save_edges_excel(edges: pd.DataFrame, path: str) -> None:
    """Save edges DataFrame to Excel with safe engine fallback."""
    # Prefer xlsxwriter; fallback to openpyxl
    try:
        import xlsxwriter  # noqa: F401
        engine = "xlsxwriter"
    except Exception:
        print(" xlsxwriter not installed — falling back to openpyxl.")
        engine = "openpyxl"

    with pd.ExcelWriter(path, engine=engine) as xw:
        edges.to_excel(xw, index=False, sheet_name="edges")
    print(f"Excel file written: {os.path.abspath(path)}")


def _clean_edges_df(edges: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize edges DataFrame:
      - rename 'expected' -> 'expected_dmz' if present
      - reorder common columns if present
    """
    if edges is None or edges.empty:
        return pd.DataFrame()

    edges_df = edges.copy()

    if "expected" in edges_df.columns:
        edges_df = edges_df.rename(columns={"expected": "expected_dmz"})

    desired = ["u", "v", "delta_name", "delta_da", "ppm", "dmz", "expected_dmz"]
    have = [c for c in desired if c in edges_df.columns]
    if have:
        edges_df = edges_df.loc[:, have]

    return edges_df


# =========================
# Pipeline wrappers
# =========================
def run_merged(
    df: pd.DataFrame,
    *,
    make_summary_ind,       # your function: (df) -> summary_df
    af_module,              # module with AdductConfig + detect_adducts + plot_graph (optional)
    mz_col: str = "merged_precmz",
    charge_col: str = "charge",         # change to 'rep_charge' if that's your field
    rt_col: str = "rt_median",
    out_dir: str = ".",
    ts: str | None = None,
    save_graph: bool = True,
    graph_title: str = "Adduct/neutral-loss links (merged)",
) -> Tuple[pd.DataFrame, pd.DataFrame, nx.Graph]:
    """
    Build merged summary from df (already filtered upstream if desired),
    detect adducts, save CSVs + PNG, and return (summary_df, edges_df, graph).

    edges_df here = BEST edges (one per node pair), suitable for graph/summary.
    All edges (including alternative adduct assignments) are written to a separate CSV.
    """
    if df is None or df.empty:
        raise ValueError("run_merged: input DataFrame is empty.")

    # Create timestamped output directory, honoring a user-supplied ts if given
    out_dir_ts, ts = _safe_outdir(out_dir, ts)

    # 1) Build merged summary
    summary = make_summary_ind(df)

    # 2) Detect adducts
    cfg = af_module.AdductConfig(ppm_tol=10, mz_tol=0.01)
    G, edges_best, edges_all = af_module.detect_adducts(
        summary,
        mz_col=mz_col,
        charge_col=charge_col,
        rt_col=rt_col,
        cfg=cfg,
    )

    print("Merged:", G.number_of_nodes(), "nodes;", G.number_of_edges(), "edges")

    # 3) Save merged summary CSV
    merged_summary_csv = os.path.join(out_dir_ts, f"indiv_merged_summary_{ts}.csv")
    summary.to_csv(merged_summary_csv, index=False)
    print(f"Saved merged summary → {os.path.abspath(merged_summary_csv)}")

    # 4) Clean and save BEST edges (CSV + Excel)
    edges_df = _clean_edges_df(edges_best) if isinstance(edges_best, pd.DataFrame) else pd.DataFrame()

    if not edges_df.empty:
        # Save CSV of best edges
        merged_edges_csv = os.path.join(out_dir_ts, f"indiv_merged_best_edges_{ts}.csv")
        edges_df.to_csv(merged_edges_csv, index=False)
        print(f"Saved merged edges → {os.path.abspath(merged_edges_csv)}")

        # Save Excel summary of best edges
        excel_name = f"parent_adduct_summary_best_edges_{ts}.xlsx"
        excel_path = os.path.join(out_dir_ts, excel_name)
        _save_edges_excel(edges_df, excel_path)
    else:
        print("Note: 'edges_best' is not a non-empty DataFrame; skipping best-edges CSV/Excel save.")

    # 4b) Save ALL adduct connections as a separate CSV
    if isinstance(edges_all, pd.DataFrame) and not edges_all.empty:
        edges_all_safe = edges_all.copy()
    
        # Prevent Excel from treating +Na, +K, etc. as formulas
        if "delta_name" in edges_all_safe.columns:
            edges_all_safe["delta_name"] = edges_all_safe["delta_name"].apply(
                lambda x: (
                    "'" + x  # leading apostrophe- Excel shows the text, not a formula
                    if isinstance(x, str) and x[:1] in {"+", "-", "=", "@"}
                    else x
                )
            )
    
        all_edges_csv = os.path.join(out_dir_ts, f"adduct_all_connections_{ts}.csv")
        edges_all_safe.to_csv(all_edges_csv, index=False)
        print(f"Saved ALL adduct connections → {os.path.abspath(all_edges_csv)}")

    else:
        print("No additional adduct connections to save (edges_all is empty or not a DataFrame).")

    # 5) Plot colored graph (PNG)
    png = os.path.join(out_dir_ts, f"adduct_graph_merged_{ts}.png") if save_graph else None
    draw_colored_graph(G, title=graph_title, out_png=png)

    return summary, edges_df, G

def run_per_file(
    df: pd.DataFrame,
    *,
    make_summary_ind,       # your function
    af_module,              # module with AdductConfig + detect_adducts
    group_col: str = "source_file",
    mz_col: str = "merged_precmz",
    charge_col: str = "charge",         # or 'rep_charge'
    rt_col: str = "rt_median",
    out_dir: str = ".",
    save_graph: bool = True,
    graph_title_prefix: str = "Adduct/neutral-loss links — File: ",
) -> None:
    """
    Loop over group_col (e.g., per source file), summarize, detect adducts,
    save CSVs + PNGs. Skips empty groups.
    """
    if df is None or df.empty:
        print("run_per_file: input DataFrame is empty; nothing to do.")
        return

    if group_col not in df.columns:
        print(f"run_per_file: '{group_col}' column not found; nothing to do.")
        return

    # Single timestamped parent directory for this batch
    out_dir_ts, _ = _safe_outdir(out_dir)

    cfg = af_module.AdductConfig(ppm_tol=10, mz_tol=0.01)

    for g, sub in df.groupby(group_col):
        base = os.path.splitext(os.path.basename(str(g)))[0]
        print(f"\n=== File: {base} ===")

        if sub.empty:
            print("No rows for this group; skipping.")
            continue

        # 1) Build summary (sub is already optionally filtered upstream)
        summary = make_summary_ind(sub)

        # Per-file timestamp, so each file gets its own time suffix
        ts = _ts()

        # 2) Save per-file summary
        out_sum_csv = os.path.join(out_dir_ts, f"indiv_summary_{base}_{ts}.csv")
        summary.to_csv(out_sum_csv, index=False)
        print(f"Saved per-file summary → {os.path.abspath(out_sum_csv)}")

        # 3) Detect adducts
        Gf, edges_f = af_module.detect_adducts(
            summary,
            mz_col=mz_col,
            charge_col=charge_col,
            rt_col=rt_col,
            cfg=cfg,
        )
        print(f"File {base}: {Gf.number_of_nodes()} nodes; {Gf.number_of_edges()} edges")

        # 4) Save edges if DataFrame
        edges_f_clean = _clean_edges_df(edges_f) if isinstance(edges_f, pd.DataFrame) else pd.DataFrame()
        if not edges_f_clean.empty:
            out_edges_csv = os.path.join(out_dir_ts, f"indiv_edges_{base}_{ts}.csv")
            edges_f_clean.to_csv(out_edges_csv, index=False)
            print(f"Saved per-file edges → {os.path.abspath(out_edges_csv)}")

            # also Excel for each file
            excel_name = f"parent_adduct_summary_{base}_{ts}.xlsx"
            excel_path = os.path.join(out_dir_ts, excel_name)
            _save_edges_excel(edges_f_clean, excel_path)
        else:
            print(f"File {base}: no non-empty edges DataFrame; skipping edges CSV/Excel save.")

        # 5) Plot colored graph (PNG)
        graph_title = f"{graph_title_prefix}{base}"
        png = os.path.join(out_dir_ts, f"adduct_graph_{base}_{ts}.png") if save_graph else None
        draw_colored_graph(Gf, title=graph_title, out_png=png)

