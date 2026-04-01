#!/usr/bin/env python3
"""
cyanometdb_match.py

- Match MS1 summaries to CyanometDB entries by precursor m/z within a tolerance.
- Second Excel sheet for unmatched rows that have >=2 diagnostic "has_ " flags
- CSV export and a heatmap PNG for those unknowns
- Safe Excel engine fallback (xlsxwriter -> openpyxl)
"""

from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path
from datetime import datetime
from typing import Iterable, Optional, Sequence
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

# Optional for heatmap; only imported when plotting
try:
    import seaborn as sns  # type: ignore
    _HAS_SNS = True
except Exception:
    _HAS_SNS = False

# --------------------
# Defaults / constants
# --------------------
# Keep core metadata + scan/file provenance; all "has_*" columns are added dynamically
MS1_BASE_KEEP: Sequence[str] = (
    "cluster_id",
    "merged_precmz",
    "n_scans",        
    "scan_nunique",   
    "scan_ids",      
    "ms1_scan_ids",   
    "files",          
    "source_file_<lambda>", 
    "rt_min",
    "rt_median",
    "rt_max",
)


LIB_MZ_COL = "Monoisotopic mass [M+H]+"
MS1_MZ_COL = "merged_precmz"
CLASS_COLS = ("Class of compound", "Alternative class names")


# --------------------
# IO helpers
# --------------------
def read_any_table(path: str | os.PathLike) -> pd.DataFrame:
    p = Path(path)
    suf = p.suffix.lower()
    if suf == ".csv":
        return pd.read_csv(p)
    if suf in {".tsv", ".txt"}:
        return pd.read_csv(p, sep="\t")
    if suf == ".parquet":
        return pd.read_parquet(p)
    raise ValueError(f"Unsupported table format: {p.suffix}")


def _latest_indiv_summary(base: str = ".", run_prefix: str = "Adduct_and_summary_outputs_") -> str:
    """
    Find the newest adduct_and_summary_outputs_* folder, then pick the newest
    indiv_merged_summary_*.csv inside it.
    """
    import glob
    import os

    # 1. Find all run folders
    run_dirs = [
        d for d in glob.glob(os.path.join(base, f"{run_prefix}*"))
        if os.path.isdir(d)
    ]

    if not run_dirs:
        raise FileNotFoundError(f"No folders matching '{run_prefix}*' found.")

    # 2. Pick newest folder by creation time
    latest_dir = max(run_dirs, key=os.path.getctime)

    # 3. Find all indiv_merged_summary CSVs in that folder
    pattern = os.path.join(latest_dir, "indiv_merged_summary_*.csv")
    files = glob.glob(pattern)

    if not files:
        raise FileNotFoundError(
            f"No indiv_merged_summary_*.csv found in newest run folder: {latest_dir}"
        )

    # 4. Pick newest summary file
    latest_file = max(files, key=os.path.getctime)

    print("=" * 60)
    print(f"Newest adduct run:     {latest_dir}")
    print(f"Using MS1 summary file: {latest_file}")
    print("=" * 60)

    return latest_file



# --------------------
# Library loading
# --------------------
def load_library(
    xlsx_path: str | os.PathLike,
    *, 
    class_filter: Optional[Iterable[str] | str] = None,
    class_cols: Sequence[str] = CLASS_COLS,
    mz_col: str = LIB_MZ_COL,
    sheet_index: int = 1,
) -> pd.DataFrame:
    """
    Load the CyanometDB Excel (sheet by numeric index, default=1 i.e., second sheet).
    Optionally filter by class (substring match across class columns).
    """
    # openpyxl preferred; falling back to default if not installed
    try:
        lib = pd.read_excel(xlsx_path, sheet_name=sheet_index, engine="openpyxl")
    except Exception:
        lib = pd.read_excel(xlsx_path, sheet_name=sheet_index)

    # Clean columns
    lib.columns = [str(c).strip() for c in lib.columns]

    # Coerce mz column to numeric and drop NaNs
    if mz_col not in lib.columns:
        raise KeyError(f"mz_col '{mz_col}' not found in library columns.")
    lib[mz_col] = pd.to_numeric(lib[mz_col], errors="coerce")
    lib = lib[lib[mz_col].notna()].copy()

    # Optional class filter
    if class_filter:
        if isinstance(class_filter, str):
            targets = {class_filter.strip().lower()}
        else:
            targets = {str(s).strip().lower() for s in class_filter if s is not None}
        def _row_ok(row):
            hay = []
            for col in class_cols:
                if col in lib.columns and pd.notna(row.get(col)):
                    hay.append(str(row[col]).lower())
            return any(any(t in h for h in hay) for t in targets)
        lib = lib[lib.apply(_row_ok, axis=1)].copy()

    return lib.sort_values(mz_col).reset_index(drop=True)


# --------------------
# MS1 selection helper
# --------------------
def select_ms1_columns(
    ms1_df: pd.DataFrame,
    base_keep: Sequence[str] = MS1_BASE_KEEP,
    drop_empty_has: bool = True,
) -> pd.DataFrame:
    """
    Keep the base MS1 columns plus 'has_*' columns present in THIS table.

    If drop_empty_has=True, keep only those has_* columns that have at least
    one True / non-zero value (i.e., actually used in this run).
    """
    # base columns that are present
    cols = [c for c in base_keep if c in ms1_df.columns]

    # all has_* columns that exist in this table
    has_cols_all = [c for c in ms1_df.columns if str(c).strip().startswith("has_")]

    if drop_empty_has:
        has_cols = []
        for c in has_cols_all:
            col = ms1_df[c]

            # normalize to something we can test
            if col.dtype == bool:
                keep = col.any()
            else:
                # treat non-zero / non-NaN as "present"
                keep = (col.fillna(0) != 0).any()

            if keep:
                has_cols.append(c)
    else:
        has_cols = has_cols_all

    # combine, preserve order, dedupe
    keep = list(dict.fromkeys(cols + has_cols))

    if not keep:
        return ms1_df.copy()

    return ms1_df.loc[:, keep].copy()



# --------------------
# Matching
# --------------------
def match_ms1_to_lib(
    ms1_df: pd.DataFrame,
    lib_df: pd.DataFrame,
    *,
    ms1_mz_col: str = MS1_MZ_COL,
    lib_mz_col: str = LIB_MZ_COL,
    tol_da: float = 0.1,
) -> pd.DataFrame:
    """
    Match MS1 features to library entries based on precursor m/z within ± tol_da.
    Returns a merged DataFrame with annotations from the library.
    """
    results = []
    ms1_df = ms1_df.copy()

    # Ensure numeric m/z
    ms1_df[ms1_mz_col] = pd.to_numeric(ms1_df[ms1_mz_col], errors="coerce")
    valid_ms1 = ms1_df[ms1_df[ms1_mz_col].notna()]

    for _, ms1_row in valid_ms1.iterrows():
        mz = ms1_row[ms1_mz_col]
        hits = lib_df[np.abs(lib_df[lib_mz_col] - mz) <= tol_da]
        if hits.empty:
            results.append({**ms1_row.to_dict(), "Compound identifier": np.nan})
        else:
            for _, hit in hits.iterrows():
                row = {**ms1_row.to_dict(), **hit.to_dict()}
                results.append(row)

    return pd.DataFrame(results)


# --------------------
# Unknowns (>=2 diag) + Excel / CSV / PNG helpers
# --------------------
def _excel_engine() -> str:
    try:
        import xlsxwriter  # noqa: F401
        return "xlsxwriter"
    except Exception:
        print(" xlsxwriter not installed — falling back to openpyxl.")
        return "openpyxl"

def build_unknowns_sheet(matches: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    From the full 'matches' table, derive an 'unknowns' DataFrame of rows with
    no library match and >=2 diagnostic 'has_' flags.
    Returns (grouped_unknowns, bool_cols).
    """
    universe = matches.copy()

    # pick ALL diagnostic flags that start with 'has_' (no space)
    bool_cols = [c for c in universe.columns if str(c).strip().startswith("has_")]

    if "Compound identifier" not in universe.columns:
        # If the caller filtered to matched_only, we can't compute unknowns
        return pd.DataFrame(), bool_cols

    unknown_mask = universe["Compound identifier"].isna()
    no_database_hits = universe.loc[unknown_mask].copy()

    if no_database_hits.empty or not bool_cols:
        return pd.DataFrame(), bool_cols

    # ensure bool dtype (works for 0/1 or True/False)
    no_database_hits[bool_cols] = no_database_hits[bool_cols].fillna(False).astype(bool)
    no_database_hits["n_diagnostic"] = no_database_hits[bool_cols].sum(axis=1)
    no_database_hits = no_database_hits.loc[no_database_hits["n_diagnostic"] >= 2].copy()

    if no_database_hits.empty:
        return pd.DataFrame(), bool_cols

    # fragment pattern string, use last token as m/z piece
    def make_pattern(row):
        return "+".join([col.split()[-1] for col in bool_cols if row[col]])

    no_database_hits["fragment_pattern"] = no_database_hits.apply(make_pattern, axis=1)

    # group by file(s), precursor m/z, and the pattern
    group_keys = []
    for maybe in ["files", "source_file_<lambda>"]:
        if maybe in no_database_hits.columns:
            group_keys.append(maybe)
            break
    group_keys += ["merged_precmz", "fragment_pattern"]

    # aggregation (OR for bools, unique join for scans)
    agg_map = {c: "max" for c in bool_cols}
    present_scan_cols = [c for c in ["scan_ids", "ms1_scan_ids", "scan_number", "ms1_scan"]
                         if c in no_database_hits.columns]

    def uniq_join_ints(x):
        xi = pd.to_numeric(x, errors="coerce")
        if xi.notna().any():
            vals = sorted(set(int(v) for v in xi.dropna()))
            return ",".join(map(str, vals))
        vals = sorted(set(map(str, x.dropna())))
        return ",".join(vals)

    for c in present_scan_cols:
        agg_map[c] = uniq_join_ints

    grouped_unknowns = (
        no_database_hits
        .groupby(group_keys, dropna=False)
        .agg(agg_map)
        .reset_index()
        .sort_values(["merged_precmz"] + [g for g in group_keys if g != "merged_precmz"])
    )

    # pretty row label
    def _files_to_str(v):
        if isinstance(v, (list, tuple)):
            return ",".join(map(str, v))
        return str(v)

    files_col = next((k for k in ["files", "source_file_<lambda>"] if k in grouped_unknowns.columns), None)
    grouped_unknowns["row_label"] = grouped_unknowns.apply(
        lambda row: (
            f"Unknown | m/z={row['merged_precmz']:.4f}"
            f" | pattern={row['fragment_pattern']}"
            + (f" | file={_files_to_str(row[files_col])}" if files_col else "")
        ),
        axis=1
    )

    return grouped_unknowns, bool_cols


import os
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as mpatches


def plot_matched_tiles(
    matched_only: pd.DataFrame,
    out_dir_ts: str,
    ts: str,
    *,
    bool_prefix: str = "has_",
    mz_col: str = "merged_precmz",
    file_col_candidates: Sequence[str] = ("files", "source_file_<lambda>"),
    compound_col: str = "Compound name",
    base_cmap: str = "tab20",  # nicer for categorical colors
) -> Optional[str]:
    """
    Plot a grid showing, for each matched compound (row), which diagnostic has_* flags
    are present (columns). Tiles are colored by contributing file(s):

      - single file  -> solid color tile
      - multi-file   -> horizontal stripes, one color per file

    Colors are consistent per file across the plot. Saves a PNG and returns its path,
    or None if skipped.
    """
    # -------------------------------------------------------------------------
    # sanity checks
    # -------------------------------------------------------------------------
    if matched_only is None or matched_only.empty:
        print("No matched rows to visualize — skipping matched-compound grid.")
        return None
    if compound_col not in matched_only.columns:
        print(f"No '{compound_col}' column found — skipping matched-compound grid.")
        return None

    # collect diagnostic fragment columns (any has_*)
    bool_cols = [c for c in matched_only.columns if str(c).strip().startswith(bool_prefix)]
    if not bool_cols:
        print(f"No columns starting with '{bool_prefix}' found — skipping matched-compound grid.")
        return None

    # pick a file column
    file_col = next((c for c in file_col_candidates if c in matched_only.columns), None)
    if file_col is None:
        print(f"No file provenance column found in {file_col_candidates} — skipping matched-compound grid.")
        return None

    df = matched_only.copy()

    # -------------------------------------------------------------------------
    # normalize file lists: ensure every cell is a list of file IDs
    # -------------------------------------------------------------------------
    def _files_to_list(v):
        if pd.isna(v):
            return []
        if isinstance(v, (list, tuple, set)):
            return list(v)
        if isinstance(v, str):
            # split on commas, strip spaces
            return [x.strip() for x in v.split(",") if x.strip()]
        # fallback: wrap scalar into a 1-element list
        return [str(v)]

    df[file_col] = df[file_col].apply(_files_to_list)

    # -------------------------------------------------------------------------
    # collapse so each (compound, mz, flags...) has a unique list of files
    # -------------------------------------------------------------------------
    grouped = (
        df.groupby([compound_col, mz_col] + bool_cols, dropna=False)[file_col]
          .apply(lambda x: sorted(set(f for sub in x for f in sub)))  # flatten & unique
          .reset_index()
    )

    if grouped.empty:
        print("Nothing to plot after grouping matched compounds.")
        return None

    # sort by precursor m/z
    grouped = grouped.sort_values(mz_col, ascending=True).reset_index(drop=True)

    # row labels: "Compound | m/z=xxx.xxxx"
    grouped["row_label"] = (
        grouped[compound_col].fillna("No match").astype(str)
        + " | m/z="
        + grouped[mz_col].map(lambda x: f"{float(x):.4f}" if pd.notna(x) else "nan")
    )

    # -------------------------------------------------------------------------
    # build file palette: one stable color per file
    # -------------------------------------------------------------------------
    all_files = sorted({f for L in grouped[file_col] for f in L})
    if not all_files:
        print("No file provenance found after grouping — nothing to color by.")
        return None

    N = max(len(all_files), 2)

    def make_many_colors(n, base_cmap="tab20"):
        # tab20 / tab10 / Set3 etc. are nicer for categorical data
        cmap = cm.get_cmap(base_cmap, n)
        return [cmap(i) for i in range(n)]

    colors = make_many_colors(N, base_cmap=base_cmap)
    file_to_color = {f: colors[i] for i, f in enumerate(all_files)}

    # -------------------------------------------------------------------------
    # draw grid
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 0.5 * len(grouped) + 2))

    for row_idx, row in grouped.iterrows():
        for col_idx, col in enumerate(bool_cols):
            if bool(row[col]):
                files_here = row[file_col]  # guaranteed list from _files_to_list

                if not files_here:
                    # no provenance: light gray tile
                    color = (0.85, 0.85, 0.85, 1.0)
                    ax.add_patch(
                        plt.Rectangle((col_idx, row_idx), 1, 1,
                                      facecolor=color, edgecolor="white")
                    )
                elif len(files_here) == 1:
                    # single file: solid tile
                    f = files_here[0]
                    color = file_to_color.get(f, (0.85, 0.85, 0.85, 1.0))
                    ax.add_patch(
                        plt.Rectangle((col_idx, row_idx), 1, 1,
                                      facecolor=color, edgecolor="white")
                    )
                else:
                    # multiple files: horizontal stripes filling the 1x1 tile
                    n = len(files_here)
                    stripe_h = 1.0 / n
                    for k, f in enumerate(files_here):
                        y0 = row_idx + k * stripe_h
                        ax.add_patch(
                            plt.Rectangle(
                                (col_idx, y0),   # (x, y) lower-left
                                1,               # full tile width
                                stripe_h,        # stripe height
                                facecolor=file_to_color.get(f, (0.85, 0.85, 0.85, 1.0)),
                                edgecolor="white",
                                linewidth=0.3,
                            )
                        )

    # -------------------------------------------------------------------------
    # axes formatting
    # -------------------------------------------------------------------------
    ax.set_xlim(0, len(bool_cols))
    ax.set_ylim(0, len(grouped))
    ax.set_xticks(np.arange(len(bool_cols)) + 0.5)
    ax.set_xticklabels(bool_cols, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(grouped)) + 0.5)
    ax.set_yticklabels(grouped["row_label"], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Diagnostic fragments / has_* flags")
    ax.set_ylabel("Matched compounds (sorted by m/z)")
    ax.set_title("Matched compounds: file-colored tiles per diagnostic fragment", fontsize=14)

    # -------------------------------------------------------------------------
    # legend: one color patch per file
    # -------------------------------------------------------------------------
    patches = [mpatches.Patch(color=file_to_color[f], label=f) for f in all_files]
    n = len(all_files)
    ncols = 1 if n <= 12 else 2 if n <= 30 else 3 if n <= 60 else 4

    ax.legend(
        handles=patches,
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        fontsize=6,
        ncol=ncols,
        title="Files",
        borderaxespad=0.0,
    )

    fig.tight_layout()

    # -------------------------------------------------------------------------
    # save
    # -------------------------------------------------------------------------
    os.makedirs(out_dir_ts, exist_ok=True)
    out_png = os.path.join(out_dir_ts, f"matched_compound_tiles_{ts}.png")
    fig.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Saved matched-compounds grid: {os.path.abspath(out_png)}")
    plt.close(fig)
    return out_png

def write_outputs(
    matches: pd.DataFrame,
    matched_only: pd.DataFrame,
    out_dir: str,
    ts: str | None = None,
    write_excel: bool = True,
    make_heatmap: bool = True,
):
    """
    Write outputs (CSV, Excel, and optionally heatmap) into a timestamped directory.
    Returns (out_dir_ts, paths_dict).
    """
    if ts is None:
        ts = datetime.now().strftime("%y-%m-%d_%H-%M-%S")

    # make timestamped folder: e.g., matches_out_25-11-11_13-25-51
    out_dir_ts = f"{out_dir}_{ts}"
    os.makedirs(out_dir_ts, exist_ok=True)

    paths = {}

    # CSV of matched_only
    csv_path = os.path.join(out_dir_ts, f"cyanometdb_matches_{ts}.csv")
    matched_only.to_csv(csv_path, index=False)
    print(f"Saved: {os.path.abspath(csv_path)} (rows: {len(matched_only)})")
    paths["matches_csv"] = csv_path

    # Build unknowns
    unknowns, bool_cols = build_unknowns_sheet(matches)

    # Excel with two sheets
    if write_excel:
        excel_path = os.path.join(out_dir_ts, f"cyanometdb_matches_{ts}.xlsx")
        engine = _excel_engine()
        with pd.ExcelWriter(excel_path, engine=engine) as xw:
            matched_only.to_excel(xw, index=False, sheet_name="matches")
            if not unknowns.empty:
                unknowns.to_excel(xw, index=False, sheet_name="unknowns_>=2diag")
            else:
                pd.DataFrame({"note": ["no unknowns with >=2 diagnostic ions"]}).to_excel(
                    xw, index=False, sheet_name="unknowns_>=2diag"
                )
        print(f"Excel written: {os.path.abspath(excel_path)}")
        paths["excel"] = excel_path

    # CSV + heatmap PNG for unknowns
    if not unknowns.empty:
        unk_csv = os.path.join(out_dir_ts, f"unknown_features_with_scans_{ts}.csv")
        unknowns.to_csv(unk_csv, index=False)
        print(f"Exported unknown features with scans: {os.path.abspath(unk_csv)} (rows: {len(unknowns)})")
        paths["unknowns_csv"] = unk_csv

        if make_heatmap and _HAS_SNS and bool_cols:
            import matplotlib.pyplot as plt
            import seaborn as sns

            matrix = unknowns.set_index("row_label")[bool_cols]
            if matrix.empty:
                print("No unknowns matrix to plot — skipping figure.")
            else:
                fig, ax = plt.subplots(figsize=(12, 0.5 * len(matrix) + 2))
                sns.heatmap(
                    matrix.astype(int),
                    cmap=["lightgrey", "orange"],
                    cbar=False,
                    linewidths=0.5,
                    linecolor="white",
                    ax=ax
                )
                ax.set_title("Unknown Features (≥2 diagnostic ions, per-file detail)", fontsize=14)
                ax.set_xlabel("Diagnostic Fragments")
                ax.set_ylabel("Unknown Precursor m/z (pattern, file)")
                plt.xticks(rotation=45, ha="right")
                plt.yticks(fontsize=7)
                fig.tight_layout()

                out_png = os.path.join(out_dir_ts, f"unknown_features_with_scans_{ts}.png")
                fig.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="white")
                print(f"Saved figure: {os.path.abspath(out_png)}")
                plt.close(fig)
                paths["unknowns_png"] = out_png
        elif make_heatmap and not _HAS_SNS:
            print("seaborn not installed — skipping heatmap PNG for unknowns.")

    return out_dir_ts, paths
