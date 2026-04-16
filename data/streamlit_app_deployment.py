from __future__ import annotations

import contextlib
import glob
import importlib.util
import io
import mimetypes
import os
from pathlib import Path
import re
import shutil
import sys
import zipfile

import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = BASE_DIR / "uploaded_mzml"
RUNS_DIR = BASE_DIR / "streamlit_runs"
UPLOAD_DIR.mkdir(exist_ok=True)
RUNS_DIR.mkdir(exist_ok=True)

POLARITY = "POSITIVE"
DEFAULT_RT_WINDOW = (2.0, 25.0)
DEFAULT_REF_RT_WINDOW = (1.0, 25.0)
DEFAULT_TOL = 0.01
DEFAULT_TOL_DA = 0.1
MAX_ZIP_MB = 250


@st.cache_resource(show_spinner=False)
def load_backend_module():
    candidates = sorted(BASE_DIR.glob("CPM_cli*.py")) + sorted(BASE_DIR.glob("*cli*.py"))
    if not candidates:
        raise FileNotFoundError(
            "No backend pipeline script found next to the app. "
            "Place CPM_cli_04_14_2026_test.py beside this Streamlit app."
        )
    backend_path = candidates[0]
    spec = importlib.util.spec_from_file_location("pipeline_backend", backend_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load backend from {backend_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["pipeline_backend"] = module
    spec.loader.exec_module(module)
    return module, backend_path


@st.cache_resource(show_spinner=False)
def resolve_library_path() -> Path:
    candidates = [
        DATA_DIR / "CyanoMetDB_Version03.xlsx",
        BASE_DIR / "CyanoMetDB_Version03.xlsx",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        "CyanoMetDB_Version03.xlsx was not found. "
        "Put it in ./data/ or beside this app."
    )


MZML_ID_PATTERN = re.compile(r'id="merged=(\d+)\s+row=\d+"')


def save_and_fix_uploaded_mzml(uploaded_file) -> Path:
    out_path = UPLOAD_DIR / uploaded_file.name
    raw_bytes = uploaded_file.getvalue()
    try:
        text = raw_bytes.decode("utf-8")
    except UnicodeDecodeError:
        out_path.write_bytes(raw_bytes)
        return out_path

    fixed_text, _ = MZML_ID_PATTERN.subn(r'id="scan=\1"', text)
    out_path.write_text(fixed_text, encoding="utf-8")
    return out_path


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".svg", ".webp"}
TABLE_EXTS = {".csv", ".tsv", ".txt", ".xlsx", ".xls"}


def find_first(patterns: list[str], root: Path) -> Path | None:
    for pattern in patterns:
        matches = sorted(root.rglob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
        if matches:
            return matches[0]
    return None



def collect_output_files(run_dir: Path) -> list[Path]:
    files: list[Path] = []
    for p in run_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in {
            ".csv", ".tsv", ".txt", ".xlsx", ".xls", ".png", ".jpg", ".jpeg", ".svg", ".pdf", ".json"
        }:
            files.append(p)
    return sorted(files)



def load_table(path: Path) -> pd.DataFrame | None:
    try:
        if path.suffix.lower() == ".csv":
            return pd.read_csv(path)
        if path.suffix.lower() in {".tsv", ".txt"}:
            return pd.read_csv(path, sep="\t")
        if path.suffix.lower() in {".xlsx", ".xls"}:
            return pd.read_excel(path)
    except Exception:
        return None
    return None



def make_zip_bytes(paths: list[Path], root: Path) -> io.BytesIO:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in paths:
            zf.write(path, arcname=str(path.relative_to(root)))
    buffer.seek(0)
    return buffer



def render_downloads(files: list[Path], root: Path) -> None:
    if not files:
        st.info("No output files found for this run.")
        return

    total_bytes = sum(p.stat().st_size for p in files if p.exists())
    total_mb = total_bytes / (1024 * 1024)
    if total_mb <= MAX_ZIP_MB:
        zip_buffer = make_zip_bytes(files, root)
        st.download_button(
            "Download all outputs as ZIP",
            data=zip_buffer,
            file_name=f"{root.name}_outputs.zip",
            mime="application/zip",
        )
    else:
        st.warning(f"Outputs are about {total_mb:.1f} MB, so ZIP download is skipped.")

    st.subheader("Individual output files")
    for path in files:
        mime, _ = mimetypes.guess_type(path.name)
        with open(path, "rb") as f:
            st.download_button(
                label=f"Download {path.relative_to(root)}",
                data=f,
                file_name=path.name,
                mime=mime or "application/octet-stream",
                key=f"dl-{path}",
            )



def render_results(run_dir: Path) -> None:
    st.success(f"Run finished. Outputs saved in: {run_dir}")

    log_file = find_first(["pipeline_log/*.txt", "*.txt"], run_dir)
    if log_file:
        with st.expander("Pipeline log", expanded=False):
            st.text(log_file.read_text(encoding="utf-8", errors="replace"))

    image_sections = [
        ("Specific cyanopeptide-class detection intensity heatmap", ["**/*heatmap*.png", "**/*heatmap*.jpg"]),
        ("Matched compound tiles", ["**/matched_compound_tiles_*.png"]),
        ("Adduct graph", ["**/adduct_graph_merged_*.png", "**/adduct_graph*.png"]),
        ("Unknown features heatmap", ["**/unknown_features_with_scans_*.png"]),
        ("RT / precursor plot", ["**/Precursor_rt_plot*.png"]),
        ("Diagnostic ion count plot", ["**/Diagnostic_ion_distribution_individual_*.png"]),
        ("Individual ion dot plot", ["**/indiv_diagnostic_ions_*.png"]),
    ]
    for title, patterns in image_sections:
        path = find_first(patterns, run_dir)
        if path:
            st.subheader(title)
            st.image(str(path), caption=path.name)

    table_sections = [
        ("Individual hits preview", ["**/individual_hits_*.csv"]),
        ("Merged summary preview", ["**/indiv_merged_summary_*.csv"]),
        ("CyanoMetDB matches preview", ["**/cyanometdb_matches_*.csv"]),
        ("Unknown features preview", ["**/unknown_features_with_scans_*.csv"]),
    ]
    for title, patterns in table_sections:
        path = find_first(patterns, run_dir)
        if path:
            df = load_table(path)
            if df is not None:
                st.subheader(title)
                st.dataframe(df.head(200), use_container_width=True)

    render_downloads(collect_output_files(run_dir), run_dir)


st.set_page_config(page_title="Cyanopeptide Pipeline", layout="wide")
st.title("CPM – Cyanopeptide Pipeline")
st.caption("Supporting Python modules are bundled with the app. Polarity is fixed to POSITIVE.")

try:
    backend, backend_path = load_backend_module()
    library_path = resolve_library_path()
except Exception as exc:
    st.error(str(exc))
    st.stop()

if "results_dir" not in st.session_state:
    st.session_state["results_dir"] = None

class_options = ["MC", "MP", "AR", "AB", "MG"]
class_tag = st.selectbox(
    "Cyanopeptide class",
    class_options,
    format_func=lambda k: f"{k} — {backend.CLASS_CONFIGS[k]['LIB_CLASS_FILTER']}",
)

st.subheader("Input mzML files")
uploaded_files = st.file_uploader(
    "Upload one or more mzML files",
    type=["mzml", "mzML"],
    accept_multiple_files=True,
)

saved_files: list[Path] = []
if uploaded_files:
    saved_files = [save_and_fix_uploaded_mzml(f) for f in uploaded_files]
    st.success(f"{len(saved_files)} file(s) ready for analysis.")
    with st.expander("Files to analyze", expanded=False):
        for path in saved_files:
            st.code(str(path))

st.markdown("### Analysis settings")
col1, col2, col3 = st.columns(3)
with col1:
    rt_min = st.number_input("RT min (minutes)", min_value=0.0, max_value=100.0, value=DEFAULT_RT_WINDOW[0], step=0.1)
with col2:
    rt_max = st.number_input("RT max (minutes)", min_value=0.0, max_value=100.0, value=DEFAULT_RT_WINDOW[1], step=0.1)
with col3:
    tol_da = st.number_input("CyanoMetDB tolerance (Da)", min_value=0.0001, max_value=5.0, value=DEFAULT_TOL_DA, step=0.0001)

col4, col5, col6 = st.columns(3)
with col4:
    ref_mz_text = st.text_input("Reference m/z for normalization (optional)", value="")
with col5:
    extract_ms1 = st.checkbox("Extract MS1 points from uploaded mzML", value=False)
with col6:
    do_blank_filter = st.checkbox("Apply blank filter", value=True)

do_batch_correct = st.checkbox("Apply batch correction", value=True)
metadata_file = st.file_uploader("Optional metadata CSV", type=["csv"])
ms1_points_file = None if extract_ms1 else st.file_uploader("Optional MS1 points CSV", type=["csv"])

st.info(f"CyanoMetDB library is preloaded from: {library_path}")
st.caption(f"Backend loaded from: {backend_path.name}")

run_clicked = st.button("Run analysis", type="primary")

if run_clicked:
    if not saved_files:
        st.error("Please upload at least one mzML file.")
    elif rt_max < rt_min:
        st.error("RT max must be greater than or equal to RT min.")
    else:
        ref_mz = None
        if ref_mz_text.strip():
            try:
                ref_mz = float(ref_mz_text)
            except ValueError:
                st.error("Reference m/z must be numeric.")
                st.stop()

        run_stamp = re.sub(r"[^0-9A-Za-z_-]", "_", class_tag)
        output_root = RUNS_DIR / f"{run_stamp}"
        output_root.mkdir(parents=True, exist_ok=True)

        metadata_path = None
        if metadata_file is not None:
            metadata_path = output_root / metadata_file.name
            metadata_path.write_bytes(metadata_file.getvalue())

        ms1_points_path = None
        if ms1_points_file is not None:
            ms1_points_path = output_root / ms1_points_file.name
            ms1_points_path.write_bytes(ms1_points_file.getvalue())

        log_capture = io.StringIO()
        with st.spinner(f"Running {class_tag} pipeline..."):
            try:
                with contextlib.redirect_stdout(log_capture), contextlib.redirect_stderr(log_capture):
                    run_dir = backend.run_pipeline_notebook(
                        class_tag=class_tag,
                        files=[str(p) for p in saved_files],
                        metadata_path=str(metadata_path) if metadata_path else None,
                        CyanoMetDBLibrary=str(library_path),
                        ms1_points_file=str(ms1_points_path) if ms1_points_path else None,
                        output_root=output_root,
                        extract_ms1=extract_ms1,
                        tol=DEFAULT_TOL,
                        polarity=POLARITY,
                        rt_window=(float(rt_min), float(rt_max)),
                        ref_rt_window=DEFAULT_REF_RT_WINDOW,
                        ref_mz=ref_mz,
                        ref_tol=0.01,
                        do_blank_filter=do_blank_filter,
                        do_batch_correct=do_batch_correct,
                        tol_da=float(tol_da),
                    )
                st.session_state["results_dir"] = str(run_dir)
                st.session_state["inline_log"] = log_capture.getvalue()
            except Exception as exc:
                st.session_state["inline_log"] = log_capture.getvalue()
                st.error(f"Pipeline failed: {exc}")

results_dir = st.session_state.get("results_dir")
if results_dir:
    run_dir = Path(results_dir)
    if run_dir.exists():
        inline_log = st.session_state.get("inline_log", "")
        if inline_log:
            with st.expander("Live run output", expanded=False):
                st.text(inline_log)
        render_results(run_dir)
else:
    st.info("Run the analysis to see results.")

st.divider()
st.subheader("Need help?")
st.info("If you see an error or something looks off, contact Sierra Hefferan @sheffera@umich.edu.")
