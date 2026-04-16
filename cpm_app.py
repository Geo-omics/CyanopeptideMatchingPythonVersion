from __future__ import annotations

import builtins
import contextlib
import importlib.util
import io
import shutil
import sys
import tempfile
import uuid
import zipfile
from datetime import datetime
from pathlib import Path
import re

import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
TEMP_BASE = Path(tempfile.gettempdir()) / "cpm_app"
TEMP_BASE.mkdir(parents=True, exist_ok=True)

POLARITY = "POSITIVE"
DEFAULT_RT_WINDOW = (2.0, 25.0)
DEFAULT_TOL = 0.01
DEFAULT_TOL_DA = 0.1
DEFAULT_REF_MZ = 198.135
DEFAULT_REF_RT_WINDOW = (6.8, 7.2)
DEFAULT_REF_TOL = 0.01
MAX_ZIP_MB = 500
MZML_ID_PATTERN = re.compile(r'id="merged=(\d+)\s+row=\d+"')


# -----------------------------
# Safe text helpers
# -----------------------------
def safe_text(value) -> str:
    text = str(value)
    replacements = {
        "\u2192": "->",
        "\u2190": "<-",
        "\u2013": "-",
        "\u2014": "-",
        "\u2212": "-",
        "\u2026": "...",
        "\u00a0": " ",
    }
    for bad, good in replacements.items():
        text = text.replace(bad, good)
    return text.encode("cp1252", errors="replace").decode("cp1252")


@contextlib.contextmanager
def patched_print():
    original_print = builtins.print

    def safe_print(*args, **kwargs):
        safe_args = tuple(safe_text(arg) for arg in args)
        sep = kwargs.get("sep", " ")
        end = kwargs.get("end", "\n")
        kwargs["sep"] = safe_text(sep)
        kwargs["end"] = safe_text(end)
        return original_print(*safe_args, **kwargs)

    builtins.print = safe_print
    try:
        yield
    finally:
        builtins.print = original_print


# -----------------------------
# Backend + session
# -----------------------------
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
    for p in [DATA_DIR / "CyanoMetDB_Version03.xlsx", BASE_DIR / "CyanoMetDB_Version03.xlsx"]:
        if p.exists():
            return p
    raise FileNotFoundError(
        "CyanoMetDB_Version03.xlsx was not found. Put it in ./data/ or beside this app."
    )


def get_session_root() -> Path:
    if "session_id" not in st.session_state:
        st.session_state["session_id"] = uuid.uuid4().hex[:10]
    root = TEMP_BASE / st.session_state["session_id"]
    (root / "uploads").mkdir(parents=True, exist_ok=True)
    (root / "runs").mkdir(parents=True, exist_ok=True)
    return root


def stage_bundled_library(session_root: Path) -> Path:
    src = resolve_library_path()
    bundled_dir = session_root / "bundled_data"
    bundled_dir.mkdir(parents=True, exist_ok=True)
    staged = bundled_dir / src.name
    if not staged.exists() or staged.stat().st_size != src.stat().st_size:
        shutil.copy2(src, staged)
    return staged


# -----------------------------
# Filesystem helpers
# -----------------------------
def save_and_fix_uploaded_mzml(uploaded_file, upload_dir: Path) -> Path:
    upload_dir.mkdir(parents=True, exist_ok=True)
    out_path = upload_dir / uploaded_file.name
    raw_bytes = uploaded_file.getvalue()
    try:
        text = raw_bytes.decode("utf-8")
    except UnicodeDecodeError:
        out_path.write_bytes(raw_bytes)
        return out_path

    fixed_text, _ = MZML_ID_PATTERN.subn(r'id="scan=\1"', text)
    out_path.write_text(fixed_text, encoding="utf-8")
    return out_path


def collect_output_files(run_dir: Path) -> list[Path]:
    allowed = {".csv", ".tsv", ".txt", ".xlsx", ".xls", ".png", ".jpg", ".jpeg", ".svg", ".pdf", ".json"}
    files: list[Path] = []
    for p in run_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in allowed:
            files.append(p)
    return sorted(files)


def normalize_pipeline_result(result, output_root: Path) -> tuple[list[Path], list[Path]]:
    roots: list[Path] = []
    files: list[Path] = []

    def _add_path(value):
        if value is None:
            return
        try:
            p = Path(value)
        except TypeError:
            return
        if p.exists():
            if p.is_dir():
                roots.append(p)
            else:
                files.append(p)

    if isinstance(result, dict):
        for run_dir in result.get("run_dirs", []) or []:
            _add_path(run_dir)
        _add_path(result.get("pipeline_log_dir"))
        _add_path(result.get("pipeline_log_file"))
        _add_path(result.get("ms1_points_file"))

    if output_root.exists():
        discovered_run_dirs = sorted(
            [p for p in output_root.rglob("*_run_*") if p.is_dir()],
            key=lambda p: p.stat().st_mtime if p.exists() else 0,
            reverse=True,
        )
        for p in discovered_run_dirs:
            roots.append(p)

        for extra in [output_root / "pipeline_log", output_root / "MS1_points"]:
            if extra.exists():
                roots.append(extra)

    if not roots and output_root.exists():
        roots.append(output_root)

    dedup_roots = []
    seen = set()
    for p in roots:
        rp = p.resolve()
        if rp not in seen:
            seen.add(rp)
            dedup_roots.append(p)

    dedup_files = []
    seen_files = set()
    for p in files:
        rp = p.resolve()
        if rp not in seen_files:
            seen_files.add(rp)
            dedup_files.append(p)

    return dedup_roots, dedup_files


def collect_output_files_from_result(result, output_root: Path) -> tuple[list[Path], list[Path], list[Path]]:
    roots, explicit_files = normalize_pipeline_result(result, output_root)
    collected: list[Path] = []

    for root in roots:
        if root.is_dir():
            collected.extend(collect_output_files(root))
        elif root.is_file():
            collected.append(root)

    for p in explicit_files:
        if p.is_file():
            collected.append(p)

    uniq = []
    seen = set()
    for p in collected:
        rp = p.resolve()
        if rp not in seen and p.exists():
            seen.add(rp)
            uniq.append(p)

    return sorted(uniq), roots, explicit_files


def make_zip_bytes(paths: list[Path], root: Path) -> bytes:
    buffer = io.BytesIO()
    used_names: set[str] = set()

    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in paths:
            if not path.exists() or not path.is_file():
                continue

            try:
                arcname = path.relative_to(root)
            except Exception:
                arcname = Path(path.name)

            arcname_str = str(arcname).replace("\\", "/")
            if arcname_str in used_names:
                arcname_str = f"{path.parent.name}/{path.name}"
            if arcname_str in used_names and path.parent.parent != path.parent:
                arcname_str = f"{path.parent.parent.name}/{path.parent.name}/{path.name}"

            used_names.add(arcname_str)
            zf.write(path, arcname=arcname_str)

    return buffer.getvalue()


def cleanup_path(path: Path | None) -> None:
    if path and path.exists():
        shutil.rmtree(path, ignore_errors=True)


# -----------------------------
# Preview extraction
# -----------------------------
def latest_match(
    paths: list[Path],
    *,
    prefix: str | None = None,
    suffix: str | None = None,
    contains: str | None = None,
    exclude_contains: list[str] | None = None,
    prefer_contains: list[str] | None = None,
) -> Path | None:
    exclude_contains = exclude_contains or []
    prefer_contains = prefer_contains or []
    candidates = []

    for p in paths:
        name = p.name
        if prefix and not name.startswith(prefix):
            continue
        if suffix and not name.endswith(suffix):
            continue
        if contains and contains not in name:
            continue
        if any(x in name for x in exclude_contains):
            continue

        score = 0
        for i, pref in enumerate(prefer_contains):
            if pref in name:
                score += 100 - i

        candidates.append((score, p.stat().st_mtime if p.exists() else 0, p))

    if not candidates:
        return None

    candidates.sort(key=lambda x: (x[0], x[1]))
    return candidates[-1][2]


def load_bytes(path: Path | None) -> bytes | None:
    if path and path.exists() and path.is_file():
        return path.read_bytes()
    return None


def load_table_preview(path: Path | None, head: int = 100) -> dict | None:
    if path is None or not path.exists() or not path.is_file():
        return None

    try:
        if path.suffix.lower() == ".csv":
            df = pd.read_csv(path)
        elif path.suffix.lower() == ".tsv":
            df = pd.read_csv(path, sep="\t")
        elif path.suffix.lower() in {".xlsx", ".xls"}:
            df = pd.read_excel(path)
        else:
            return None

        return {
            "columns": list(df.columns),
            "rows": df.head(head).to_dict(orient="records"),
            "n_rows": int(len(df)),
        }
    except Exception:
        return None


def build_previews(files: list[Path], class_tag: str) -> dict:
    previews: dict = {}

    previews["cyano_heatmap_png"] = load_bytes(
        latest_match(files, suffix="_heatmap.png", contains="indiv_merged_summary_with_intensities_")
    )
    previews["rt_plot_png"] = load_bytes(latest_match(files, prefix="Precursor_rt_plot_", suffix=".png"))
    previews["dot_plot_png"] = load_bytes(latest_match(files, prefix="indiv_diagnostic_ions_", suffix=".png"))
    previews["diagnostic_individual_png"] = load_bytes(
        latest_match(files, prefix="Diagnostic_ion_distribution_individual_", suffix=".png", exclude_contains=["stacked"])
    )
    previews["matched_tiles_png"] = load_bytes(latest_match(files, prefix="matched_compound_tiles_", suffix=".png"))
    previews["unknown_features_png"] = load_bytes(
        latest_match(files, prefix="unknown_features_with_scans_", suffix=".png", prefer_contains=["CLEAN", "RAW"])
    )
    previews["adduct_graph_png"] = load_bytes(latest_match(files, prefix="adduct_graph_merged_", suffix=".png"))

    previews["ind_hits_table"] = load_table_preview(
        latest_match(files, prefix=f"individual_hits_{class_tag}_", suffix=".csv"), head=5
    )
    previews["indiv_merged_table"] = load_table_preview(
        latest_match(files, prefix="indiv_merged_summary_", suffix=".csv", exclude_contains=["with_intensities", "best_edges"]),
        head=5,
    )
    previews["unknown_features_table"] = load_table_preview(
        latest_match(files, prefix="unknown_features_with_scans_", suffix=".csv", prefer_contains=["CLEAN", "RAW"]),
        head=100,
    )

    counts = {"csv": 0, "excel": 0, "images": 0, "other": 0}
    for p in files:
        suf = p.suffix.lower()
        if suf in {".csv", ".tsv"}:
            counts["csv"] += 1
        elif suf in {".xlsx", ".xls"}:
            counts["excel"] += 1
        elif suf in {".png", ".jpg", ".jpeg", ".svg", ".pdf"}:
            counts["images"] += 1
        else:
            counts["other"] += 1
    previews["file_type_counts"] = counts
    return previews


# -----------------------------
# Session state helpers
# -----------------------------
def reset_download_state() -> None:
    for key in [
        "zip_bytes",
        "zip_name",
        "run_summary",
        "download_status",
        "download_status_detail",
        "download_ready",
        "download_consumed",
        "run_error",
        "inline_log",
        "previews",
    ]:
        st.session_state.pop(key, None)


def clear_session_memory() -> None:
    cleanup_path(TEMP_BASE / st.session_state.get("session_id", ""))
    reset_download_state()
    for key in ["last_saved_files", "session_id"]:
        st.session_state.pop(key, None)


def consume_download() -> None:
    st.session_state["zip_bytes"] = None
    st.session_state["zip_name"] = None
    st.session_state["download_ready"] = False
    st.session_state["download_consumed"] = True
    st.session_state["download_status"] = "downloaded"
    st.session_state["download_status_detail"] = (
        "Download started. The in-memory ZIP has now been removed from the app session. "
        "Run the pipeline again if you need another copy."
    )


# -----------------------------
# UI config
# -----------------------------
st.set_page_config(page_title="Cyanopeptide Pipeline", layout="wide")
st.markdown(
    """
    <style>
    div.stDownloadButton > button[kind="primary"] {
        background-color: #2563eb;
        border: 1px solid #2563eb;
        color: white;
    }
    div.stDownloadButton > button[kind="primary"]:hover {
        background-color: #1d4ed8;
        border-color: #1d4ed8;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def render_home_page():
    st.title("CPM – Cyanopeptide Metabolomics Pipeline")
    st.subheader("What this app does")

    st.markdown(
        """
        This application screens LC-MS/MS mzML files for cyanopeptide classes using class-specific
        diagnostic ions, summarizes precursor features, performs optional QC and blank handling,
        links related adduct features, and compares putative matches against the bundled
        CyanoMetDB reference library.

        **Workflow overview**
        1. Upload one or more mzML files.
        2. Optionally upload a metadata CSV for blank filtering and batch correction.
        3. Optionally enable reference compound normalization.
        4. Run the selected cyanopeptide class pipeline.
        5. Review plots, tables, and annotations on screen.
        6. Download a ZIP of all generated outputs.
        """
    )

    st.caption(
        "CyanoMetDB reference: Jones MR et al., CyanoMetDB, a comprehensive public "
        "database of secondary metabolites from cyanobacteria, Water Research 196 "
        "(2021) 117017. https://doi.org/10.1016/j.watres.2021.117017; "
        "Janssen et al., 2024, DOI: 10.5281/zenodo.13854577"
    )

    st.subheader("What the metadata file is for")
    st.markdown(
        """
        The metadata CSV is optional, but recommended when you want blank filtering and/or
        batch correction. The most useful columns are:

        - `source_file`: exact mzML filename
        - `sample_type`: for example `sample` or `blank`
        - `batch`: batch number or batch label

        You can include additional columns for your own recordkeeping.

        **Optional reference normalization**
        If you have a reference compound, you can enable reference normalization on the Run page and provide:
        - a reference precursor m/z
        - a retention time window to search for that compound
        - an m/z tolerance for matching

        If you do not have a reference compound, leave this section off and the pipeline will skip normalization.
        """
    )

    example_meta = pd.DataFrame(
        [
            {"source_file": "meoh.mzML", "sample_type": "blank", "batch": 1, "sample_id": "blank_01"},
            {"source_file": "mp_m2.mzML", "sample_type": "sample", "batch": 1, "sample_id": "sample_01"},
            {"source_file": "mpbr_ms2.mzML", "sample_type": "sample", "batch": 1, "sample_id": "sample_02"},
        ]
    )
    st.markdown("**Example metadata table**")
    st.dataframe(example_meta, use_container_width=True)
    st.download_button(
        "Download example metadata CSV",
        data=example_meta.to_csv(index=False).encode("utf-8"),
        file_name="example_metadata.csv",
        mime="text/csv",
        key="example-metadata-download",
    )

    with st.expander("Notes about the bundled reference library", expanded=False):
        st.write(
            "CyanoMetDB is bundled with the app package. Keep `CyanoMetDB_Version03.xlsx` "
        )

    st.info("Use the sidebar to switch to **Run pipeline** when you're ready to analyze mzML files.")


def render_run_page():
    st.title("CPM – Cyanopeptide Pipeline")
    st.caption(
        "Supporting Python modules are bundled with the app. Polarity is fixed to POSITIVE. "
        "Download-only mode uses a temporary workspace and does not keep permanent outputs."
    )

    try:
        backend, backend_path = load_backend_module()
        library_path = resolve_library_path()
    except Exception as exc:
        st.error(str(exc))
        st.stop()

    session_root = get_session_root()
    upload_dir = session_root / "uploads"
    runs_dir = session_root / "runs"
    staged_library_path = stage_bundled_library(session_root)

    st.info("CyanoMetDB library is bundled with the app and staged automatically for each run.")
    with st.expander("Show app resources"):
        st.caption(f"Bundled library source: {library_path.name}")
        st.caption(f"Backend loaded from: {backend_path.name}")

    for key, default in {
        "last_saved_files": [],
        "download_status": "idle",
        "download_status_detail": "Run the analysis to generate a downloadable ZIP.",
        "download_ready": False,
        "download_consumed": False,
        "previews": {},
    }.items():
        if key not in st.session_state:
            st.session_state[key] = default

    download_status_placeholder = st.container()

    class_options = ["MC", "MP", "AR", "AB", "MG"]
    class_tag = st.selectbox(
        "Cyanopeptide class",
        class_options,
        format_func=lambda k: f"{k} - {backend.CLASS_CONFIGS[k]['LIB_CLASS_FILTER']}",
    )

    st.subheader("Input mzML files")
    uploaded_files = st.file_uploader(
        "Upload one or more mzML files",
        type=["mzml", "mzML"],
        accept_multiple_files=True,
    )

    saved_files: list[Path] = []
    if uploaded_files:
        saved_files = [save_and_fix_uploaded_mzml(f, upload_dir) for f in uploaded_files]
        st.session_state["last_saved_files"] = [str(p) for p in saved_files]
        st.success(f"{len(saved_files)} file(s) ready for analysis.")
    elif st.session_state.get("last_saved_files"):
        saved_files = [Path(p) for p in st.session_state["last_saved_files"] if Path(p).exists()]

    with st.expander("Analysis settings", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            rt_min = st.number_input(
                "RT min (minutes)",
                min_value=0.0,
                max_value=100.0,
                value=DEFAULT_RT_WINDOW[0],
                step=0.1,
            )
        with col2:
            rt_max = st.number_input(
                "RT max (minutes)",
                min_value=0.0,
                max_value=100.0,
                value=DEFAULT_RT_WINDOW[1],
                step=0.1,
            )
        with col3:
            tol_da = st.number_input(
                "CyanoMetDB tolerance (Da)",
                min_value=0.0001,
                max_value=5.0,
                value=DEFAULT_TOL_DA,
                step=0.0001,
            )

        col4, _, _ = st.columns(3)
        with col4:
            extract_ms1 = st.checkbox("Extract MS1 points from uploaded mzML", value=True)

        use_reference = st.checkbox(
            "Use reference compound normalization",
            value=False,
            help="Enable this only if you have a reference compound for normalization.",
        )

        ref_mz = None
        ref_tol = DEFAULT_REF_TOL
        ref_rt_window = None

        if use_reference:
            st.markdown("**Reference compound settings**")
            ref_col1, ref_col2, ref_col3 = st.columns(3)
            with ref_col1:
                ref_mz = st.number_input(
                    "Reference compound m/z",
                    min_value=0.0,
                    value=DEFAULT_REF_MZ,
                    step=0.0001,
                    format="%.4f",
                    help="Reference compound precursor m/z.",
                )
            with ref_col2:
                ref_rt_min = st.number_input(
                    "Reference RT min (minutes)",
                    min_value=0.0,
                    value=DEFAULT_REF_RT_WINDOW[0],
                    step=0.1,
                    help="Lower bound of the RT window used to search for your reference compound.",
                )
            with ref_col3:
                ref_rt_max = st.number_input(
                    "Reference RT max (minutes)",
                    min_value=0.0,
                    value=DEFAULT_REF_RT_WINDOW[1],
                    step=0.1,
                    help="Upper bound of the RT window used to search for your reference compound.",
                )

            ref_tol = st.number_input(
                "Reference m/z tolerance (Da)",
                min_value=0.0001,
                max_value=5.0,
                value=DEFAULT_REF_TOL,
                step=0.0001,
                format="%.4f",
                help="Allowed m/z tolerance for the reference compound.",
            )
            ref_rt_window = (float(ref_rt_min), float(ref_rt_max))

        st.markdown("**Optional metadata and QC settings**")
        metadata_file = st.file_uploader("Optional metadata CSV", type=["csv"])

        do_blank_filter = False
        do_batch_correct = False
        if metadata_file is not None:
            st.caption("Metadata detected. Blank filtering and batch correction options are now available.")
            qc_col1, qc_col2 = st.columns(2)
            with qc_col1:
                do_blank_filter = st.checkbox("Apply blank filter", value=True)
            with qc_col2:
                do_batch_correct = st.checkbox("Apply batch correction", value=True)

        ms1_points_file = None
        if not extract_ms1:
            ms1_points_file = st.file_uploader("Optional MS1 points CSV", type=["csv"])

    col_run, col_clear = st.columns([3, 1])
    with col_run:
        run_clicked = st.button("Run analysis", type="primary")
    with col_clear:
        if st.button("Clear session"):
            clear_session_memory()
            st.rerun()

    if run_clicked:
        reset_download_state()
        st.session_state["download_consumed"] = False

        if not saved_files:
            st.session_state["download_status"] = "error"
            st.session_state["download_status_detail"] = "Please upload at least one mzML file before running the pipeline."
        elif rt_max < rt_min:
            st.session_state["download_status"] = "error"
            st.session_state["download_status_detail"] = "RT max must be greater than or equal to RT min."
        else:
            ref_inputs_valid = True
            if use_reference and ref_rt_window is not None and ref_rt_window[1] < ref_rt_window[0]:
                st.session_state["download_status"] = "error"
                st.session_state["download_status_detail"] = (
                    "Reference RT max must be greater than or equal to Reference RT min."
                )
                ref_inputs_valid = False

            if ref_inputs_valid:
                stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_root = runs_dir / class_tag / stamp
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
                st.session_state["download_status"] = "running"
                st.session_state["download_status_detail"] = (
                    "Pipeline is running and outputs will be packaged into a ZIP when complete."
                )

                try:
                    with st.spinner(f"Running {class_tag} pipeline..."):
                        with patched_print(), contextlib.redirect_stdout(log_capture), contextlib.redirect_stderr(log_capture):
                            pipeline_result = backend.run_pipeline_notebook(
                                class_tag=class_tag,
                                files=[str(p) for p in saved_files],
                                metadata_path=str(metadata_path) if metadata_path else None,
                                CyanoMetDBLibrary=str(staged_library_path),
                                ms1_points_file=str(ms1_points_path) if ms1_points_path else None,
                                output_root=output_root,
                                extract_ms1=extract_ms1,
                                tol=DEFAULT_TOL,
                                polarity=POLARITY,
                                rt_window=(float(rt_min), float(rt_max)),
                                ref_rt_window=ref_rt_window,
                                ref_mz=ref_mz,
                                ref_tol=float(ref_tol),
                                do_blank_filter=do_blank_filter,
                                do_batch_correct=do_batch_correct,
                                tol_da=float(tol_da),
                            )

                    output_files, roots, explicit_files = collect_output_files_from_result(pipeline_result, output_root)
                    if not output_files and output_root.exists():
                        output_files = collect_output_files(output_root)
                        if not roots:
                            roots = [output_root]

                    if not output_files:
                        raise RuntimeError("The pipeline finished but no output files were found to package.")

                    total_bytes = sum(p.stat().st_size for p in output_files if p.exists())
                    total_mb = total_bytes / (1024 * 1024)
                    if total_mb > MAX_ZIP_MB:
                        raise RuntimeError(
                            f"Output package is about {total_mb:.1f} MB, which exceeds the in-memory ZIP limit of {MAX_ZIP_MB} MB."
                        )

                    previews = build_previews(output_files, class_tag)
                    zip_bytes = make_zip_bytes(output_files, session_root)
                    source_roots = [str(p) for p in roots] + [str(p) for p in explicit_files if p.is_file()]

                    st.session_state["zip_bytes"] = zip_bytes
                    st.session_state["zip_name"] = f"CPM_{class_tag}_{stamp}.zip"
                    st.session_state["inline_log"] = safe_text(log_capture.getvalue())
                    st.session_state["previews"] = previews
                    st.session_state["run_summary"] = {
                        "class_tag": class_tag,
                        "file_count": len(saved_files),
                        "zip_size_mb": len(zip_bytes) / (1024 * 1024),
                        "output_count": len(output_files),
                        "source_roots": source_roots,
                        "root_folder": str(output_root),
                    }
                    st.session_state["download_status"] = "ready"
                    st.session_state["download_status_detail"] = (
                        "ZIP package is ready below. Temporary run files were deleted after packaging."
                    )
                    st.session_state["download_ready"] = True
                except Exception as exc:
                    st.session_state["inline_log"] = safe_text(log_capture.getvalue())
                    st.session_state["download_status"] = "error"
                    st.session_state["download_status_detail"] = safe_text(exc)
                finally:
                    cleanup_path(session_root)
                    st.session_state["last_saved_files"] = []

    zip_bytes = st.session_state.get("zip_bytes")
    zip_name = st.session_state.get("zip_name")
    run_summary = st.session_state.get("run_summary")
    inline_log = st.session_state.get("inline_log", "")
    status = st.session_state.get("download_status", "idle")
    status_detail = st.session_state.get("download_status_detail", "")
    previews = st.session_state.get("previews", {})

    with download_status_placeholder:
        if status == "running":
            st.info("Pipeline is running. Scroll down after completion for the blue download button.")
            st.write(status_detail)
        elif status == "error":
            st.error("No ZIP available")
            st.write(status_detail)
        elif status == "ready" and run_summary:
            st.success("Run complete. Review the outputs below, then download the ZIP immediately after them.")
            st.write(
                f"Class: {run_summary['class_tag']} | Inputs: {run_summary['file_count']} | "
                f"Packaged files: {run_summary['output_count']} | ZIP size: {run_summary['zip_size_mb']:.1f} MB"
            )
        elif status == "downloaded":
            st.success("ZIP removed from session")
            st.write(status_detail)
        else:
            st.info("No ZIP yet")
            st.write(status_detail)

    if status in {"ready", "downloaded"} and previews:
        st.subheader("Run outputs")

        heatmap_png = previews.get("cyano_heatmap_png")
        if heatmap_png:
            st.subheader("Specific cyanopeptide-class detection intensity heatmap")
            st.image(heatmap_png)
            st.markdown(
                "<p style='text-align:center; font-size:15px; color:black;'>"
                "Output indicates presence (measured by sum of intensities) of cyanopeptides found in sample(s)<br>"
                "</p>",
                unsafe_allow_html=True,
            )

        ind_hits = previews.get("ind_hits_table")
        if ind_hits:
            st.subheader("Individual hits (labeled) – preview")
            st.dataframe(pd.DataFrame(ind_hits["rows"], columns=ind_hits["columns"]))

        rt_png = previews.get("rt_plot_png")
        if rt_png:
            st.subheader("Precursor RT plot")
            st.image(rt_png)
            st.markdown(
                "<p style='text-align:center; font-size:15px; color:black;'>"
                "Retention time plot versus diagnostic product ion detected for cyanopeptide class<br>"
                "</p>",
                unsafe_allow_html=True,
            )

        dot_png = previews.get("dot_plot_png")
        if dot_png:
            st.subheader("Individual ion dot plot")
            st.image(dot_png)
            st.markdown(
                "<p style='text-align:center; font-size:15px; color:black;'>"
                "Scatter plot of diagnostic ion detection across precursor m/z values. Each point represents the presence of a diagnostic ion associated with a given precursor ion. Point size is scaled by the number of scans in which the precursor was observed, reflecting relative abundance or detection frequency, while color indicates the source file.<br>"
                "</p>",
                unsafe_allow_html=True,
            )

        diag_png = previews.get("diagnostic_individual_png")
        if diag_png:
            st.subheader("Diagnostic ion distribution – individual")
            st.image(diag_png)
            st.markdown(
                "<p style='text-align:center; font-size:15px; color:black;'>"
                "Counts of each diagnostic product ion for each file detected. <br>"
                "</p>",
                unsafe_allow_html=True,
            )

        matched_tiles_png = previews.get("matched_tiles_png")
        if matched_tiles_png:
            st.subheader("Matched compound tiles (Putative annotations)")
            st.image(matched_tiles_png)
            st.markdown(
                "<p style='text-align:center; font-size:15px; color:black;'>"
                "Level 3 identification of putative matches to cyanopeptides based on m/z, class-specific CyanoMetDB search, and presence of specific diagnostic product ions. <br>"
                "</p>",
                unsafe_allow_html=True,
            )
        else:
            st.info("No 'matched_compound_tiles' PNG found for this run.")

        adduct_graph_png = previews.get("adduct_graph_png")
        if adduct_graph_png:
            st.subheader("Adduct graph (merged) if messy please open Parent Adduct Summary Excel")
            st.image(adduct_graph_png)
            st.markdown(
                "<p style='text-align:center; font-size:15px; color:black;'>"
                "Adduct network of compounds related by common adducts. Use this to compare your matched and putative novel congeners! <br>"
                "</p>",
                unsafe_allow_html=True,
            )
        else:
            st.info("No 'adduct_graph_merged_' PNG found for this run.")

        indiv_merged = previews.get("indiv_merged_table")
        if indiv_merged:
            st.subheader("Individual merged summary")
            st.dataframe(pd.DataFrame(indiv_merged["rows"], columns=indiv_merged["columns"]))

        unknown_table = previews.get("unknown_features_table")
        if unknown_table and unknown_table["rows"]:
            st.subheader("Unknown features with scans (table)")
            st.dataframe(pd.DataFrame(unknown_table["rows"], columns=unknown_table["columns"]))
            st.markdown(
                "<p style='text-align:center; font-size:15px; color:black;'>"
                "Unknown putative novel congeners with 2 or more diagnostic product ions for your class of compounds. Remember to compare with your adduct network! The putative novel congeners list can aid in prioritization of metabolites for further characterization analysis.<br>"
                "</p>",
                unsafe_allow_html=True,
            )
        else:
            st.info("No unknown features detected.")

        st.divider()
        st.subheader("Download Results")
        downloaded = st.download_button(
            "Download All Outputs (.zip)",
            data=zip_bytes,
            file_name=zip_name,
            mime="application/zip",
            type="primary",
            use_container_width=True,
            key="bottom-download-button",
        )
        st.caption("After the first successful click, the ZIP is removed from app memory.")

        if run_summary:
            with st.expander("Show packaged folders", expanded=False):
                for folder in run_summary.get("source_roots", []):
                    st.code(folder)

        if downloaded:
            consume_download()
            st.rerun()

    if inline_log:
        with st.expander("Run log", expanded=False):
            st.text(inline_log)

    st.divider()
    st.subheader("Need help?")
    st.info("If you see an error or something looks off, contact Sierra Hefferan @sheffera@umich.edu.")


page = st.sidebar.radio("Navigation", ["About this app", "Run pipeline"], index=0)

if page == "About this app":
    render_home_page()
else:
    render_run_page()
