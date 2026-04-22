from __future__ import annotations

import builtins
import contextlib
import hashlib
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

APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data"

POLARITY = "POSITIVE"
DEFAULT_RT_WINDOW = (2.0, 25.0)
DEFAULT_TOL = 0.01
DEFAULT_TOL_DA = 0.1
DEFAULT_REF_MZ = 198.135
DEFAULT_REF_RT_WINDOW = (6.8, 7.2)
DEFAULT_REF_TOL = 0.01
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
# Save-root helpers
# -----------------------------
def get_default_save_root() -> Path:
    return Path(tempfile.gettempdir()) / "CPM_Output"

def normalize_save_root(save_root_text: str) -> Path:
    text = (save_root_text or "").strip()
    if not text:
        return get_default_save_root()
    return Path(text).expanduser()


def get_paths(save_root_text: str) -> dict[str, Path]:
    root = normalize_save_root(save_root_text)
    session_id = st.session_state.get("session_id")
    if not session_id:
        session_id = uuid.uuid4().hex[:10]
        st.session_state["session_id"] = session_id

    paths = {
        "root": root,
        "uploads": root / "uploads",
        "metadata": root / "metadata",
        "ms1_points_uploads": root / "uploaded_ms1_points",
        "runs": root / "runs",
        "downloads": root / "zips",
        "logs": root / "logs",
        "bundled_data": root / "bundled_data",
        "session_temp": root / "_session_temp" / session_id,
    }

    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)

    return paths


# -----------------------------
# Backend + library
# -----------------------------
@st.cache_resource(show_spinner=False)
def load_backend_module():
    candidates = sorted(APP_DIR.glob("CPM_cli*.py")) + sorted(APP_DIR.glob("*cli*.py"))
    if not candidates:
        raise FileNotFoundError(
            "No backend pipeline script found next to the app. "
            "Place your CPM_cli_*.py file beside this Streamlit app."
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
    for p in [DATA_DIR / "CyanoMetDB_Version03.xlsx", APP_DIR / "CyanoMetDB_Version03.xlsx"]:
        if p.exists():
            return p
    raise FileNotFoundError(
        "CyanoMetDB_Version03.xlsx was not found. Put it in ./data/ or beside this app."
    )


def stage_bundled_library(save_root_text: str) -> Path:
    src = resolve_library_path()
    paths = get_paths(save_root_text)
    staged = paths["bundled_data"] / src.name
    if not staged.exists() or staged.stat().st_size != src.stat().st_size:
        shutil.copy2(src, staged)
    return staged


# -----------------------------
# Persistent upload helpers
# -----------------------------
def _dedupe_path_by_name(upload_dir: Path, filename: str) -> Path:
    """
    Minimal-change version:
    - If the same filename already exists, reuse it.
    - Avoids reading whole old/new files into memory for comparison.
    """
    upload_dir.mkdir(parents=True, exist_ok=True)
    return upload_dir / filename


def save_and_fix_uploaded_mzml(uploaded_file, upload_dir: Path) -> Path:
    """
    Stream upload to disk and apply the mzML id fix line-by-line.
    This avoids:
    - uploaded_file.getvalue()
    - decoding the entire file into one giant string
    - creating multiple large in-memory copies
    """
    out_path = _dedupe_path_by_name(upload_dir, uploaded_file.name)

    uploaded_file.seek(0)

    with open(out_path, "wb") as out_f:
        for raw_line in uploaded_file:
            try:
                line = raw_line.decode("utf-8")
                line = MZML_ID_PATTERN.sub(r'id="scan=\1"', line)
                out_f.write(line.encode("utf-8"))
            except UnicodeDecodeError:
                # Fallback: write raw bytes if this line cannot be decoded
                out_f.write(raw_line)

    return out_path


def save_uploaded_binary(uploaded_file, upload_dir: Path) -> Path:
    """
    Stream any uploaded file to disk in chunks.
    Avoids uploaded_file.getvalue().
    """
    out_path = _dedupe_path_by_name(upload_dir, uploaded_file.name)

    uploaded_file.seek(0)
    with open(out_path, "wb") as out_f:
        shutil.copyfileobj(uploaded_file, out_f, length=1024 * 1024)

    return out_path

# -----------------------------
# Filesystem helpers
# -----------------------------
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
        roots.extend(discovered_run_dirs)

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


def build_zip_on_disk(paths: list[Path], root: Path, zip_path: Path) -> Path:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    used_names: set[str] = set()

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=3) as zf:
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

    return zip_path


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
        latest_match(
            files,
            prefix="indiv_merged_summary_",
            suffix=".csv",
            exclude_contains=["with_intensities", "best_edges"],
        ),
        head=5,
    )
    previews["unknown_features_table"] = load_table_preview(
        latest_match(files, prefix="unknown_features_with_scans_", suffix=".csv", prefer_contains=["CLEAN", "RAW"]),
        head=100,
    )

    return previews


def render_all_discovered_outputs(files: list[Path]) -> None:
    image_suffixes = {".png", ".jpg", ".jpeg"}
    table_suffixes = {".csv", ".tsv", ".xlsx", ".xls"}

    with st.expander("All discovered output files", expanded=False):
        for p in files:
            st.code(str(p))

    for p in files:
        suffix = p.suffix.lower()

        if suffix in image_suffixes:
            st.subheader(p.name)
            try:
                st.image(str(p))
            except Exception as exc:
                st.warning(f"Could not display image {p.name}: {exc}")

        elif suffix in table_suffixes:
            st.subheader(p.name)
            try:
                if suffix == ".csv":
                    df = pd.read_csv(p)
                elif suffix == ".tsv":
                    df = pd.read_csv(p, sep="\t")
                else:
                    df = pd.read_excel(p)
                st.dataframe(df.head(50), use_container_width=True)
            except Exception as exc:
                st.warning(f"Could not preview table {p.name}: {exc}")


# -----------------------------
# Session state helpers
# -----------------------------
def reset_download_state() -> None:
    for key in [
        "zip_path",
        "zip_name",
        "run_summary",
        "download_status",
        "download_status_detail",
        "download_ready",
        "download_consumed",
        "inline_log",
        "previews",
        "discovered_output_files",
    ]:
        st.session_state.pop(key, None)


def clear_session_state_only() -> None:
    reset_download_state()
    for key in ["last_saved_files", "last_metadata_file", "last_ms1_points_file", "session_id"]:
        st.session_state.pop(key, None)


def clear_folder_contents(folder: Path) -> None:
    if not folder.exists():
        return
    for item in folder.iterdir():
        if item.is_dir():
            shutil.rmtree(item, ignore_errors=True)
        else:
            item.unlink(missing_ok=True)


def consume_download() -> None:
    zip_path = st.session_state.get("zip_path")
    if zip_path:
        try:
            Path(zip_path).unlink(missing_ok=True)
        except Exception:
            pass

    st.session_state["zip_path"] = None
    st.session_state["zip_name"] = None
    st.session_state["download_ready"] = False
    st.session_state["download_consumed"] = True
    st.session_state["download_status"] = "downloaded"
    st.session_state["download_status_detail"] = (
        "Download started. The ZIP file was removed from the session download slot."
    )


# -----------------------------
# UI config
# -----------------------------
st.set_page_config(page_title="Cyanopeptide Pipeline", layout="wide")


def render_home_page():
    st.title("CPM – Cyanopeptide Metabolomics Pipeline")
    st.markdown(
        """
        This version lets you choose the base output folder inside Streamlit.

        Everything is saved under the folder you choose:
        - uploads
        - uploaded metadata
        - uploaded MS1 points
        - run outputs
        - ZIP files
        - staged library copy
        """
    )


    save_root_text = str(get_default_save_root())

    try:
        paths = get_paths(save_root_text)
    except Exception as exc:
        st.error(f"Could not create or access temporary working folder: {exc}")
        st.stop()

    staged_library_path = stage_bundled_library(save_root_text)


    st.subheader("Save location")

    default_save_root = str(get_default_save_root())

    save_root_text = st.text_input(
        "Base folder for uploads, runs, logs, and ZIPs",
        value=default_save_root,
        help="Example: C:\\Users\\you\\Documents\\CPM_Output or D:\\CPM_Output",
    )


    try:
        paths = get_paths(save_root_text)
    except Exception as exc:
        st.error(f"Could not create or access that save folder: {exc}")
        st.stop()

    staged_library_path = stage_bundled_library(save_root_text)

    with st.expander("Current save folders", expanded=False):
        for k, p in paths.items():
            st.code(f"{k}: {p}")
        st.caption(f"Backend loaded from: {backend_path.name}")
        st.caption(f"Bundled library source: {library_path.name}")

    for key, default in {
        "last_saved_files": [],
        "last_metadata_file": None,
        "last_ms1_points_file": None,
        "download_status": "idle",
        "download_status_detail": "Run the analysis to generate a downloadable ZIP.",
        "download_ready": False,
        "download_consumed": False,
        "previews": {},
        "discovered_output_files": [],
    }.items():
        if key not in st.session_state:
            st.session_state[key] = default

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
        try:
            saved_files = [save_and_fix_uploaded_mzml(f, paths["uploads"]) for f in uploaded_files]
            st.session_state["last_saved_files"] = [str(p) for p in saved_files]

            total_size_mb = sum(getattr(f, "size", 0) for f in uploaded_files) / (1024 * 1024)
            st.success(f"{len(uploaded_files)} file(s) selected.")
            st.caption(f"Total selected size: {total_size_mb:.1f} MB")

            with st.expander("Files selected for analysis", expanded=False):
                for f, p in zip(uploaded_files, saved_files):
                    size_mb = getattr(f, "size", 0) / (1024 * 1024)
                    st.code(f"{f.name} ({size_mb:.1f} MB) -> {p}")
        except Exception as exc:
            st.error(f"Failed while saving uploaded mzML files: {safe_text(exc)}")

    elif st.session_state.get("last_saved_files"):
        saved_files = [Path(p) for p in st.session_state["last_saved_files"] if Path(p).exists()]
        if saved_files:
            st.info(f"Using {len(saved_files)} previously saved mzML file(s).")
            with st.expander("Files that will be analyzed", expanded=False):
                for p in saved_files:
                    st.code(str(p))

    with st.expander("Analysis settings", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            rt_min = st.number_input("RT min (minutes)", min_value=0.0, max_value=100.0, value=2.0, step=0.1)
        with col2:
            rt_max = st.number_input("RT max (minutes)", min_value=0.0, max_value=100.0, value=25.0, step=0.1)
        with col3:
            tol_da = st.number_input("CyanoMetDB tolerance (Da)", min_value=0.0001, max_value=5.0, value=0.1, step=0.0001)

        extract_ms1 = st.checkbox("Extract MS1 points from uploaded mzML", value=True)
        use_reference = st.checkbox("Use reference compound normalization", value=False)

        ref_mz = None
        ref_tol = DEFAULT_REF_TOL
        ref_rt_window = None

        if use_reference:
            ref_col1, ref_col2, ref_col3 = st.columns(3)
            with ref_col1:
                ref_mz = st.number_input("Reference compound m/z", min_value=0.0, value=DEFAULT_REF_MZ, step=0.0001, format="%.4f")
            with ref_col2:
                ref_rt_min = st.number_input("Reference RT min (minutes)", min_value=0.0, value=DEFAULT_REF_RT_WINDOW[0], step=0.1)
            with ref_col3:
                ref_rt_max = st.number_input("Reference RT max (minutes)", min_value=0.0, value=DEFAULT_REF_RT_WINDOW[1], step=0.1)
            ref_tol = st.number_input("Reference m/z tolerance (Da)", min_value=0.0001, max_value=5.0, value=DEFAULT_REF_TOL, step=0.0001, format="%.4f")
            ref_rt_window = (float(ref_rt_min), float(ref_rt_max))

        metadata_file = st.file_uploader("Optional metadata CSV", type=["csv"])
        metadata_path = None
        do_blank_filter = False
        do_batch_correct = False
        if metadata_file is not None:
            metadata_path = save_uploaded_binary(metadata_file, paths["metadata"])
            st.session_state["last_metadata_file"] = str(metadata_path)
            qc_col1, qc_col2 = st.columns(2)
            with qc_col1:
                do_blank_filter = st.checkbox("Apply blank filter", value=True)
            with qc_col2:
                do_batch_correct = st.checkbox("Apply batch correction", value=True)
        elif st.session_state.get("last_metadata_file"):
            prev = Path(st.session_state["last_metadata_file"])
            if prev.exists():
                metadata_path = prev
                st.info(f"Using previously uploaded metadata file: {prev.name}")

        ms1_points_path = None
        if not extract_ms1:
            ms1_points_file = st.file_uploader("Optional MS1 points CSV", type=["csv"])
            if ms1_points_file is not None:
                ms1_points_path = save_uploaded_binary(ms1_points_file, paths["ms1_points_uploads"])
                st.session_state["last_ms1_points_file"] = str(ms1_points_path)
            elif st.session_state.get("last_ms1_points_file"):
                prev = Path(st.session_state["last_ms1_points_file"])
                if prev.exists():
                    ms1_points_path = prev
                    st.info(f"Using previously uploaded MS1 points file: {prev.name}")

    col_run, col_clear_state, col_clear_uploads, col_clear_runs = st.columns([3, 1, 1, 1])
    with col_run:
        run_clicked = st.button("Run analysis", type="primary")
    with col_clear_state:
        if st.button("Clear session state"):
            clear_session_state_only()
            st.rerun()
    with col_clear_uploads:
        if st.button("Clear saved uploads"):
            clear_folder_contents(paths["uploads"])
            clear_folder_contents(paths["metadata"])
            clear_folder_contents(paths["ms1_points_uploads"])
            st.session_state.pop("last_saved_files", None)
            st.session_state.pop("last_metadata_file", None)
            st.session_state.pop("last_ms1_points_file", None)
            st.rerun()
    with col_clear_runs:
        if st.button("Clear saved runs"):
            clear_folder_contents(paths["runs"])
            clear_folder_contents(paths["downloads"])
            clear_folder_contents(paths["logs"])
            reset_download_state()
            st.rerun()

    if run_clicked:
        reset_download_state()

        if not saved_files:
            st.session_state["download_status"] = "error"
            st.session_state["download_status_detail"] = "Please upload at least one mzML file before running the pipeline."
        elif rt_max < rt_min:
            st.session_state["download_status"] = "error"
            st.session_state["download_status_detail"] = "RT max must be greater than or equal to RT min."
        elif use_reference and ref_rt_window is not None and ref_rt_window[1] < ref_rt_window[0]:
            st.session_state["download_status"] = "error"
            st.session_state["download_status_detail"] = "Reference RT max must be greater than or equal to Reference RT min."
        else:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_root = paths["runs"] / class_tag / stamp
            output_root.mkdir(parents=True, exist_ok=True)

            log_capture = io.StringIO()
            st.session_state["download_status"] = "running"
            st.session_state["download_status_detail"] = "Pipeline is running."

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
                    raise RuntimeError("The pipeline finished but no output files were found.")

                previews = build_previews(output_files, class_tag)

                zip_path = paths["downloads"] / f"CPM_{class_tag}_{stamp}.zip"
                build_zip_on_disk(output_files, paths["root"], zip_path)

                st.session_state["zip_path"] = str(zip_path)
                st.session_state["zip_name"] = zip_path.name
                st.session_state["inline_log"] = safe_text(log_capture.getvalue())
                st.session_state["previews"] = previews
                st.session_state["discovered_output_files"] = [str(p) for p in output_files]
                st.session_state["run_summary"] = {
                    "class_tag": class_tag,
                    "file_count": len(saved_files),
                    "zip_size_mb": zip_path.stat().st_size / (1024 * 1024),
                    "output_count": len(output_files),
                    "root_folder": str(output_root),
                    "save_root": str(paths["root"]),
                }
                st.session_state["download_status"] = "ready"
                st.session_state["download_status_detail"] = "Run complete. Outputs and ZIP were saved to your chosen folder."

            except Exception as exc:
                st.session_state["inline_log"] = safe_text(log_capture.getvalue())
                st.session_state["download_status"] = "error"
                st.session_state["download_status_detail"] = safe_text(exc)

    status = st.session_state.get("download_status", "idle")
    status_detail = st.session_state.get("download_status_detail", "")
    previews = st.session_state.get("previews", {})
    run_summary = st.session_state.get("run_summary")
    inline_log = st.session_state.get("inline_log", "")
    zip_path = st.session_state.get("zip_path")
    zip_name = st.session_state.get("zip_name")
    discovered_output_files = [Path(p) for p in st.session_state.get("discovered_output_files", []) if Path(p).exists()]

    if status == "running":
        st.info(status_detail)
    elif status == "error":
        st.error("Run failed or packaging failed.")
        st.write(status_detail)
    elif status == "ready" and run_summary:
        st.success("Run complete.")
        st.write(
            f"Class: {run_summary['class_tag']} | Inputs: {run_summary['file_count']} | "
            f"Packaged files: {run_summary['output_count']} | ZIP size: {run_summary['zip_size_mb']:.1f} MB"
        )
        st.write(f"Saved under: {run_summary['save_root']}")
    else:
        st.info(status_detail)

    if status in {"ready", "downloaded"} and previews:
        st.subheader("Run outputs")

        for key, title in [
            ("cyano_heatmap_png", "Specific cyanopeptide-class detection intensity heatmap"),
            ("rt_plot_png", "Precursor RT plot"),
            ("dot_plot_png", "Individual ion dot plot"),
            ("diagnostic_individual_png", "Diagnostic ion distribution – individual"),
            ("matched_tiles_png", "Matched compound tiles (Putative annotations)"),
            ("adduct_graph_png", "Adduct graph (merged)"),
        ]:
            data = previews.get(key)
            if data:
                st.subheader(title)
                st.image(data)

        ind_hits = previews.get("ind_hits_table")
        if ind_hits:
            st.subheader("Individual hits (labeled) – preview")
            st.dataframe(pd.DataFrame(ind_hits["rows"], columns=ind_hits["columns"]))

        indiv_merged = previews.get("indiv_merged_table")
        if indiv_merged:
            st.subheader("Individual merged summary")
            st.dataframe(pd.DataFrame(indiv_merged["rows"], columns=indiv_merged["columns"]))

        unknown_table = previews.get("unknown_features_table")
        if unknown_table and unknown_table["rows"]:
            st.subheader("Unknown features with scans (table)")
            st.dataframe(pd.DataFrame(unknown_table["rows"], columns=unknown_table["columns"]))

        if discovered_output_files:
            st.divider()
            render_all_discovered_outputs(discovered_output_files)

        st.divider()
        st.subheader("Download Results")
        if zip_path and zip_name and Path(zip_path).exists():
            with open(zip_path, "rb") as fh:
                downloaded = st.download_button(
                    "Download All Outputs (.zip)",
                    data=fh,
                    file_name=zip_name,
                    mime="application/zip",
                    type="primary",
                    use_container_width=True,
                )
            st.caption("MS1 points are included in the ZIP. The ZIP is also saved to your chosen output folder.")
            if downloaded:
                consume_download()
                st.rerun()
            st.code(str(zip_path))
        else:
            st.info("ZIP file not currently available in session.")

        if inline_log:
            with st.expander("Run log", expanded=False):
                st.text(inline_log)


page = st.sidebar.radio("Navigation", ["About this app", "Run pipeline"], index=1)

if page == "About this app":
    render_home_page()
else:
    render_run_page()
