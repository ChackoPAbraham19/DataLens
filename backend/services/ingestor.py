import zipfile
from pathlib import Path
from typing import Optional

import chardet
import pandas as pd


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class IngestorError(Exception):
    """Base class for ingestor errors."""

class EmptyDatasetError(IngestorError):
    """Raised when the loaded DataFrame has no rows or no columns."""

class EncodingDetectionError(IngestorError):
    """Raised when file encoding cannot be determined."""

class UnsupportedFileError(IngestorError):
    """Raised for unsupported file types."""


# ---------------------------------------------------------------------------
# Encoding detection
# ---------------------------------------------------------------------------

def _detect_encoding(path: Path, sample_bytes: int = 100_000) -> str:
    """
    Detect file encoding by reading a sample.
    Falls back to utf-8 if detection confidence is too low.
    """
    with open(path, "rb") as f:
        raw = f.read(sample_bytes)

    result = chardet.detect(raw)
    encoding = result.get("encoding")
    confidence = result.get("confidence", 0)

    if not encoding or confidence < 0.6:
        return "utf-8"

    # Normalize common aliases
    encoding = encoding.lower()
    if encoding in ("ascii", "utf-8-sig"):
        return "utf-8"

    return encoding


# ---------------------------------------------------------------------------
# CSV loader
# ---------------------------------------------------------------------------

def _load_csv(path: Path) -> pd.DataFrame:
    """
    Load a CSV into a DataFrame.
    Tries multiple separators and encodings before giving up.
    """
    encoding = _detect_encoding(path)

    separators = [",", ";", "\t", "|"]

    for sep in separators:
        try:
            df = pd.read_csv(
                path,
                sep=sep,
                encoding=encoding,
                low_memory=False,
                on_bad_lines="warn",
            )
            # A single-column result usually means wrong separator
            if df.shape[1] > 1:
                return df
        except UnicodeDecodeError:
            # Try latin-1 as last resort
            try:
                df = pd.read_csv(
                    path,
                    sep=sep,
                    encoding="latin-1",
                    low_memory=False,
                    on_bad_lines="warn",
                )
                if df.shape[1] > 1:
                    return df
            except Exception:
                continue
        except Exception:
            continue

    # Last attempt: comma sep, utf-8, accept single column
    try:
        return pd.read_csv(path, encoding=encoding, low_memory=False, on_bad_lines="warn")
    except Exception as e:
        raise IngestorError(f"Failed to parse CSV '{path.name}': {e}")


# ---------------------------------------------------------------------------
# ZIP extractor
# ---------------------------------------------------------------------------

def _extract_csv_from_zip(zip_path: Path, extract_dir: Path) -> list[Path]:
    """Extract all CSVs from a zip. Returns sorted list by file size desc."""
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_dir)
    except zipfile.BadZipFile:
        raise IngestorError(f"'{zip_path.name}' is not a valid ZIP file.")

    csvs = list(extract_dir.rglob("*.csv"))
    if not csvs:
        raise IngestorError(f"No CSV files found inside '{zip_path.name}'.")

    # Sort largest first — most likely the main dataset
    return sorted(csvs, key=lambda p: p.stat().st_size, reverse=True)


# ---------------------------------------------------------------------------
# Post-load validation and cleanup
# ---------------------------------------------------------------------------

def _validate_and_clean(df: pd.DataFrame, filename: str) -> pd.DataFrame:
    """
    Basic validation and light cleanup on the raw DataFrame.
    - Strips whitespace from column names
    - Drops fully empty rows/columns
    - Validates non-empty result
    """
    if df.empty or df.shape[1] == 0:
        raise EmptyDatasetError(f"'{filename}' loaded as an empty DataFrame.")

    # Strip whitespace from column names
    df.columns = df.columns.str.strip()

    # Drop columns that are entirely unnamed (pandas artifact)
    unnamed_cols = [c for c in df.columns if str(c).startswith("Unnamed:")]
    if unnamed_cols:
        df = df.drop(columns=unnamed_cols)

    # Drop rows that are 100% NaN
    df = df.dropna(how="all")

    if df.shape[0] == 0:
        raise EmptyDatasetError(f"'{filename}' has no data rows after cleanup.")

    if df.shape[1] == 0:
        raise EmptyDatasetError(f"'{filename}' has no valid columns after cleanup.")

    return df


# ---------------------------------------------------------------------------
# Type inference refinement
# ---------------------------------------------------------------------------

def _refine_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Attempt smarter type inference on object columns.
    - Try numeric conversion
    - Try datetime conversion
    Leaves column as-is if conversion fails.
    """
    for col in df.select_dtypes(include=["object"]).columns:
        # Try numeric
        converted = pd.to_numeric(df[col], errors="coerce")
        non_null_original = df[col].notna().sum()
        non_null_converted = converted.notna().sum()

        # Accept numeric conversion if >80% of non-null values convert cleanly
        if non_null_original > 0 and (non_null_converted / non_null_original) >= 0.80:
            df[col] = converted
            continue

        # Try datetime
        try:
            converted_dt = pd.to_datetime(df[col], errors="coerce")
            non_null_dt = converted_dt.notna().sum()
            if non_null_original > 0 and (non_null_dt / non_null_original) >= 0.80:
                df[col] = converted_dt
        except Exception:
            pass

    return df


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def ingest(
    path: Path,
    target_csv: Optional[str] = None,
    refine_types: bool = True,
) -> tuple[pd.DataFrame, str]:
    """
    Load a CSV or ZIP file into a cleaned pandas DataFrame.

    Args:
        path:         Path to .csv or .zip file
        target_csv:   If ZIP has multiple CSVs, pick this one by filename
        refine_types: Whether to run dtype refinement (default True)

    Returns:
        (df, filename) tuple

    Raises:
        IngestorError, EmptyDatasetError, UnsupportedFileError
    """
    suffix = path.suffix.lower()

    if suffix == ".csv":
        df = _load_csv(path)
        filename = path.name

    elif suffix == ".zip":
        extract_dir = path.parent / f"{path.stem}_extracted"
        extract_dir.mkdir(exist_ok=True)

        csv_files = _extract_csv_from_zip(path, extract_dir)

        if target_csv:
            matches = [f for f in csv_files if f.name == target_csv]
            if not matches:
                available = [f.name for f in csv_files]
                raise IngestorError(
                    f"'{target_csv}' not found in ZIP. Available: {available}"
                )
            chosen = matches[0]
        else:
            chosen = csv_files[0]   # largest CSV

        df = _load_csv(chosen)
        filename = chosen.name

    else:
        raise UnsupportedFileError(
            f"Cannot ingest '{path.suffix}' files. Only .csv and .zip are supported."
        )

    df = _validate_and_clean(df, filename)

    if refine_types:
        df = _refine_dtypes(df)

    return df, filename


def ingest_from_dataframe(df: pd.DataFrame, filename: str = "dataframe") -> tuple[pd.DataFrame, str]:
    """
    Accept an already-loaded DataFrame (e.g. from Kaggle client).
    Still runs validation + type refinement.
    """
    df = _validate_and_clean(df, filename)
    df = _refine_dtypes(df)
    return df, filename