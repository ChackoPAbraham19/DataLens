import os
import uuid
import zipfile
import shutil
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse

from services.profiler import run_profile
from utils.file_handler import save_temp_file, cleanup_temp_files, TEMP_DIR

router = APIRouter(prefix="/upload", tags=["upload"])

ALLOWED_EXTENSIONS = {".csv", ".zip"}
MAX_FILE_SIZE_MB = 200
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024


def validate_extension(filename: str) -> str:
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Only .csv and .zip are allowed."
        )
    return ext


def extract_csv_from_zip(zip_path: Path, extract_dir: Path) -> list[Path]:
    """Extract all CSVs from a zip archive. Returns list of CSV paths."""
    csv_files = []
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            # Security check: prevent zip slip attacks
            for member in zf.namelist():
                member_path = (extract_dir / member).resolve()
                if not str(member_path).startswith(str(extract_dir.resolve())):
                    raise HTTPException(
                        status_code=400,
                        detail="Invalid zip file: contains unsafe paths."
                    )

            zf.extractall(extract_dir)

            # Find all CSVs recursively
            csv_files = list(extract_dir.rglob("*.csv"))

    except zipfile.BadZipFile:
        raise HTTPException(status_code=400, detail="Uploaded ZIP file is corrupted or invalid.")

    if not csv_files:
        raise HTTPException(
            status_code=422,
            detail="No CSV files found inside the ZIP archive."
        )

    return csv_files


@router.post("/")
async def upload_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    target_csv: Optional[str] = None,   # if ZIP has multiple CSVs, user can specify one
):
    """
    Upload a CSV or ZIP file for dataset profiling.

    - **file**: .csv or .zip file (max 200MB)
    - **target_csv**: (optional) filename to use if ZIP contains multiple CSVs

    Returns a job_id to poll at GET /profile/{job_id}
    """

    # --- Validate filename ---
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided.")

    ext = validate_extension(file.filename)

    # --- Read file content with size check ---
    contents = await file.read()
    if len(contents) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum allowed size is {MAX_FILE_SIZE_MB}MB."
        )

    # --- Generate job ID and working directory ---
    job_id = str(uuid.uuid4())
    job_dir = TEMP_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    csv_path: Optional[Path] = None

    try:
        if ext == ".csv":
            # Save directly
            csv_path = save_temp_file(contents, file.filename, job_dir)

        elif ext == ".zip":
            # Save zip, then extract
            zip_path = save_temp_file(contents, file.filename, job_dir)
            extract_dir = job_dir / "extracted"
            extract_dir.mkdir(exist_ok=True)

            csv_files = extract_csv_from_zip(zip_path, extract_dir)

            if len(csv_files) == 1:
                csv_path = csv_files[0]

            elif len(csv_files) > 1:
                # If user specified a target, find it
                if target_csv:
                    matches = [f for f in csv_files if f.name == target_csv]
                    if not matches:
                        available = [f.name for f in csv_files]
                        raise HTTPException(
                            status_code=422,
                            detail={
                                "message": f"'{target_csv}' not found in ZIP.",
                                "available_csvs": available
                            }
                        )
                    csv_path = matches[0]
                else:
                    # Return list and ask user to pick
                    available = [f.name for f in csv_files]
                    # Clean up and return selection prompt
                    shutil.rmtree(job_dir, ignore_errors=True)
                    return JSONResponse(
                        status_code=300,
                        content={
                            "detail": "Multiple CSVs found in ZIP. Re-upload with target_csv parameter.",
                            "available_csvs": available
                        }
                    )

    except HTTPException:
        shutil.rmtree(job_dir, ignore_errors=True)
        raise
    except Exception as e:
        shutil.rmtree(job_dir, ignore_errors=True)
        raise HTTPException(status_code=500, detail=f"File processing error: {str(e)}")

    # --- Kick off background profiling job ---
    background_tasks.add_task(run_profile, job_id=job_id, csv_path=csv_path)

    # --- Schedule cleanup after 1 hour ---
    background_tasks.add_task(cleanup_temp_files, job_dir=job_dir, delay_seconds=3600)

    return JSONResponse(
        status_code=202,
        content={
            "job_id": job_id,
            "status": "queued",
            "filename": file.filename,
            "size_bytes": len(contents),
            "message": "Profiling started. Poll GET /profile/{job_id} for results."
        }
    )