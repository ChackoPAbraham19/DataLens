import re
import uuid
import shutil
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, field_validator

from services.profiler import run_profile
from utils.file_handler import cleanup_temp_files, TEMP_DIR
from utils.kaggle_client import (
    download_kaggle_dataset,
    find_csvs_in_dir,
    KaggleAuthError,
    KaggleDatasetNotFoundError,
    KaggleDownloadError,
)

router = APIRouter(prefix="/kaggle", tags=["kaggle"])


# ---------------------------------------------------------------------------
# Request schema
# ---------------------------------------------------------------------------

class KaggleRequest(BaseModel):
    """
    Accepts either:
      - A full Kaggle URL:   https://www.kaggle.com/datasets/owner/dataset-name
      - A short slug:        owner/dataset-name
    Optionally pin a specific CSV if the dataset has multiple files.
    """
    source: str
    target_csv: Optional[str] = None

    @field_validator("source")
    @classmethod
    def parse_and_normalize(cls, v: str) -> str:
        v = v.strip()

        # Full URL form
        url_pattern = re.compile(
            r"https?://(?:www\.)?kaggle\.com/datasets/([^/]+/[^/?#]+)"
        )
        match = url_pattern.match(v)
        if match:
            return match.group(1)   # normalize to "owner/dataset"

        # Short slug form: owner/dataset
        slug_pattern = re.compile(r"^[\w-]+/[\w-]+$")
        if slug_pattern.match(v):
            return v

        raise ValueError(
            "Invalid Kaggle source. Provide a full URL "
            "(https://www.kaggle.com/datasets/owner/dataset) "
            "or a slug (owner/dataset-name)."
        )


# ---------------------------------------------------------------------------
# Route
# ---------------------------------------------------------------------------

@router.post("/")
async def ingest_kaggle_dataset(
    payload: KaggleRequest,
    background_tasks: BackgroundTasks,
):
    """
    Download a Kaggle dataset and profile it.

    - **source**: Kaggle URL or slug (e.g. `zillow/zecon` or full URL)
    - **target_csv**: (optional) specific CSV filename if dataset has multiple

    Returns a job_id to poll at GET /profile/{job_id}

    Requires KAGGLE_USERNAME and KAGGLE_KEY in .env
    """

    dataset_slug = payload.source   # already normalized by validator
    job_id = str(uuid.uuid4())
    job_dir = TEMP_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    try:
        # --- Download from Kaggle ---
        download_dir = job_dir / "kaggle_download"
        download_dir.mkdir(exist_ok=True)

        await download_kaggle_dataset(
            dataset_slug=dataset_slug,
            download_dir=download_dir,
        )

        # --- Find CSVs ---
        csv_files = find_csvs_in_dir(download_dir)

        if not csv_files:
            raise HTTPException(
                status_code=422,
                detail=f"No CSV files found in Kaggle dataset '{dataset_slug}'."
            )

        csv_path: Optional[Path] = None

        if len(csv_files) == 1:
            csv_path = csv_files[0]

        elif len(csv_files) > 1:
            if payload.target_csv:
                matches = [f for f in csv_files if f.name == payload.target_csv]
                if not matches:
                    available = [f.name for f in csv_files]
                    raise HTTPException(
                        status_code=422,
                        detail={
                            "message": f"'{payload.target_csv}' not found in dataset.",
                            "available_csvs": available,
                        }
                    )
                csv_path = matches[0]
            else:
                # Return available files for user to choose
                available = [f.name for f in csv_files]
                shutil.rmtree(job_dir, ignore_errors=True)
                return JSONResponse(
                    status_code=300,
                    content={
                        "detail": (
                            "Multiple CSVs found in this Kaggle dataset. "
                            "Re-submit with target_csv set to one of the options below."
                        ),
                        "dataset": dataset_slug,
                        "available_csvs": available,
                    }
                )

    except KaggleAuthError:
        shutil.rmtree(job_dir, ignore_errors=True)
        raise HTTPException(
            status_code=401,
            detail=(
                "Kaggle authentication failed. "
                "Set KAGGLE_USERNAME and KAGGLE_KEY in your .env file. "
                "Get your API token from https://www.kaggle.com/settings/account"
            )
        )

    except KaggleDatasetNotFoundError:
        shutil.rmtree(job_dir, ignore_errors=True)
        raise HTTPException(
            status_code=404,
            detail=f"Kaggle dataset '{dataset_slug}' not found or is private."
        )

    except KaggleDownloadError as e:
        shutil.rmtree(job_dir, ignore_errors=True)
        raise HTTPException(
            status_code=502,
            detail=f"Kaggle download failed: {str(e)}"
        )

    except HTTPException:
        shutil.rmtree(job_dir, ignore_errors=True)
        raise

    except Exception as e:
        shutil.rmtree(job_dir, ignore_errors=True)
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error during Kaggle ingestion: {str(e)}"
        )

    # --- Kick off background profiling ---
    background_tasks.add_task(
        run_profile,
        job_id=job_id,
        csv_path=csv_path,
        metadata={"source": "kaggle", "dataset": dataset_slug},
    )

    # --- Schedule cleanup after 1 hour ---
    background_tasks.add_task(cleanup_temp_files, job_dir=job_dir, delay_seconds=3600)

    return JSONResponse(
        status_code=202,
        content={
            "job_id": job_id,
            "status": "queued",
            "dataset": dataset_slug,
            "file": csv_path.name,
            "message": "Profiling started. Poll GET /profile/{job_id} for results.",
        }
    )