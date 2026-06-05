import json
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

from utils.file_handler import TEMP_DIR

router = APIRouter(prefix="/profile", tags=["profile"])

# ---------------------------------------------------------------------------
# Job status files written by services/profiler.py
#
# Job directory layout (TEMP_DIR / job_id /):
#   status.json       — always present, written first
#   result.json       — written on success
#   error.json        — written on failure
#
# status.json shape:
#   { "status": "queued" | "running" | "done" | "failed",
#     "progress": 0-100,
#     "step": "Loading CSV" | "Running analysis" | ... }
# ---------------------------------------------------------------------------

JOB_STATES = {"queued", "running", "done", "failed"}


def _job_dir(job_id: str) -> Path:
    return TEMP_DIR / job_id


def _read_json(path: Path) -> Optional[dict]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def _assert_job_exists(job_id: str) -> Path:
    """Validate job_id format and existence. Returns job_dir."""
    # Basic UUID format guard — prevents path traversal
    import re
    if not re.fullmatch(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", job_id):
        raise HTTPException(status_code=400, detail="Invalid job_id format.")

    job_dir = _job_dir(job_id)
    if not job_dir.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Job '{job_id}' not found. It may have expired or never existed."
        )
    return job_dir


# ---------------------------------------------------------------------------
# GET /profile/{job_id}  — main polling endpoint
# ---------------------------------------------------------------------------

@router.get("/{job_id}")
async def get_profile_status(job_id: str):
    """
    Poll the status of a profiling job.

    **Lifecycle:**
    - `queued`  → job is waiting to start
    - `running` → profiling in progress (check `progress` + `step`)
    - `done`    → full results available in `result` field
    - `failed`  → see `error` field for reason

    Poll every 2-3 seconds until status is `done` or `failed`.
    Results are retained for 1 hour after completion.
    """
    job_dir = _assert_job_exists(job_id)

    status_data = _read_json(job_dir / "status.json")

    if not status_data:
        # Directory exists but status file not yet written — race condition window
        return JSONResponse(
            status_code=202,
            content={
                "job_id": job_id,
                "status": "queued",
                "progress": 0,
                "step": "Waiting to start...",
            }
        )

    current_status = status_data.get("status", "queued")

    # --- Still in progress ---
    if current_status in ("queued", "running"):
        return JSONResponse(
            status_code=202,
            content={
                "job_id": job_id,
                "status": current_status,
                "progress": status_data.get("progress", 0),
                "step": status_data.get("step", ""),
            }
        )

    # --- Failed ---
    if current_status == "failed":
        error_data = _read_json(job_dir / "error.json") or {}
        return JSONResponse(
            status_code=200,
            content={
                "job_id": job_id,
                "status": "failed",
                "error": error_data.get("message", "Unknown error occurred."),
                "detail": error_data.get("detail", None),
            }
        )

    # --- Done ---
    if current_status == "done":
        result_data = _read_json(job_dir / "result.json")
        if not result_data:
            raise HTTPException(
                status_code=500,
                detail="Job marked done but result file is missing. This is a server error."
            )
        return JSONResponse(
            status_code=200,
            content={
                "job_id": job_id,
                "status": "done",
                "result": result_data,
            }
        )

    # --- Unknown state ---
    raise HTTPException(
        status_code=500,
        detail=f"Job is in an unknown state: '{current_status}'."
    )


# ---------------------------------------------------------------------------
# GET /profile/{job_id}/summary  — lightweight status-only check
# ---------------------------------------------------------------------------

@router.get("/{job_id}/summary")
async def get_profile_summary(job_id: str):
    """
    Lightweight endpoint — returns only job status + top-level stats.
    Does NOT return full chart data. Useful for dashboards or listings.
    """
    job_dir = _assert_job_exists(job_id)
    status_data = _read_json(job_dir / "status.json") or {}
    current_status = status_data.get("status", "queued")

    response: dict[str, Any] = {
        "job_id": job_id,
        "status": current_status,
        "progress": status_data.get("progress", 0),
        "step": status_data.get("step", ""),
    }

    if current_status == "done":
        result_data = _read_json(job_dir / "result.json") or {}
        overview = result_data.get("overview", {})
        response["overview"] = {
            "filename":        overview.get("filename"),
            "source":          overview.get("source", "upload"),
            "rows":            overview.get("rows"),
            "columns":         overview.get("columns"),
            "total_cells":     overview.get("total_cells"),
            "missing_cells":   overview.get("missing_cells"),
            "missing_pct":     overview.get("missing_pct"),
            "duplicate_rows":  overview.get("duplicate_rows"),
            "memory_usage_mb": overview.get("memory_usage_mb"),
            "profiled_at":     overview.get("profiled_at"),
        }

    return JSONResponse(status_code=200, content=response)


# ---------------------------------------------------------------------------
# DELETE /profile/{job_id}  — manual cleanup
# ---------------------------------------------------------------------------

@router.delete("/{job_id}")
async def delete_profile_job(job_id: str):
    """
    Manually delete a profiling job and its associated files.
    Jobs auto-expire after 1 hour regardless.
    """
    import shutil

    job_dir = _assert_job_exists(job_id)

    try:
        shutil.rmtree(job_dir)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete job files: {str(e)}"
        )

    return JSONResponse(
        status_code=200,
        content={
            "job_id": job_id,
            "deleted": True,
            "message": "Job and all associated files have been removed."
        }
    )