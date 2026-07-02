from typing import Any, Optional
from pydantic import BaseModel


class JobResponse(BaseModel):
    job_id: str
    status: str
    message: Optional[str] = None


class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    progress: int
    step: str


class JobResultResponse(BaseModel):
    job_id: str
    status: str
    result: Optional[dict[str, Any]] = None
    error: Optional[str] = None


class KaggleRequest(BaseModel):
    source: str
    target_csv: Optional[str] = None