import shutil
import threading
from pathlib import Path

TEMP_DIR = Path("/tmp/datalens_jobs")
TEMP_DIR.mkdir(parents=True, exist_ok=True)


def save_temp_file(contents: bytes, filename: str, job_dir: Path) -> Path:
    """Write raw bytes to job_dir / filename. Returns the file path."""
    dest = job_dir / filename
    dest.write_bytes(contents)
    return dest


def schedule_cleanup(job_dir: Path, delay_seconds: int = 3600) -> None:
    """
    Fire-and-forget cleanup: spawns a daemon thread that sleeps then deletes.

    This must NOT be added as a FastAPI BackgroundTask, because TestClient
    (and any HTTP client) awaits all background tasks before returning the
    response — a 1-hour sleep would make every request hang.

    Using a daemon thread means:
      - The response returns immediately.
      - The cleanup still happens after the delay.
      - Daemon threads are killed when the process exits, which is fine
        for temp-file cleanup (OS can reap /tmp on its own).
    """
    import time

    def _worker():
        time.sleep(delay_seconds)
        shutil.rmtree(job_dir, ignore_errors=True)

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
