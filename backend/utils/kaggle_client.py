import asyncio
import os
from pathlib import Path


# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------

class KaggleAuthError(Exception):
    pass

class KaggleDatasetNotFoundError(Exception):
    pass

class KaggleDownloadError(Exception):
    pass


# ---------------------------------------------------------------------------
# Internal: authenticate once
# ---------------------------------------------------------------------------

def _get_api():
    from kaggle.api.kaggle_api_extended import KaggleApiExtended  # deferred

    username = os.getenv("KAGGLE_USERNAME")
    key = os.getenv("KAGGLE_KEY")

    if not username or not key:
        raise KaggleAuthError("KAGGLE_USERNAME and KAGGLE_KEY must be set in .env")

    os.environ["KAGGLE_USERNAME"] = username
    os.environ["KAGGLE_KEY"] = key

    api = KaggleApiExtended()
    try:
        api.authenticate()
    except Exception as e:
        raise KaggleAuthError(f"Kaggle authentication failed: {e}")

    return api


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def download_kaggle_dataset(dataset_slug: str, download_dir: Path) -> None:
    """
    Download a Kaggle dataset into download_dir.
    Runs the blocking kaggle API call in a thread executor.

    Args:
        dataset_slug: "owner/dataset-name"
        download_dir: directory to download into (will be created if missing)
    """
    download_dir.mkdir(parents=True, exist_ok=True)

    def _download():
        from kaggle.rest import ApiException  # deferred

        api = _get_api()
        try:
            api.dataset_download_files(
                dataset=dataset_slug,
                path=str(download_dir),
                unzip=True,
                quiet=True,
            )
        except ApiException as e:
            if e.status == 403 or e.status == 401:
                raise KaggleAuthError(f"Access denied for dataset '{dataset_slug}'.")
            if e.status == 404:
                raise KaggleDatasetNotFoundError(f"Dataset '{dataset_slug}' not found.")
            raise KaggleDownloadError(f"Kaggle API error {e.status}: {e.reason}")
        except KaggleAuthError:
            raise
        except KaggleDatasetNotFoundError:
            raise
        except Exception as e:
            raise KaggleDownloadError(f"Unexpected download error: {e}")

    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, _download)


def find_csvs_in_dir(directory: Path) -> list[Path]:
    """Recursively find all CSVs in a directory, sorted by size descending."""
    csvs = list(directory.rglob("*.csv"))
    return sorted(csvs, key=lambda p: p.stat().st_size, reverse=True)