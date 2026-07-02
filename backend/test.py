import io
import json
import zipfile
import time
from unittest.mock import patch
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from main import app

client = TestClient(app)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_csv_bytes(content: str = None) -> bytes:
    if content is None:
        content = "name,age,score\nAlice,30,88.5\nBob,25,92.0\nCharlie,,75.0\nAlice,30,88.5\n"
    return content.encode("utf-8")


def make_zip_bytes(files: dict) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, content in files.items():
            zf.writestr(name, content)
    return buf.getvalue()


def poll_until_done(job_id: str, timeout: int = 30) -> dict:
    deadline = time.time() + timeout
    dots = 0
    while time.time() < deadline:
        res = client.get(f"/profile/{job_id}")
        data = res.json()
        status = data["status"]
        progress = data.get("progress", "?")
        step = data.get("step", "")
        print(f"\r  polling [{job_id[:8]}] status={status} progress={progress}% step={step!r}    ", end="", flush=True)
        if status in ("done", "failed"):
            print()
            return data
        time.sleep(0.3)
    print()
    pytest.fail(f"Job {job_id} did not complete within {timeout}s")


def upload_csv(content=None, filename="test.csv"):
    return client.post(
        "/upload/",
        files={"file": (filename, make_csv_bytes(content), "text/csv")},
    )


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

def test_health(capsys):
    with capsys.disabled():
        print("\n[health] GET /health")
    res = client.get("/health")
    with capsys.disabled():
        print(f"  → {res.status_code} {res.json()}")
    assert res.status_code == 200
    assert res.json()["status"] == "ok"


# ---------------------------------------------------------------------------
# POST /upload/
# ---------------------------------------------------------------------------

class TestUpload:

    def test_upload_csv_returns_202(self, capsys):
        with capsys.disabled():
            print("\n[upload] POST /upload/ with valid CSV")
        res = upload_csv()
        with capsys.disabled():
            print(f"  → {res.status_code} job_id={res.json().get('job_id','N/A')[:8]}...")
        assert res.status_code == 202
        data = res.json()
        assert "job_id" in data
        assert data["status"] == "queued"
        assert data["filename"] == "test.csv"

    def test_upload_invalid_extension_returns_400(self, capsys):
        with capsys.disabled():
            print("\n[upload] POST /upload/ with .xlsx (invalid)")
        res = client.post(
            "/upload/",
            files={"file": ("data.xlsx", b"garbage", "application/octet-stream")},
        )
        with capsys.disabled():
            print(f"  → {res.status_code} detail={res.json().get('detail','')[:60]}")
        assert res.status_code == 400
        assert "Unsupported file type" in res.json()["detail"]

    def test_upload_no_file_returns_422(self, capsys):
        with capsys.disabled():
            print("\n[upload] POST /upload/ with no file")
        res = client.post("/upload/")
        with capsys.disabled():
            print(f"  → {res.status_code}")
        assert res.status_code == 422

    def test_upload_zip_single_csv_returns_202(self, capsys):
        with capsys.disabled():
            print("\n[upload] POST /upload/ with ZIP (1 CSV inside)")
        csv = "col1,col2\n1,2\n3,4\n"
        zipped = make_zip_bytes({"data.csv": csv})
        res = client.post("/upload/", files={"file": ("archive.zip", zipped, "application/zip")})
        with capsys.disabled():
            print(f"  → {res.status_code} status={res.json().get('status')}")
        assert res.status_code == 202
        assert res.json()["status"] == "queued"

    def test_upload_zip_multiple_csvs_returns_300(self, capsys):
        with capsys.disabled():
            print("\n[upload] POST /upload/ with ZIP (2 CSVs — expect 300)")
        zipped = make_zip_bytes({"train.csv": "a,b\n1,2\n", "test.csv": "a,b\n3,4\n"})
        res = client.post("/upload/", files={"file": ("archive.zip", zipped, "application/zip")})
        with capsys.disabled():
            data = res.json()
            print(f"  → {res.status_code} available_csvs={data.get('available_csvs')}")
        assert res.status_code == 300
        assert "available_csvs" in res.json()
        assert len(res.json()["available_csvs"]) == 2

    def test_upload_zip_with_target_csv(self, capsys):
        with capsys.disabled():
            print("\n[upload] POST /upload/?target_csv=train.csv with ZIP")
        zipped = make_zip_bytes({"train.csv": "a,b\n1,2\n3,4\n", "test.csv": "a,b\n5,6\n"})
        res = client.post("/upload/?target_csv=train.csv", files={"file": ("archive.zip", zipped, "application/zip")})
        with capsys.disabled():
            print(f"  → {res.status_code}")
        assert res.status_code == 202

    def test_upload_zip_bad_target_csv_returns_422(self, capsys):
        with capsys.disabled():
            print("\n[upload] POST /upload/?target_csv=nonexistent.csv (expect 422)")
        zipped = make_zip_bytes({"train.csv": "a,b\n1,2\n", "test.csv": "a,b\n3,4\n"})
        res = client.post("/upload/?target_csv=nonexistent.csv", files={"file": ("archive.zip", zipped, "application/zip")})
        with capsys.disabled():
            print(f"  → {res.status_code}")
        assert res.status_code == 422

    def test_upload_corrupt_zip_returns_400(self, capsys):
        with capsys.disabled():
            print("\n[upload] POST /upload/ with corrupt ZIP")
        res = client.post("/upload/", files={"file": ("bad.zip", b"not a zip", "application/zip")})
        with capsys.disabled():
            print(f"  → {res.status_code}")
        assert res.status_code == 400

    def test_upload_empty_zip_returns_422(self, capsys):
        with capsys.disabled():
            print("\n[upload] POST /upload/ with ZIP containing no CSVs")
        zipped = make_zip_bytes({})
        res = client.post("/upload/", files={"file": ("empty.zip", zipped, "application/zip")})
        with capsys.disabled():
            print(f"  → {res.status_code} detail={res.json().get('detail','')[:60]}")
        assert res.status_code == 422
        assert "No CSV" in res.json()["detail"]


# ---------------------------------------------------------------------------
# GET /profile/{job_id}
# ---------------------------------------------------------------------------

class TestProfilePolling:

    def test_invalid_job_id_format_returns_400(self, capsys):
        with capsys.disabled():
            print("\n[profile] GET /profile/not-a-uuid (expect 400)")
        res = client.get("/profile/not-a-uuid")
        with capsys.disabled():
            print(f"  → {res.status_code} detail={res.json().get('detail','')[:60]}")
        assert res.status_code == 400
        assert "Invalid job_id" in res.json()["detail"]

    def test_nonexistent_job_returns_404(self, capsys):
        with capsys.disabled():
            print("\n[profile] GET /profile/00000000-... (expect 404)")
        res = client.get("/profile/00000000-0000-0000-0000-000000000000")
        with capsys.disabled():
            print(f"  → {res.status_code}")
        assert res.status_code == 404

    def test_valid_job_queued_or_running(self, capsys):
        with capsys.disabled():
            print("\n[profile] upload CSV then immediately poll status")
        res = upload_csv()
        job_id = res.json()["job_id"]
        poll_res = client.get(f"/profile/{job_id}")
        with capsys.disabled():
            print(f"  → job_id={job_id[:8]}... status={poll_res.json().get('status')}")
        assert poll_res.status_code in (200, 202)
        assert poll_res.json()["status"] in ("queued", "running", "done", "failed")

    def test_job_completes_with_result(self, capsys):
        with capsys.disabled():
            print("\n[profile] upload CSV → poll until done → check top-level keys")
        res = upload_csv()
        job_id = res.json()["job_id"]
        with capsys.disabled():
            result = poll_until_done(job_id)
        assert result["status"] == "done"
        r = result["result"]
        for key in ("overview","dtype_info","missing","numeric_stats","outliers","correlation","categorical","distributions","charts","insights"):
            assert key in r, f"Missing key: {key}"
        with capsys.disabled():
            print(f"  ✓ all keys present: overview, dtype_info, missing, numeric_stats, outliers, correlation, categorical, distributions, charts, insights")

    def test_result_overview_fields(self, capsys):
        with capsys.disabled():
            print("\n[profile] check overview fields (rows=3, cols=3, duplicates=1)")
        res = upload_csv()
        job_id = res.json()["job_id"]
        with capsys.disabled():
            result = poll_until_done(job_id)
        overview = result["result"]["overview"]
        with capsys.disabled():
            print(f"  overview: rows={overview['rows']} cols={overview['columns']} dups={overview['duplicate_rows']} missing={overview['missing_pct']}%")
        assert overview["rows"] == 4
        assert overview["columns"] == 3
        assert overview["filename"] == "test.csv"
        assert overview["duplicate_rows"] == 1
        assert "missing_pct" in overview
        assert "memory_usage_mb" in overview

    def test_result_missing_analysis(self, capsys):
        with capsys.disabled():
            print("\n[profile] check missing analysis — age col should have 1 missing")
        res = upload_csv()
        job_id = res.json()["job_id"]
        with capsys.disabled():
            result = poll_until_done(job_id)
        missing = result["result"]["missing"]
        age_missing = missing["per_column"]["age"]["missing"]
        with capsys.disabled():
            print(f"  age missing={age_missing}")
        assert "per_column" in missing
        assert "heatmap" in missing
        assert age_missing == 1

    def test_result_charts_structure(self, capsys):
        with capsys.disabled():
            print("\n[profile] check charts keys")
        res = upload_csv()
        job_id = res.json()["job_id"]
        with capsys.disabled():
            result = poll_until_done(job_id)
        charts = result["result"]["charts"]
        with capsys.disabled():
            print(f"  chart keys: {list(charts.keys())}")
        assert "distributions" in charts
        assert "missing_heatmap" in charts
        assert "outliers" in charts

    def test_result_insights_structure(self, capsys):
        with capsys.disabled():
            print("\n[profile] check insights structure + quality score")
        res = upload_csv()
        job_id = res.json()["job_id"]
        with capsys.disabled():
            result = poll_until_done(job_id)
        insights = result["result"]["insights"]
        rb = insights["rule_based"]
        with capsys.disabled():
            print(f"  quality_score={rb['quality_score']} label={rb['quality_label']} warnings={len(rb['warnings'])} flags={len(rb['flags'])}")
        assert "rule_based" in insights
        assert "ai_available" in insights
        assert "quality_score" in rb
        assert "quality_label" in rb
        assert "flags" in rb
        assert "warnings" in rb
        assert "suggestions" in rb
        assert 0 <= rb["quality_score"] <= 100

    def test_job_summary_endpoint(self, capsys):
        with capsys.disabled():
            print("\n[profile] GET /profile/{id}/summary after job done")
        res = upload_csv()
        job_id = res.json()["job_id"]
        with capsys.disabled():
            poll_until_done(job_id)
        summary_res = client.get(f"/profile/{job_id}/summary")
        data = summary_res.json()
        with capsys.disabled():
            print(f"  → {summary_res.status_code} status={data.get('status')} rows={data.get('overview',{}).get('rows')}")
        assert summary_res.status_code == 200
        assert data["status"] == "done"
        assert "overview" in data
        assert "rows" in data["overview"]

    def test_delete_job(self, capsys):
        with capsys.disabled():
            print("\n[profile] DELETE /profile/{id} then verify 404")
        res = upload_csv()
        job_id = res.json()["job_id"]
        with capsys.disabled():
            poll_until_done(job_id)
        del_res = client.delete(f"/profile/{job_id}")
        with capsys.disabled():
            print(f"  delete → {del_res.status_code} deleted={del_res.json().get('deleted')}")
        assert del_res.status_code == 200
        assert del_res.json()["deleted"] is True
        assert client.get(f"/profile/{job_id}").status_code == 404


# ---------------------------------------------------------------------------
# POST /kaggle/ (mocked)
# ---------------------------------------------------------------------------

class TestKaggle:

    def _fake_download(self):
        async def fake(dataset_slug, download_dir):
            download_dir.mkdir(parents=True, exist_ok=True)
            (download_dir / "dataset.csv").write_text(
                "feature1,feature2,label\n1.0,2.0,0\n3.0,4.0,1\n5.0,6.0,0\n",
                encoding="utf-8",
            )
        return fake

    def test_valid_slug_returns_202(self, capsys):
        with capsys.disabled():
            print("\n[kaggle] POST /kaggle/ valid slug (mocked)")
        with patch("routers.kaggle.download_kaggle_dataset", new=self._fake_download()):
            res = client.post("/kaggle/", json={"source": "owner/dataset-name"})
        with capsys.disabled():
            print(f"  → {res.status_code} job_id={res.json().get('job_id','N/A')[:8]}...")
        assert res.status_code == 202
        assert "job_id" in res.json()

    def test_full_url_normalized(self, capsys):
        with capsys.disabled():
            print("\n[kaggle] POST /kaggle/ full URL (should normalize to slug)")
        with patch("routers.kaggle.download_kaggle_dataset", new=self._fake_download()):
            res = client.post("/kaggle/", json={"source": "https://www.kaggle.com/datasets/zillow/zecon"})
        with capsys.disabled():
            print(f"  → {res.status_code}")
        assert res.status_code == 202

    def test_invalid_source_returns_422(self, capsys):
        with capsys.disabled():
            print("\n[kaggle] POST /kaggle/ invalid source (expect 422)")
        res = client.post("/kaggle/", json={"source": "not-valid"})
        with capsys.disabled():
            print(f"  → {res.status_code}")
        assert res.status_code == 422

    def test_auth_error_returns_401(self, capsys):
        with capsys.disabled():
            print("\n[kaggle] POST /kaggle/ auth failure (expect 401)")
        from utils.kaggle_client import KaggleAuthError
        async def fake_auth_fail(dataset_slug, download_dir):
            raise KaggleAuthError("bad creds")
        with patch("routers.kaggle.download_kaggle_dataset", new=fake_auth_fail):
            res = client.post("/kaggle/", json={"source": "owner/dataset"})
        with capsys.disabled():
            print(f"  → {res.status_code}")
        assert res.status_code == 401

    def test_dataset_not_found_returns_404(self, capsys):
        with capsys.disabled():
            print("\n[kaggle] POST /kaggle/ dataset not found (expect 404)")
        from utils.kaggle_client import KaggleDatasetNotFoundError
        async def fake_not_found(dataset_slug, download_dir):
            raise KaggleDatasetNotFoundError("not found")
        with patch("routers.kaggle.download_kaggle_dataset", new=fake_not_found):
            res = client.post("/kaggle/", json={"source": "owner/dataset"})
        with capsys.disabled():
            print(f"  → {res.status_code}")
        assert res.status_code == 404

    def test_kaggle_job_completes(self, capsys):
        with capsys.disabled():
            print("\n[kaggle] POST /kaggle/ mocked → poll until done")
        with patch("routers.kaggle.download_kaggle_dataset", new=self._fake_download()):
            res = client.post("/kaggle/", json={"source": "owner/dataset"})
        assert res.status_code == 202
        job_id = res.json()["job_id"]
        with capsys.disabled():
            result = poll_until_done(job_id)
        with capsys.disabled():
            print(f"  ✓ status={result['status']} cols={result['result']['overview']['columns']}")
        assert result["status"] == "done"
        assert result["result"]["overview"]["columns"] == 3


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestProfilerEdgeCases:

    def test_all_missing_column(self, capsys):
        with capsys.disabled():
            print("\n[edge] all-missing column b")
        csv = "a,b,c\n1,,3\n4,,6\n7,,9\n"
        res = client.post("/upload/", files={"file": ("edge.csv", csv.encode(), "text/csv")})
        job_id = res.json()["job_id"]
        with capsys.disabled():
            result = poll_until_done(job_id)
        b_missing = result["result"]["missing"]["per_column"]["b"]["missing"]
        with capsys.disabled():
            print(f"  b missing={b_missing}")
        assert result["status"] == "done"
        assert b_missing == 3

    def test_single_column_csv(self, capsys):
        with capsys.disabled():
            print("\n[edge] single column CSV → correlation should be unavailable")
        csv = "value\n1\n2\n3\n4\n5\n"
        res = client.post("/upload/", files={"file": ("single.csv", csv.encode(), "text/csv")})
        job_id = res.json()["job_id"]
        with capsys.disabled():
            result = poll_until_done(job_id)
        corr = result["result"]["charts"]["correlation_matrix"]
        with capsys.disabled():
            print(f"  correlation available={corr.get('available')} reason={corr.get('reason','')}")
        assert result["status"] == "done"
        assert corr["available"] is False

    def test_all_categorical_csv(self, capsys):
        with capsys.disabled():
            print("\n[edge] all categorical columns")
        csv = "city,country\nMumbai,India\nParis,France\nTokyo,Japan\n"
        res = client.post("/upload/", files={"file": ("cats.csv", csv.encode(), "text/csv")})
        job_id = res.json()["job_id"]
        with capsys.disabled():
            result = poll_until_done(job_id)
        dtype_info = result["result"]["dtype_info"]
        with capsys.disabled():
            print(f"  numeric={dtype_info['numeric_count']} categorical={dtype_info['categorical_count']}")
        assert result["status"] == "done"
        assert dtype_info["categorical_count"] == 2
        assert dtype_info["numeric_count"] == 0

    def test_large_ish_csv(self, capsys):
        with capsys.disabled():
            print("\n[edge] 2000 row CSV")
        import random
        rows = ["col_a,col_b,col_c,col_d"]
        for _ in range(2000):
            rows.append(f"{random.random()},{random.randint(0,100)},{'cat'+str(random.randint(1,5))},{random.random() if random.random() > 0.1 else ''}")
        csv = "\n".join(rows)
        res = client.post("/upload/", files={"file": ("large.csv", csv.encode(), "text/csv")})
        job_id = res.json()["job_id"]
        with capsys.disabled():
            result = poll_until_done(job_id, timeout=60)
        with capsys.disabled():
            print(f"  rows={result['result']['overview']['rows']}")
        assert result["status"] == "done"
        assert result["result"]["overview"]["rows"] == 2000

    def test_semicolon_separated_csv(self, capsys):
        with capsys.disabled():
            print("\n[edge] semicolon-separated CSV")
        csv = "name;age;city\nAlice;30;NYC\nBob;25;LA\n"
        res = client.post("/upload/", files={"file": ("semi.csv", csv.encode(), "text/csv")})
        job_id = res.json()["job_id"]
        with capsys.disabled():
            result = poll_until_done(job_id)
        with capsys.disabled():
            print(f"  cols={result['result']['overview']['columns']}")
        assert result["status"] == "done"
        assert result["result"]["overview"]["columns"] == 3

    def test_truly_empty_csv_fails_gracefully(self, capsys):
        with capsys.disabled():
            print("\n[edge] empty CSV → expect job status=failed")
        csv = "\n\n\n"
        res = client.post("/upload/", files={"file": ("empty.csv", csv.encode(), "text/csv")})
        job_id = res.json()["job_id"]
        with capsys.disabled():
            result = poll_until_done(job_id)
        with capsys.disabled():
            print(f"  status={result['status']} error={result.get('error','')[:60]}")
        assert result["status"] == "failed"
        assert "error" in result