# DataLens

Automated dataset quality profiler. Upload a CSV (or a ZIP of CSVs), or point it at a Kaggle dataset, and DataLens produces a full quality report: overview stats, column types, missing-value analysis, outliers, correlations, categorical breakdowns, distribution charts, a 0–100 quality score, and optional AI-written insights.

It is a FastAPI backend with a single-page HTML frontend served from the same process.

## Features

- **Multiple input sources** — `.csv` upload, `.zip` upload (with a picker when the archive holds several CSVs), or Kaggle dataset URL/slug
- **Robust CSV loading** — automatic encoding detection, separator sniffing (`,` `;` tab `|`), numeric/datetime type refinement
- **Profiling**
  - Overview: rows, columns, missing cells, duplicate rows, memory usage
  - Per-column type, missing %, and uniqueness
  - Missing-value heatmap and co-missing patterns
  - Descriptive stats with skewness, kurtosis, coefficient of variation
  - Outliers via IQR and Z-score, with box-plot data
  - Pearson correlation matrix with strong-pair detection (|r| ≥ 0.7)
  - Categorical value counts and cardinality
  - Histogram / bar chart data for every column
- **Insights**
  - Rule-based quality score (Excellent → Critical) with warnings and suggestions
  - Optional Claude-generated summary, ML-readiness verdict and recommended actions (requires `ANTHROPIC_API_KEY`)
- **Async job model** — submissions return a `job_id` immediately; profiling runs in the background and results are kept for 1 hour

## Project structure

```
DataLens/
├── backend/
│   ├── main.py               # FastAPI app, CORS, router registration, serves frontend/
│   ├── routers/
│   │   ├── upload.py         # POST /upload/
│   │   ├── kaggle.py         # POST /kaggle/
│   │   └── profile.py        # GET/DELETE /profile/{job_id}
│   ├── services/
│   │   ├── ingestor.py       # CSV/ZIP loading, encoding + separator detection, cleanup
│   │   ├── profiler.py       # Profiling pipeline (runs as background task)
│   │   ├── visualizer.py     # Chart-ready payloads
│   │   └── insight.py        # Rule-based + Claude insights
│   ├── utils/
│   │   ├── file_handler.py   # Temp job dirs and delayed cleanup
│   │   └── kaggle_client.py  # Kaggle download wrapper
│   ├── models/schemas.py     # Pydantic response models
│   ├── test.py               # pytest suite
│   ├── requirements.txt
│   └── .env.example
└── frontend/
    └── index.html            # Single-page UI (no build step)
```

## Getting started

Requires Python 3.10+.

```bash
git clone https://github.com/ChackoPAbraham19/DataLens.git
cd DataLens

python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r backend/requirements.txt

cp backend/.env.example backend/.env   # optional: add Kaggle / Anthropic keys

cd backend
uvicorn main:app --reload --port 8000
```

Open http://localhost:8000 for the UI, or http://localhost:8000/docs for the interactive API docs.

### Configuration

All settings are environment variables, loaded from `backend/.env` if present:

| Variable | Required | Purpose |
|---|---|---|
| `KAGGLE_USERNAME`, `KAGGLE_KEY` | For Kaggle ingestion | API token from https://www.kaggle.com/settings/account |
| `ANTHROPIC_API_KEY` | No | Enables AI insights; without it only rule-based insights are returned |
| `ANTHROPIC_MODEL` | No | Claude model for insights (default `claude-sonnet-5-5`) |

Job files are written to `/tmp/datalens_jobs/<job_id>/` and removed after one hour.

## API

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Health check |
| `POST` | `/upload/` | Upload a `.csv` or `.zip` (max 200 MB). Optional query param `target_csv` |
| `POST` | `/kaggle/` | Body: `{"source": "owner/dataset" or full URL, "target_csv": optional}` |
| `GET` | `/profile/{job_id}` | Poll job status; returns full `result` when done |
| `GET` | `/profile/{job_id}/summary` | Status plus overview only |
| `DELETE` | `/profile/{job_id}` | Delete a job and its files |

**Response codes for submissions:** `202` job queued · `300` archive/dataset has several CSVs — resubmit with `target_csv` (response lists `available_csvs`) · `400` bad file · `401` Kaggle auth failed · `404` Kaggle dataset not found · `413` file too large · `422` validation error / no CSVs.

**Job lifecycle:** `queued` → `running` (with `progress` 0–100 and `step`) → `done` or `failed`. Poll every 2–3 seconds.

### Example

```bash
curl -F file=@data.csv http://localhost:8000/upload/
# {"job_id":"7a34...","status":"queued",...}

curl http://localhost:8000/profile/7a34...
```

A completed result contains: `overview`, `dtype_info`, `missing`, `numeric_stats`, `outliers`, `correlation`, `categorical`, `distributions`, `charts`, `insights`.

## Running tests

```bash
pip install pytest
cd backend
pytest test.py -v
```

Kaggle calls are mocked, so the suite runs offline without credentials.

## License

[MIT](LICENSE)
