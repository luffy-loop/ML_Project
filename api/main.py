"""FastAPI service for the Fieldnote student-support research interface."""
import csv
import json
import math
import os
from functools import lru_cache
from io import BytesIO, StringIO
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from src.predict import load_artifact, predict_frame

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = Path(os.getenv("MODEL_PATH", ROOT / "models/student_risk_pipeline.joblib"))
METRICS_PATH = Path(os.getenv("METRICS_PATH", ROOT / "reports/metrics.json"))
QUALITY_PATH = Path(os.getenv("QUALITY_PATH", ROOT / "reports/data_quality.json"))
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(10 * 1024 * 1024)))
MAX_PREDICTION_ROWS = int(os.getenv("MAX_PREDICTION_ROWS", "50000"))

if MAX_UPLOAD_BYTES < 1 or MAX_PREDICTION_ROWS < 1:
    raise ValueError("MAX_UPLOAD_BYTES and MAX_PREDICTION_ROWS must be positive integers.")

app = FastAPI(
    title="Fieldnote · Student Success Analytics",
    version="1.1.0",
    description="Research prototype for voluntary, educator-reviewed student support.",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.strip() for origin in os.getenv(
        "CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173"
    ).split(",") if origin.strip()],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


def read_report(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


@lru_cache(maxsize=1)
def artifact() -> dict[str, Any]:
    return load_artifact(MODEL_PATH)


def safe(value: Any) -> Any:
    """Convert pandas/numpy values to JSON-safe values."""
    if isinstance(value, dict):
        return {str(key): safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe(item) for item in value]
    if hasattr(value, "item"):
        return safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return str(value)


def _validate_csv_header(payload: bytes) -> None:
    """Reject duplicate/blank headers before pandas can silently rename them."""
    try:
        text = payload.decode("utf-8-sig")
        header = next(csv.reader(StringIO(text)), None)
    except (UnicodeDecodeError, csv.Error) as exc:
        raise HTTPException(status_code=400, detail="CSV must be valid UTF-8 text.") from exc
    if not header or not any(name.strip() for name in header):
        raise HTTPException(status_code=400, detail="CSV must include a non-empty header row.")
    cleaned = [name.strip() for name in header]
    if any(not name for name in cleaned):
        raise HTTPException(status_code=400, detail="CSV column names must not be blank.")
    if len(cleaned) != len(set(cleaned)):
        raise HTTPException(status_code=400, detail="CSV contains duplicate column names.")


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "service": "fieldnote",
        "model_available": MODEL_PATH.is_file(),
        "evaluation_available": read_report(METRICS_PATH) is not None,
        "data_quality_available": read_report(QUALITY_PATH) is not None,
        "validation_note": "Artifact availability does not establish temporal validity, calibration, or fairness.",
    }


@app.get("/api/metrics")
def metrics():
    report = read_report(METRICS_PATH)
    return {"available": True, "report": report} if report is not None else {
        "available": False,
        "message": "No evaluation report exists. Run the training pipeline to generate reports/metrics.json.",
    }


@app.get("/api/data-quality")
def data_quality():
    report = read_report(QUALITY_PATH)
    return {"available": True, "report": report} if report is not None else {
        "available": False,
        "message": "No data-quality report exists. Run src/build_features.py first.",
    }


@app.get("/api/model-info")
def model_info():
    if not MODEL_PATH.is_file():
        return {"available": False, "message": "No trained model artifact is installed."}
    try:
        model = artifact()
    except Exception as exc:
        raise HTTPException(status_code=500, detail="The model artifact could not be loaded or validated.") from exc
    pipeline = model["pipeline"]
    classifier = pipeline.named_steps.get("classifier", pipeline) if hasattr(pipeline, "named_steps") else pipeline
    return {
        "available": True,
        "model_type": type(classifier).__name__,
        "feature_count": len(model["feature_columns"]),
        "features": model["feature_columns"],
        "artifact_name": MODEL_PATH.name,
        "note": "Scores are not diagnoses or validated/calibrated probabilities.",
    }


@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    if not file.filename or Path(file.filename).suffix.lower() != ".csv":
        raise HTTPException(status_code=400, detail="Upload a CSV file with a .csv extension.")

    payload = await file.read(MAX_UPLOAD_BYTES + 1)
    if not payload:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")
    if len(payload) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="CSV exceeds the configured upload-size limit.")
    if not MODEL_PATH.is_file():
        raise HTTPException(status_code=503, detail="No trained model is available. Generate features and train the model first.")

    _validate_csv_header(payload)
    try:
        frame = pd.read_csv(BytesIO(payload), encoding="utf-8-sig")
    except (UnicodeDecodeError, pd.errors.ParserError, pd.errors.EmptyDataError, ValueError) as exc:
        raise HTTPException(status_code=400, detail="Could not parse the uploaded CSV. Check its encoding and row structure.") from exc

    if frame.empty:
        raise HTTPException(status_code=400, detail="The CSV has a header but no data rows.")
    if len(frame) > MAX_PREDICTION_ROWS:
        raise HTTPException(
            status_code=413,
            detail=f"CSV has {len(frame):,} rows; the configured limit is {MAX_PREDICTION_ROWS:,}.",
        )

    try:
        model = artifact()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail="No trained model is available.") from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail="The model artifact could not be loaded or validated.") from exc

    try:
        output = predict_frame(frame, model)
    except (KeyError, ValueError, TypeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Prediction failed while scoring this CSV.") from exc

    scores = pd.to_numeric(output["At_Risk_Score"], errors="coerce")
    return {
        "available": True,
        "rows_scored": int(len(output)),
        "median_score": float(scores.median()),
        "mean_score": float(scores.mean()),
        "score_distribution": {
            str(key): int(value)
            for key, value in output["Score_Band"].value_counts(dropna=False).items()
        },
        "score_column_note": "Scores are model outputs, not diagnoses or validated/calibrated probabilities.",
        "records": safe(output.to_dict(orient="records")),
    }


WEB_DIST = ROOT / "web" / "dist"
if WEB_DIST.is_dir():
    app.mount("/assets", StaticFiles(directory=WEB_DIST / "assets"), name="assets")

    @app.get("/", include_in_schema=False)
    def frontend():
        return FileResponse(WEB_DIST / "index.html")

    @app.get("/{path:path}", include_in_schema=False)
    def frontend_routes(path: str):
        candidate = (WEB_DIST / path).resolve()
        if candidate.is_file() and WEB_DIST.resolve() in candidate.parents:
            return FileResponse(candidate)
        return FileResponse(WEB_DIST / "index.html")
