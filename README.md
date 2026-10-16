# Fieldnote — Student Success Analytics

A polished full-stack research workspace for exploring whether early-course activity and assessment data may help educators offer timely, voluntary academic support. It combines a React + TypeScript interface with FastAPI and the cutoff-aware Python ML pipeline.

> **Research status:** This is a prototype, not a validated early-warning system. The official OULAD dataset is not bundled, and the final-result label may include withdrawal events that occurred before the prediction cutoff. Scores are not validated or calibrated probabilities. Do not use this tool to automate grading, discipline, admissions, funding, exclusion, or access restrictions. Real-data, temporal-validity, calibration and subgroup checks must be completed before prospective claims.

## Included

- React + TypeScript interface: overview, prediction desk, model review, data quality and study notes.
- FastAPI service: health, model metadata, saved reports and guarded CSV scoring.
- Cutoff-aware feature generation and grouped evaluation diagnostics.
- Evidence-first empty states; no fabricated demonstration metrics or predictions.
- Existing Streamlit fallback, Python tests and GitHub Actions checks.

## Requirements and dataset

Python 3.10+ (3.11 recommended), Node.js 20+ and npm. Obtain OULAD from the [official dataset page](https://analyse.kmi.open.ac.uk/open_dataset), review its terms, and place studentInfo.csv, studentAssessment.csv, assessments.csv and studentVle.csv in data/raw/. Raw student data and generated model artifacts are not committed.

## Python setup

```bash
python -m venv .venv
# Windows PowerShell: .venv\\Scripts\\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Generate features and train

```bash
python src/build_features.py --data-dir data/raw --cutoff-day 28
python src/evaluate_baseline.py --features data/processed/features_day28.csv
python src/evaluate_temporal.py --features data/processed/features_day28.csv
python src/train.py --features data/processed/features_day28.csv
pytest -q
```

This generates local model and report files such as models/student_risk_pipeline.joblib, reports/metrics.json and reports/data_quality.json. Verify date semantics, label timing and temporal-holdout suitability before citing metrics as evidence of prospective performance. A grouped random holdout does not establish future-cohort generalization.

## Run the full-stack app

Terminal 1 — API from the repository root:

```bash
uvicorn api.main:app --reload --host 127.0.0.1 --port 8000
```

Terminal 2 — frontend:

```bash
cd web
npm install
npm run dev
```

Open http://localhost:5173. Vite proxies API requests to FastAPI. API docs: http://127.0.0.1:8000/docs. Scoring requires a trained artifact and prepared feature CSV with the model's predictor columns. Uploads default to 10 MB and 50,000 rows.

## Streamlit fallback

```bash
streamlit run app.py
```

## Docker

After feature generation and training have created local model artifacts and reports:

```bash
docker compose up --build
```

Open http://localhost:8000. Compose mounts model and report directories read-only. Training is never triggered implicitly on startup.

## API endpoints

| Endpoint | Purpose |
|---|---|
| GET /api/health | Service and artifact availability |
| GET /api/model-info | Model type and predictor fields |
| GET /api/metrics | Saved evaluation report |
| GET /api/data-quality | Saved feature-generation report |
| POST /api/predict | Score a prepared CSV |
| GET /docs | Interactive API documentation |

Environment options: MODEL_PATH, METRICS_PATH, QUALITY_PATH, MAX_UPLOAD_BYTES, MAX_PREDICTION_ROWS and CORS_ORIGINS. This demo has no authentication; protect it appropriately before using non-public data.

## Quality checks

```bash
pytest -q
cd web && npm install && npm run build
```

Python tests use synthetic inputs and do not validate predictive quality on the official dataset. The frontend only displays saved metrics and quality counts when real reports exist.

## Evaluation and responsible use

1. Define the prediction point and use only information available by that time.
2. Keep id_student as a grouping key, never a predictor.
3. Verify assessment and VLE date semantics against official documentation.
4. Define a prospective target that respects withdrawal timing before making early-warning claims.
5. Evaluate on a genuinely later presentation and inspect calibration and subgroup errors.
6. Permutation importance is predictive association, not causation; subgroup diagnostics are not fairness certification.
7. Use outputs only for voluntary supportive outreach with educator review. Never automate punitive or exclusionary decisions.

See MODEL_CARD.md for intended use and limitations.
