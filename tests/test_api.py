import numpy as np
from fastapi.testclient import TestClient

import api.main as api


class DummyPipeline:
    def predict_proba(self, frame):
        values = frame["score"].astype(float).to_numpy()
        probabilities = np.clip(values, 0, 1)
        return np.column_stack([1 - probabilities, probabilities])


DUMMY_ARTIFACT = {"pipeline": DummyPipeline(), "feature_columns": ["score"]}
client = TestClient(api.app)


def setup_model(monkeypatch, tmp_path):
    model_path = tmp_path / "dummy.joblib"
    model_path.write_bytes(b"test artifact marker")
    monkeypatch.setattr(api, "MODEL_PATH", model_path)
    monkeypatch.setattr(api, "artifact", lambda: DUMMY_ARTIFACT)


def test_health_reports_real_artifact_state():
    response = client.get("/api/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert isinstance(body["model_available"], bool)
    assert "validation_note" in body


def test_metrics_has_explicit_empty_state():
    response = client.get("/api/metrics")
    assert response.status_code == 200
    body = response.json()
    assert isinstance(body["available"], bool)
    assert body["available"] or "message" in body


def test_prediction_rejects_non_csv_file():
    response = client.post("/api/predict", files={"file": ("notes.txt", b"x,y\n1,2\n", "text/plain")})
    assert response.status_code == 400


def test_prediction_rejects_empty_payload():
    response = client.post("/api/predict", files={"file": ("empty.csv", b"", "text/csv")})
    assert response.status_code == 400


def test_prediction_returns_scored_rows(monkeypatch, tmp_path):
    setup_model(monkeypatch, tmp_path)
    response = client.post(
        "/api/predict",
        files={"file": ("features.csv", b"score,comment\n0.2,ok\n0.9,review\n", "text/csv")},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["rows_scored"] == 2
    assert len(body["records"]) == 2
    assert body["records"][0]["At_Risk_Score"] == 0.2
    assert "score_column_note" in body


def test_prediction_rejects_duplicate_headers(monkeypatch, tmp_path):
    setup_model(monkeypatch, tmp_path)
    response = client.post(
        "/api/predict",
        files={"file": ("duplicate.csv", b"score,score\n0.2,0.8\n", "text/csv")},
    )
    assert response.status_code == 400
    assert "duplicate" in response.json()["detail"].lower()


def test_prediction_rejects_missing_predictors(monkeypatch, tmp_path):
    setup_model(monkeypatch, tmp_path)
    response = client.post(
        "/api/predict",
        files={"file": ("missing.csv", b"other\n1\n", "text/csv")},
    )
    assert response.status_code == 422
    assert "score" in response.json()["detail"]


def test_prediction_rejects_infinite_predictors(monkeypatch, tmp_path):
    setup_model(monkeypatch, tmp_path)
    response = client.post(
        "/api/predict",
        files={"file": ("infinite.csv", b"score\ninf\n", "text/csv")},
    )
    assert response.status_code == 422
    assert "infinity" in response.json()["detail"].lower()


def test_prediction_rejects_oversized_upload(monkeypatch, tmp_path):
    setup_model(monkeypatch, tmp_path)
    monkeypatch.setattr(api, "MAX_UPLOAD_BYTES", 8)
    response = client.post(
        "/api/predict",
        files={"file": ("large.csv", b"score\n0.123456789\n", "text/csv")},
    )
    assert response.status_code == 413


def test_prediction_rejects_too_many_rows(monkeypatch, tmp_path):
    setup_model(monkeypatch, tmp_path)
    monkeypatch.setattr(api, "MAX_PREDICTION_ROWS", 1)
    response = client.post(
        "/api/predict",
        files={"file": ("many.csv", b"score\n0.2\n0.8\n", "text/csv")},
    )
    assert response.status_code == 413
