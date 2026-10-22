"""Unit tests for target mapping, cutoff enforcement and grouped model training."""
import json
from pathlib import Path

import pandas as pd
import pytest

from src.data import build_features
from src.train import train, threshold_analysis, subgroup_audit
from src.evaluate_baseline import evaluate as evaluate_baseline


def _write_minimal_oulad(directory: Path):
    pd.DataFrame({
        "id_student": [1, 2, 3, 4],
        "code_module": ["AAA"] * 4,
        "code_presentation": ["2014J"] * 4,
        "gender": ["M", "F", "M", "F"],
        "region": ["North", "South", "North", "South"],
        "highest_education": ["A Level"] * 4,
        "age_band": ["0-35"] * 4,
        "num_of_prev_attempts": [0, 0, 1, 0],
        "studied_credits": [60, 60, 120, 60],
        "final_result": ["Pass", "Fail", "Withdrawn", "Distinction"],
    }).to_csv(directory / "studentInfo.csv", index=False)
    pd.DataFrame({
        "id_assessment": [10, 10, 11, 11, 12, 12],
        "id_student": [1, 2, 3, 4, 1, 2],
        "date_submitted": [5, 7, 20, 35, 35, 36],
        "is_banked": [0] * 6,
        "score": [80, 30, 0, 90, 100, 45],
    }).to_csv(directory / "studentAssessment.csv", index=False)
    pd.DataFrame({
        "id_assessment": [10, 11, 12],
        "code_module": ["AAA"] * 3,
        "code_presentation": ["2014J"] * 3,
        "assessment_type": ["TMA"] * 3,
        "date": [10, 28, 40],
        "weight": [30, 30, 40],
    }).to_csv(directory / "assessments.csv", index=False)
    pd.DataFrame({
        "code_module": ["AAA"] * 5,
        "code_presentation": ["2014J"] * 5,
        "id_student": [1, 1, 2, 3, 4],
        "id_site": [100, 101, 100, 100, 102],
        "date": [0, 28, 29, 10, 70],
        "sum_click": [2, 4, 99, 5, 7],
    }).to_csv(directory / "studentVle.csv", index=False)


def test_build_features_enforces_day_cutoff_and_one_row_per_student(tmp_path):
    _write_minimal_oulad(tmp_path)
    features, quality = build_features(tmp_path, cutoff_day=28)
    assert len(features) == 4
    assert features["id_student"].is_unique
    student1 = features.set_index("id_student").loc[1]
    assert student1["assessment_count"] == 1
    assert student1["mean_score"] == 80
    assert student1["total_clicks"] == 6
    assert student1["unique_learning_sites"] == 2
    # An assessment due by day 28 but submitted on day 35 must not contribute.
    student4 = features.set_index("id_student").loc[4]
    assert pd.isna(student4["assessment_count"])
    assert quality["cutoff_day"] == 28
    assert quality["duplicate_student_course_rows"] == 0


def test_target_mapping_drops_unknown_labels(tmp_path):
    _write_minimal_oulad(tmp_path)
    info_path = tmp_path / "studentInfo.csv"
    info = pd.read_csv(info_path)
    info.loc[len(info)] = [5, "AAA", "2014J", "M", "North", "A Level", "0-35", 0, 60, "Unknown"]
    info.to_csv(info_path, index=False)
    features, _ = build_features(tmp_path, cutoff_day=28)
    assert set(features["At_Risk"].unique()) == {0, 1}
    assert 5 not in set(features["id_student"])


def test_train_saves_pipeline_metrics_and_keeps_groups_separate(tmp_path):
    rows = []
    # 40 unique students with balanced labels and a predictive signal.
    for student in range(40):
        rows.append({
            "id_student": student,
            "At_Risk": student % 2,
            "signal": student % 2,
            "category": "A" if student % 2 else "B",
        })
    features_path = tmp_path / "features.csv"
    model_path = tmp_path / "models" / "model.joblib"
    metrics_path = tmp_path / "reports" / "metrics.json"
    pd.DataFrame(rows).to_csv(features_path, index=False)
    metrics = train(features_path, model_path, metrics_path, seed=7)
    assert model_path.exists()
    assert metrics_path.exists()
    saved = json.loads(metrics_path.read_text())
    assert saved["split"] == "GroupShuffleSplit by id_student"
    assert saved["train_students"] + saved["test_students"] == 40
    assert saved["roc_auc"] is not None
    assert 0 <= metrics["pr_auc"] <= 1
    assert 0 <= metrics["brier_score"] <= 1
    assert [row["threshold"] for row in metrics["threshold_analysis"]] == [0.25, 0.5, 0.75]
    assert len(metrics["permutation_importance_top10"]) > 0
    assert "audit_notes" in metrics


def test_threshold_analysis_exposes_workload_tradeoffs():
    rows = threshold_analysis([0, 1, 1, 0], [0.1, 0.9, 0.4, 0.6])
    assert [row["flagged_count"] for row in rows] == [3, 2, 1]
    assert all(0 <= row["flagged_rate"] <= 1 for row in rows)


def test_subgroup_audit_reports_only_groups_meeting_minimum_size():
    X = pd.DataFrame({"gender": ["F"] * 5 + ["M"] * 2})
    y = pd.Series([1, 0, 1, 0, 1, 0, 1])
    result = subgroup_audit(X, y, [0.8, 0.2, 0.7, 0.3, 0.9, 0.8, 0.1], min_group_size=5)
    assert len(result["gender"]) == 1
    assert result["gender"][0]["group"] == "F"
    assert result["gender"][0]["n"] == 5


def test_train_rejects_single_class_dataset(tmp_path):
    features_path = tmp_path / "features.csv"
    pd.DataFrame({
        "id_student": range(6),
        "At_Risk": [1] * 6,
        "signal": range(6),
    }).to_csv(features_path, index=False)
    with pytest.raises(ValueError, match="both At_Risk classes"):
        train(features_path, tmp_path / "model.joblib", tmp_path / "metrics.json")


def test_train_rejects_final_result_target_leakage(tmp_path):
    features_path = tmp_path / "leaky_features.csv"
    pd.DataFrame({
        "id_student": range(8),
        "At_Risk": [0, 1] * 4,
        "signal": [0, 1] * 4,
        "final_result": ["Pass", "Fail"] * 4,
    }).to_csv(features_path, index=False)
    with pytest.raises(ValueError, match="final_result.*must not be used as a predictor"):
        train(features_path, tmp_path / "model.joblib", tmp_path / "metrics.json")


def test_baseline_evaluation_rejects_final_result_target_leakage(tmp_path):
    features_path = tmp_path / "leaky_features.csv"
    pd.DataFrame({
        "id_student": range(8),
        "At_Risk": [0, 1] * 4,
        "signal": [0, 1] * 4,
        "final_result": ["Pass", "Fail"] * 4,
    }).to_csv(features_path, index=False)
    with pytest.raises(ValueError, match="final_result.*must not be used as a predictor"):
        evaluate_baseline(features_path, output=tmp_path / "baseline.json")
