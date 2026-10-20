import pandas as pd
import pytest

from src.evaluate_temporal import evaluate


def test_temporal_evaluation_purges_students_shared_with_latest_presentation(tmp_path):
    rows = []
    # Two earlier presentations contain the same three students.
    for presentation in ("2013B", "2013J"):
        for student, label, score in ((1, 0, 0.1), (2, 1, 0.9), (3, 0, 0.2)):
            rows.append({
                "id_student": student,
                "code_presentation": presentation,
                "At_Risk": label,
                "score": score,
            })
    # Student 3 has a record in the held-out presentation too and must be
    # removed from training; students 4 and 5 are new to the earlier cohorts.
    for student, label, score in ((3, 0, 0.3), (4, 1, 0.8), (5, 0, 0.15)):
        rows.append({
            "id_student": student,
            "code_presentation": "2014B",
            "At_Risk": label,
            "score": score,
        })
    source = tmp_path / "features.csv"
    output = tmp_path / "temporal_metrics.json"
    pd.DataFrame(rows).to_csv(source, index=False)

    report = evaluate(source, output)

    assert report["test_presentation"] == "2014B"
    assert report["student_overlap"] == 0
    assert report["purged_overlapping_student_rows"] == 2
    assert report["train_students"] == 2
    assert report["test_students"] == 3
    assert output.is_file()


def test_temporal_evaluation_requires_multiple_presentations(tmp_path):
    source = tmp_path / "features.csv"
    pd.DataFrame({
        "id_student": [1, 2],
        "code_presentation": ["2014B", "2014B"],
        "At_Risk": [0, 1],
        "score": [0.1, 0.9],
    }).to_csv(source, index=False)

    with pytest.raises(ValueError, match="At least two presentations"):
        evaluate(source, tmp_path / "report.json")
