import pandas as pd

from src.data import build_features


def test_features_exclude_assessments_and_vle_events_after_cutoff(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    pd.DataFrame([
        {
            "id_student": 1, "code_module": "AAA", "code_presentation": "2014B",
            "gender": "F", "region": "North", "highest_education": "A Level",
            "age_band": "0-35", "num_of_prev_attempts": 0, "studied_credits": 60,
            "final_result": "Pass",
        },
        {
            "id_student": 2, "code_module": "AAA", "code_presentation": "2014B",
            "gender": "M", "region": "South", "highest_education": "A Level",
            "age_band": "0-35", "num_of_prev_attempts": 0, "studied_credits": 60,
            "final_result": "Fail",
        },
    ]).to_csv(raw / "studentInfo.csv", index=False)
    pd.DataFrame([
        {"id_assessment": 10, "id_student": 1, "date_submitted": 5, "score": 80},
        {"id_assessment": 11, "id_student": 1, "date_submitted": 35, "score": 90},
        {"id_assessment": 10, "id_student": 2, "date_submitted": 10, "score": 50},
    ]).to_csv(raw / "studentAssessment.csv", index=False)
    pd.DataFrame([
        {"id_assessment": 10, "code_module": "AAA", "code_presentation": "2014B", "date": 10, "weight": 50},
        {"id_assessment": 11, "code_module": "AAA", "code_presentation": "2014B", "date": 15, "weight": 50},
    ]).to_csv(raw / "assessments.csv", index=False)
    pd.DataFrame([
        {"id_student": 1, "code_module": "AAA", "code_presentation": "2014B", "id_site": 100, "date": 10, "sum_click": 4},
        {"id_student": 1, "code_module": "AAA", "code_presentation": "2014B", "id_site": 101, "date": 35, "sum_click": 100},
        {"id_student": 2, "code_module": "AAA", "code_presentation": "2014B", "id_site": 100, "date": 28, "sum_click": 5},
    ]).to_csv(raw / "studentVle.csv", index=False)

    features, quality = build_features(raw, cutoff_day=28)
    by_student = features.set_index("id_student")

    assert by_student.loc[1, "assessment_count"] == 1
    assert by_student.loc[1, "mean_score"] == 80
    assert by_student.loc[1, "total_clicks"] == 4
    assert by_student.loc[2, "mean_score"] == 50
    assert by_student.loc[2, "total_clicks"] == 5
    assert quality["cutoff_day"] == 28
    assert quality["duplicate_student_course_rows"] == 0
