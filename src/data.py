"""Load and validate OULAD-style CSVs and construct cutoff-safe features."""
from pathlib import Path
import pandas as pd

KEYS = ["id_student", "code_module", "code_presentation"]
REQUIRED_FILES = ["studentInfo.csv", "studentAssessment.csv", "assessments.csv", "studentVle.csv"]


def load_raw_data(data_dir):
    data_dir = Path(data_dir)
    missing = [name for name in REQUIRED_FILES if not (data_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing files in {data_dir}: {', '.join(missing)}")
    info = pd.read_csv(data_dir / "studentInfo.csv")
    student_assessment = pd.read_csv(data_dir / "studentAssessment.csv")
    assessments = pd.read_csv(data_dir / "assessments.csv")
    # Report malformed lines rather than silently hiding the count.
    bad_lines = []
    vle = pd.read_csv(
        data_dir / "studentVle.csv",
        engine="python",
        on_bad_lines=lambda row: bad_lines.append(row) or None,
    )
    quality = {
        "student_rows": len(info),
        "assessment_rows": len(student_assessment),
        "assessment_definitions": len(assessments),
        "vle_rows_kept": len(vle),
        "vle_malformed_rows_skipped": len(bad_lines),
    }
    return info, student_assessment, assessments, vle, quality


def build_features(data_dir, cutoff_day=28):
    """Return features available by cutoff_day and the final-result label.

    OULAD dates are relative to the start of a course presentation. Assessment
    scores are included only when both the assessment due date and submission
    date are <= cutoff_day; VLE events are filtered by studentVle.date. Verify
    dataset date conventions and withdrawal timing before operational claims.
    """
    if cutoff_day < 0:
        raise ValueError("cutoff_day must be >= 0")
    info, sa, assessments, vle, quality = load_raw_data(data_dir)
    info = info.copy()
    info["At_Risk"] = info["final_result"].map(
        {"Withdrawn": 1, "Fail": 1, "Pass": 0, "Distinction": 0}
    )
    info = info.dropna(subset=["At_Risk"])
    info["At_Risk"] = info["At_Risk"].astype(int)

    for col in ["score"]:
        sa[col] = pd.to_numeric(sa[col], errors="coerce")
    assessments["date"] = pd.to_numeric(assessments["date"], errors="coerce")
    assessments["weight"] = pd.to_numeric(assessments["weight"], errors="coerce")
    sa = sa.merge(
        assessments[["id_assessment", "code_module", "code_presentation", "date", "weight"]],
        on="id_assessment", how="left", validate="many_to_one"
    )
    # The assessment due date alone is not enough: a student score is only
    # available to the predictor after it was submitted. Enforce both clocks.
    sa["date_submitted"] = pd.to_numeric(sa.get("date_submitted"), errors="coerce")
    early_sa = sa[
        sa["date"].notna()
        & (sa["date"] <= cutoff_day)
        & sa["date_submitted"].notna()
        & (sa["date_submitted"] <= cutoff_day)
    ].copy()
    assessment_features = early_sa.groupby(KEYS).agg(
        mean_score=("score", "mean"),
        max_score=("score", "max"),
        assessment_count=("score", "count"),
        assessment_weight_seen=("weight", "sum"),
    ).reset_index()

    for col in ["date", "sum_click", "id_student"]:
        vle[col] = pd.to_numeric(vle[col], errors="coerce")
    vle = vle.dropna(subset=["id_student", "date"])
    vle["id_student"] = vle["id_student"].astype("int64")
    early_vle = vle[(vle["date"] >= 0) & (vle["date"] <= cutoff_day)]
    vle_features = early_vle.groupby(KEYS).agg(
        total_clicks=("sum_click", "sum"),
        active_days=("date", "nunique"),
        unique_learning_sites=("id_site", "nunique"),
    ).reset_index()

    base_cols = KEYS + [
        "gender", "region", "highest_education", "age_band",
        "num_of_prev_attempts", "studied_credits", "At_Risk"
    ]
    data = info[base_cols].copy()
    data = data.merge(assessment_features, on=KEYS, how="left", validate="one_to_one")
    data = data.merge(vle_features, on=KEYS, how="left", validate="one_to_one")
    feature_cols = [c for c in data.columns if c not in {"At_Risk"}]
    data[feature_cols] = data[feature_cols].copy()
    quality.update({
        "labelled_rows": len(data),
        "missing_assessment_feature_rows": int(data["assessment_count"].isna().sum()),
        "missing_vle_feature_rows": int(data["total_clicks"].isna().sum()),
        "duplicate_student_course_rows": int(data.duplicated(KEYS).sum()),
        "cutoff_day": cutoff_day,
    })
    return data, quality
