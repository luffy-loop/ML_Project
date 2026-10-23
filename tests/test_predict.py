import numpy as np
import pandas as pd
import pytest

from src.predict import predict_frame


class DummyPipeline:
    classes_ = np.array([0, 1])

    def __init__(self, probabilities=None):
        self.probabilities = probabilities

    def predict_proba(self, frame):
        probabilities = self.probabilities
        if probabilities is None:
            probabilities = np.full(len(frame), 0.5)
        return np.column_stack([1 - probabilities, probabilities])


def artifact(probabilities=None):
    return {"pipeline": DummyPipeline(probabilities), "feature_columns": ["score"]}


def test_predict_frame_preserves_input_and_adds_scores():
    frame = pd.DataFrame({"score": [0.1, 0.8], "note": ["a", "b"]})
    result = predict_frame(frame, artifact())
    assert list(frame.columns) == ["score", "note"]
    assert result["At_Risk_Score"].tolist() == [0.5, 0.5]
    assert result["Score_Band"].tolist() == ["ELEVATED_SCORE", "ELEVATED_SCORE"]


def test_predict_frame_uses_class_label_one_not_probability_column_position():
    class ReverseOrderPipeline:
        classes_ = np.array([1, 0])

        def predict_proba(self, frame):
            # Column 0 is At_Risk=1, column 1 is At_Risk=0.
            return np.column_stack([np.full(len(frame), 0.9), np.full(len(frame), 0.1)])

    result = predict_frame(
        pd.DataFrame({"score": [0.2]}),
        {"pipeline": ReverseOrderPipeline(), "feature_columns": ["score"]},
    )
    assert result.loc[0, "At_Risk_Score"] == pytest.approx(0.9)
    assert result.loc[0, "Score_Band"] == "VERY_HIGH_SCORE"


def test_predict_frame_rejects_empty_input():
    with pytest.raises(ValueError, match="no rows"):
        predict_frame(pd.DataFrame(columns=["score"]), artifact())


def test_predict_frame_rejects_missing_columns():
    with pytest.raises(ValueError, match="Missing required predictor columns"):
        predict_frame(pd.DataFrame({"other": [1]}), artifact())


def test_predict_frame_rejects_duplicate_columns():
    frame = pd.DataFrame([[0.2, 0.3]], columns=["score", "score"])
    with pytest.raises(ValueError, match="Duplicate input columns"):
        predict_frame(frame, artifact())


def test_predict_frame_rejects_infinite_numeric_values():
    with pytest.raises(ValueError, match="infinity"):
        predict_frame(pd.DataFrame({"score": [np.inf]}), artifact())


@pytest.mark.parametrize("probabilities", [[np.nan], [np.inf], [-0.1], [1.1]])
def test_predict_frame_rejects_invalid_model_scores(probabilities):
    with pytest.raises(ValueError, match="scores"):
        predict_frame(pd.DataFrame({"score": [0.2]}), artifact(np.asarray(probabilities)))


def test_predict_frame_rejects_invalid_artifact():
    with pytest.raises(ValueError, match="feature_columns"):
        predict_frame(pd.DataFrame({"score": [0.2]}), {"pipeline": DummyPipeline(), "feature_columns": []})


@pytest.mark.parametrize(
    "matrix, message",
    [
        ([[1.5, 0.5]], "finite and in \\[0, 1\\]"),
        ([[np.nan, 0.5]], "finite and in \\[0, 1\\]"),
        ([[0.8, 0.8]], "each row must sum to 1"),
    ],
)
def test_predict_frame_rejects_invalid_probability_matrix(matrix, message):
    class MalformedPipeline:
        classes_ = np.array([0, 1])

        def predict_proba(self, frame):
            return np.asarray(matrix, dtype=float)

    with pytest.raises(ValueError, match=message):
        predict_frame(
            pd.DataFrame({"score": [0.2]}),
            {"pipeline": MalformedPipeline(), "feature_columns": ["score"]},
        )
