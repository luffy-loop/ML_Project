import numpy as np
import pandas as pd
import pytest
from src.predict import predict_frame

class DummyPipeline:
    def predict_proba(self, X):
        return np.column_stack([np.full(len(X), .4), np.full(len(X), .6)])

def test_predict_frame_scores_and_preserves_input():
    frame = pd.DataFrame({"x": [1, 2], "extra": ["a", "b"]})
    result = predict_frame(frame, {"pipeline": DummyPipeline(), "feature_columns": ["x"]})
    assert result["At_Risk_Score"].tolist() == [.6, .6]
    assert result["Score_Band"].tolist() == ["HIGH_SCORE", "HIGH_SCORE"]
    assert "extra" in result.columns

def test_predict_frame_rejects_missing_columns():
    with pytest.raises(ValueError, match="Missing required"):
        predict_frame(pd.DataFrame({"extra": [1]}), {"pipeline": DummyPipeline(), "feature_columns": ["x"]})

def test_predict_frame_rejects_empty_input():
    with pytest.raises(ValueError, match="no rows"):
        predict_frame(pd.DataFrame(), {"pipeline": DummyPipeline(), "feature_columns": ["x"]})
