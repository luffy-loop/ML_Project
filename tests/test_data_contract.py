"""Small contract tests for labels and cutoff logic."""
import pandas as pd
import pytest


def test_target_mapping_is_explicit():
    labels = pd.Series(["Withdrawn", "Fail", "Pass", "Distinction", "Unknown"])
    mapped = labels.map({"Withdrawn": 1, "Fail": 1, "Pass": 0, "Distinction": 0})
    assert mapped.iloc[:4].tolist() == [1, 1, 0, 0]
    assert pd.isna(mapped.iloc[4])


def test_cutoff_filters_later_rows():
    events = pd.DataFrame({"date": [0, 28, 29, 100], "clicks": [1, 2, 3, 4]})
    early = events[(events["date"] >= 0) & (events["date"] <= 28)]
    assert early["clicks"].tolist() == [1, 2]
