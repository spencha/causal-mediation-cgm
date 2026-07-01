from __future__ import annotations

import numpy as np
import pandas as pd

from data_processing.diatrend.merge_demographics import merge_demographics


def _embeddings() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "subject_id": ["Subject1", "Subject1", "Subject2", "Subject3"],
            "phi_1": [0.1, 0.2, 0.3, 0.4],
            "treat_meal_carbs": [30.0, 40.0, 50.0, 60.0],
        }
    )


def _demographics() -> pd.DataFrame:
    # Mirrors the real workbook's columns (integer Subject, extra columns
    # we intentionally do not carry).
    return pd.DataFrame(
        {
            "Subject": [1, 2, 3],
            "Age": [25, 40, 55],
            "Gender": ["F", "M", "F"],
            "Race": ["a", "b", "c"],
            "Hemoglobin A1C": [7.1, 8.2, 6.9],
            "CGM model": ["x", "y", "z"],
        }
    )


def test_merge_broadcasts_subject_constants():
    m = merge_demographics(_embeddings(), _demographics())
    # Subject1's two episodes both get Subject1's values.
    assert list(m["demo_age"]) == [25, 25, 40, 55]
    assert list(m["demo_sex"]) == ["F", "F", "M", "F"]
    assert list(m["demo_hba1c"]) == [7.1, 7.1, 8.2, 6.9]
    # Row count and existing columns are preserved.
    assert len(m) == 4
    assert "phi_1" in m.columns and "treat_meal_carbs" in m.columns


def test_unrequested_columns_not_carried():
    m = merge_demographics(_embeddings(), _demographics())
    for col in ("Race", "demo_race", "CGM model"):
        assert col not in m.columns


def test_missing_subject_gets_nan():
    demo = _demographics().iloc[:2]  # drop Subject3
    m = merge_demographics(_embeddings(), demo)
    s3 = m[m["subject_id"] == "Subject3"]
    assert s3[["demo_age", "demo_sex", "demo_hba1c"]].isna().all(axis=None)
    # Present subjects are unaffected.
    assert not m[m["subject_id"] == "Subject1"]["demo_age"].isna().any()


def test_raises_on_missing_demographic_column():
    demo = _demographics().drop(columns=["Hemoglobin A1C"])
    try:
        merge_demographics(_embeddings(), demo)
    except ValueError as exc:
        assert "Hemoglobin A1C" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError for missing column")
