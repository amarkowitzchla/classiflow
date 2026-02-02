import pandas as pd
import pytest

from classiflow.io import load_data, load_data_with_groups


def test_load_data_rejects_label_in_features(tmp_path):
    df = pd.DataFrame({
        "label": ["A", "B", "A"],
        "f1": [1, 2, 3],
    })
    path = tmp_path / "data.csv"
    df.to_csv(path, index=False)

    with pytest.raises(ValueError, match="feature_cols contains forbidden columns"):
        load_data(path, label_col="label", feature_cols=["label", "f1"])


def test_load_data_with_groups_rejects_label_or_patient(tmp_path):
    df = pd.DataFrame({
        "label": ["A", "B", "A"],
        "patient_id": ["p1", "p2", "p3"],
        "f1": [1, 2, 3],
    })
    path = tmp_path / "data.csv"
    df.to_csv(path, index=False)

    with pytest.raises(ValueError, match="feature_cols contains forbidden columns"):
        load_data_with_groups(
            path,
            label_col="label",
            patient_col="patient_id",
            feature_cols=["patient_id", "f1"],
        )

    with pytest.raises(ValueError, match="feature_cols contains forbidden columns"):
        load_data_with_groups(
            path,
            label_col="label",
            patient_col="patient_id",
            feature_cols=["label", "f1"],
        )
