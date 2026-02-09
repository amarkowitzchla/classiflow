import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import StratifiedShuffleSplit

from classiflow.config import HierarchicalConfig

pytest.importorskip("torch")
import classiflow.training.hierarchical_cv as hier


def test_hierarchical_es_split_uses_outer_train(tmp_path, monkeypatch):
    n_samples = 30
    labels = np.array(["A", "B"] * (n_samples // 2))
    df = pd.DataFrame({
        "label": labels,
        "f": np.arange(n_samples, dtype=float),
    })
    data_path = tmp_path / "data.csv"
    df.to_csv(data_path, index=False)

    captured = []

    class DummyMLP:
        def __init__(
            self,
            input_dim,
            num_classes,
            hidden_dims,
            lr,
            epochs,
            dropout,
            batch_size,
            early_stopping_patience,
            device,
            random_state,
            verbose,
        ):
            self.num_classes = num_classes

        def fit(self, X_train, y_train, X_val=None, y_val=None, class_weights=None):
            if X_val is not None:
                captured.append(np.array(X_val))
            else:
                captured.append(None)
            return self

        def predict(self, X):
            return np.zeros(len(X), dtype=int)

        def predict_proba(self, X):
            return np.full((len(X), self.num_classes), 1.0 / self.num_classes)

        def save(self, path):
            path.write_text("dummy")

        def get_config(self):
            return {}

        @property
        def best_epoch(self):
            return 0

    monkeypatch.setattr(hier, "TorchMLPWrapper", DummyMLP)
    monkeypatch.setattr(hier, "plot_roc_curve", lambda *args, **kwargs: None)
    monkeypatch.setattr(hier, "plot_pr_curve", lambda *args, **kwargs: None)
    monkeypatch.setattr(hier, "plot_confusion_matrix", lambda *args, **kwargs: None)
    monkeypatch.setattr(hier, "plot_feature_importance", lambda *args, **kwargs: None)
    monkeypatch.setattr(hier, "extract_feature_importance_mlp", lambda *args, **kwargs: None)

    config = HierarchicalConfig(
        data_path=data_path,
        label_l1="label",
        outdir=tmp_path / "out",
        outer_folds=1,
        inner_splits=2,
        random_state=0,
        mlp_epochs=1,
        mlp_hidden=4,
        mlp_batch_size=4,
        output_format="csv",
        verbose=0,
    )

    hier.train_hierarchical(config)

    assert captured
    assert captured[0] is not None
    outer_df = pd.read_csv(config.outdir / "metrics_outer_eval.csv")
    assert "mcc" in outer_df.columns
    l1_rows = outer_df[outer_df["level"] == "L1"]
    assert not l1_rows.empty
    assert l1_rows["mcc"].notna().all()

    stratify_ids = np.arange(len(df))
    outer_cv = StratifiedShuffleSplit(
        n_splits=1,
        test_size=0.2,
        random_state=config.random_state,
    )
    tr_idx, va_idx = next(outer_cv.split(stratify_ids, labels))
    train_ids = set(stratify_ids[tr_idx])
    val_ids = set(stratify_ids[va_idx])

    f_train = df.loc[list(train_ids), "f"].values
    mean = f_train.mean()
    std = np.std(f_train, ddof=0)

    recovered = np.round(captured[0][:, 0] * std + mean).astype(int)
    assert set(recovered).issubset(train_ids)
    assert not set(recovered).intersection(val_ids)
