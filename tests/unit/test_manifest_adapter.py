import textwrap

import pytest

from classiflow.ui_api.adapters.manifest import parse_metrics


def test_parse_multiclass_metrics_csv(tmp_path):
    """Technical validation CSV with multiclass metrics should be parsed."""
    csv_content = textwrap.dedent(
        """\
        phase,f1_macro,balanced_accuracy,roc_auc_ovr_macro
        val,0.80,0.75,0.88
        val,0.90,0.85,0.92
        """
    )
    csv_path = tmp_path / "metrics_outer_multiclass_eval.csv"
    csv_path.write_text(csv_content)

    metrics = parse_metrics(tmp_path, "technical_validation")

    assert metrics["summary"]["f1_macro"] == pytest.approx(0.85)
    assert metrics["summary"]["balanced_accuracy"] == pytest.approx(0.80)
    assert metrics["per_fold"]["f1_macro"] == [0.80, 0.90]
    assert metrics["per_fold"]["balanced_accuracy"] == [0.75, 0.85]
