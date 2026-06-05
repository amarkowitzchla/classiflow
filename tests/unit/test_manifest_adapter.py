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


def test_parse_metrics_reads_hierarchical_payload_from_metrics_summary(tmp_path):
    summary_path = tmp_path / "metrics_summary.json"
    summary_path.write_text(
        textwrap.dedent(
            """\
            {
              "summary": {"f1_macro": 0.82},
              "per_fold": {"f1_macro": [0.8, 0.84]},
              "hierarchical": {
                "L1": {
                  "summary": {"accuracy": 0.9}
                },
                "L2": {
                  "summary": {"accuracy": 0.78}
                }
              }
            }
            """
        )
    )

    metrics = parse_metrics(tmp_path, "technical_validation")

    assert metrics["summary"]["f1_macro"] == pytest.approx(0.82)
    assert metrics["hierarchical"]["L1"]["summary"]["accuracy"] == pytest.approx(0.9)
    assert metrics["hierarchical"]["L2"]["summary"]["accuracy"] == pytest.approx(0.78)


def test_parse_metrics_reads_hierarchical_payload_from_metrics_json(tmp_path):
    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text(
        textwrap.dedent(
            """\
            {
              "overall": {
                "accuracy": 0.88,
                "f1_macro": 0.85,
                "sensitivity": 0.81,
                "specificity": 0.93,
                "ppv": 0.77,
                "npv": 0.95
              },
              "hierarchical": {
                "L1": {"summary": {"accuracy": 0.9}},
                "L2": {"summary": {"accuracy": 0.8}}
              }
            }
            """
        )
    )

    metrics = parse_metrics(tmp_path, "independent_test")

    assert metrics["summary"]["accuracy"] == pytest.approx(0.88)
    assert metrics["summary"]["sensitivity"] == pytest.approx(0.81)
    assert metrics["summary"]["specificity"] == pytest.approx(0.93)
    assert metrics["summary"]["ppv"] == pytest.approx(0.77)
    assert metrics["summary"]["npv"] == pytest.approx(0.95)
    assert metrics["hierarchical"]["L1"]["summary"]["accuracy"] == pytest.approx(0.9)
    assert metrics["hierarchical"]["L2"]["summary"]["accuracy"] == pytest.approx(0.8)
