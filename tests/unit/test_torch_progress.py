"""Unit tests for torch fit progress helpers."""

from classiflow.backends.torch_progress import next_torch_fit_progress, torch_fit_progress


def test_torch_fit_progress_counts_sequential_steps() -> None:
    assert next_torch_fit_progress() is None

    with torch_fit_progress("demo", total=3):
        state1 = next_torch_fit_progress()
        state2 = next_torch_fit_progress()

    assert state1 is not None
    assert state2 is not None
    assert state1.label == "demo"
    assert state1.total == 3
    assert state1.current == 2
    assert state2.current == 2
    assert next_torch_fit_progress() is None
