from sklearn.ensemble import BaggingClassifier

from classiflow.models.estimators import get_estimators, get_param_grids


def test_sklearn_estimators_support_bagged_final_strategy() -> None:
    estimators = get_estimators(final_estimator_strategy="bagged")

    assert isinstance(estimators["LogisticRegression"], BaggingClassifier)
    assert isinstance(estimators["SVM"], BaggingClassifier)
    assert isinstance(estimators["RandomForest"], BaggingClassifier)
    assert isinstance(estimators["GradientBoosting"], BaggingClassifier)


def test_sklearn_param_grids_rewrite_keys_for_bagged_strategy() -> None:
    grids = get_param_grids(final_estimator_strategy="bagged")

    assert "clf__estimator__C" in grids["LogisticRegression"]
    assert "clf__estimator__C" in grids["SVM"]
    assert "clf__estimator__n_estimators" in grids["RandomForest"]
    assert "clf__estimator__learning_rate" in grids["GradientBoosting"]
