"""Data schema validation using pydantic."""

from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel, Field, field_validator


class DataSchema(BaseModel):
    """Schema for ML-ready CSV data."""

    n_samples: int = Field(ge=10, description="Number of samples")
    n_features: int = Field(ge=1, description="Number of features")
    n_classes: int = Field(ge=2, description="Number of classes")
    class_counts: dict = Field(description="Per-class sample counts")
    feature_names: List[str] = Field(description="Feature column names")
    class_names: List[str] = Field(description="Class labels")
    missing_features: Optional[List[str]] = Field(default=None, description="Features with missing values")

    @field_validator("class_counts")
    @classmethod
    def validate_class_counts(cls, v):
        """Ensure all classes have at least 2 samples."""
        for cls_name, count in v.items():
            if count < 2:
                raise ValueError(f"Class '{cls_name}' has only {count} sample(s). Need at least 2.")
        return v

    @classmethod
    def from_data(cls, X, y) -> DataSchema:
        """
        Create schema from data.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix
        y : pd.Series
            Labels

        Returns
        -------
        DataSchema
        """
        import pandas as pd

        class_counts = y.value_counts().to_dict()
        missing_features = X.columns[X.isna().any()].tolist() or None

        return cls(
            n_samples=X.shape[0],
            n_features=X.shape[1],
            n_classes=y.nunique(),
            class_counts=class_counts,
            feature_names=list(X.columns),
            class_names=sorted(y.unique().tolist()),
            missing_features=missing_features,
        )
