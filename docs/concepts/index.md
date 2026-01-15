# Concepts

Understand the theory and design decisions behind classiflow.

<div class="grid cards" markdown>

-   :material-folder-outline:{ .lg .middle } **What is a Run**

    ---

    Understanding training runs, artifacts, and the run manifest.

    [:octicons-arrow-right-24: Learn More](what-is-a-run.md)

-   :material-pipeline:{ .lg .middle } **Pipeline Boundaries**

    ---

    What happens inside each fold and why it matters for validity.

    [:octicons-arrow-right-24: Learn More](pipeline-boundaries.md)

-   :material-scale-balance:{ .lg .middle } **Resampling vs Weighting**

    ---

    SMOTE, class weights, and when to use each.

    [:octicons-arrow-right-24: Learn More](resampling-vs-weighting.md)

-   :material-chart-line:{ .lg .middle } **Metrics Interpretation**

    ---

    Understanding AUC, F1, MCC, and calibration for imbalanced data.

    [:octicons-arrow-right-24: Learn More](metrics-interpretation.md)

-   :material-sync:{ .lg .middle } **Reproducibility**

    ---

    Seeds, determinism, and what classiflow guarantees.

    [:octicons-arrow-right-24: Learn More](reproducibility.md)

</div>

## Core Principles

Classiflow is built on these principles:

### 1. Unbiased Evaluation

Nested cross-validation separates model selection from performance estimation, preventing the optimistic bias common in single-level CV.

### 2. Leakage Prevention

Every transformation that learns from data (scaling, SMOTE, feature selection) is applied inside training folds only.

### 3. Reproducibility by Default

Every run captures its environment, configuration, and data hash in a manifest for complete reproducibility.

### 4. Publication Readiness

Outputs are designed for direct inclusion in peer-reviewed manuscripts, with proper uncertainty quantification.
