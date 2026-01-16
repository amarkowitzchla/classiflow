"""Streamlit page for model inference."""

from pathlib import Path
import streamlit as st
import pandas as pd
import sys

# Add src to path for package imports
root = Path(__file__).parent.parent.parent.parent
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from classiflow.inference import run_inference, InferenceConfig

st.set_page_config(page_title="Inference Pipeline", page_icon="🔮", layout="wide")

st.title("Inference Pipeline")
st.markdown(
    """
    Apply trained models to new data for prediction and evaluation.

    **Features:**
    - Load trained model artifacts (binary, meta-classifier, or hierarchical)
    - Upload new data for inference
    - Automatic feature alignment (strict or lenient mode)
    - Optional evaluation (if ground-truth labels provided)
    - Export predictions, metrics, and publication-ready plots
    """
)

# Paths
DERIVED = Path("derived")
DERIVED_HIER = Path("derived_hierarchical")
DATA = Path("data")

# Session state for uploaded file
if "inference_data" not in st.session_state:
    st.session_state["inference_data"] = None

st.markdown("---")

# Step 1: Select model run
st.subheader("1️⃣ Select Model Run")

col1, col2 = st.columns(2)

with col1:
    run_source = st.selectbox(
        "Run source",
        options=["Standard (binary/meta)", "Hierarchical"],
        help="Type of training run to load",
    )

with col2:
    if run_source == "Standard (binary/meta)":
        base_dir = DERIVED
    else:
        base_dir = DERIVED_HIER

    if not base_dir.exists():
        st.warning(f"Directory not found: {base_dir}")
        available_runs = []
    else:
        # Find fold directories
        available_runs = sorted([d for d in base_dir.glob("fold*") if d.is_dir()])

    if available_runs:
        run_dir = st.selectbox(
            "Select fold",
            options=available_runs,
            format_func=lambda x: x.name,
        )
    else:
        st.error(f"No trained runs found in {base_dir}")
        st.stop()

st.caption(f"**Selected run:** `{run_dir}`")

# Step 2: Upload data
st.markdown("---")
st.subheader("2️⃣ Upload Inference Data")

uploaded_file = st.file_uploader(
    "Upload CSV file",
    type=["csv"],
    help="CSV file with same features as training data",
)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.session_state["inference_data"] = df

    # Save to disk
    DATA.mkdir(exist_ok=True)
    inference_csv = DATA / uploaded_file.name
    with open(inference_csv, "wb") as f:
        f.write(uploaded_file.getvalue())

    st.success(f"✅ Loaded {len(df)} samples with {len(df.columns)} columns")

    with st.expander("📋 Preview data (first 20 rows)"):
        st.dataframe(df.head(20), use_container_width=True)

else:
    st.info("☝️ Upload a CSV file to begin")
    st.stop()

# Step 3: Configure inference
st.markdown("---")
st.subheader("3️⃣ Configure Inference")

col1, col2, col3 = st.columns(3)

with col1:
    id_col = st.selectbox(
        "ID column (optional)",
        options=[None] + list(df.columns),
        help="Column with sample identifiers",
    )

with col2:
    label_col = st.selectbox(
        "Label column (optional)",
        options=[None] + list(df.columns),
        help="Ground-truth labels for evaluation",
    )

with col3:
    strict_mode = st.checkbox(
        "Strict feature mode",
        value=True,
        help="If enabled, fail on missing features. If disabled, fill with zeros/median.",
    )

# Advanced options
with st.expander("⚙️ Advanced Options"):
    adv_col1, adv_col2 = st.columns(2)

    with adv_col1:
        fill_strategy = st.selectbox(
            "Fill strategy (lenient mode)",
            options=["zero", "median"],
            help="How to fill missing features when strict mode is disabled",
        )

        max_roc_curves = st.number_input(
            "Max ROC curves to plot",
            min_value=1,
            max_value=50,
            value=10,
            help="Maximum number of per-class ROC curves",
        )

    with adv_col2:
        include_plots = st.checkbox("Generate plots", value=True)
        include_excel = st.checkbox("Generate Excel workbook", value=True)

        device = st.selectbox(
            "Device (PyTorch models)",
            options=["auto", "cpu", "cuda", "mps"],
            help="Computation device for hierarchical models",
        )

# Step 4: Run inference
st.markdown("---")
st.subheader("4️⃣ Run Inference")

output_dir = Path("inference_output") / run_dir.name
output_dir.mkdir(parents=True, exist_ok=True)

if st.button("🚀 Run Inference", type="primary", use_container_width=True):
    try:
        # Create config
        config = InferenceConfig(
            run_dir=run_dir,
            data_csv=inference_csv,
            output_dir=output_dir,
            id_col=id_col,
            label_col=label_col,
            strict_features=strict_mode,
            lenient_fill_strategy=fill_strategy,
            max_roc_curves=max_roc_curves,
            include_plots=include_plots,
            include_excel=include_excel,
            device=device,
            batch_size=512,
            verbose=1,
        )

        # Run inference
        with st.spinner("Running inference... This may take a few moments."):
            results = run_inference(config)

        st.success("✅ Inference completed successfully!")

        # Display results
        st.markdown("---")
        st.subheader("📊 Results")

        # Predictions preview
        predictions = results["predictions"]
        st.markdown(f"**Samples processed:** {len(predictions)}")

        tab1, tab2, tab3 = st.tabs(["📋 Predictions", "📈 Metrics", "📁 Files"])

        with tab1:
            st.markdown("**Prediction Preview (first 50 rows):**")
            st.dataframe(predictions.head(50), use_container_width=True)

            # Download button
            st.download_button(
                label="⬇️ Download Full Predictions (CSV)",
                data=predictions.to_csv(index=False).encode("utf-8"),
                file_name=f"predictions_{run_dir.name}.csv",
                mime="text/csv",
                use_container_width=True,
            )

        with tab2:
            if "metrics" in results and results["metrics"]:
                metrics = results["metrics"]

                # Overall metrics
                if "overall" in metrics:
                    st.markdown("**Overall Performance:**")

                    metric_cols = st.columns(4)
                    overall = metrics["overall"]

                    with metric_cols[0]:
                        st.metric("Accuracy", f"{overall.get('accuracy', 0):.3f}")

                    with metric_cols[1]:
                        st.metric("Balanced Accuracy", f"{overall.get('balanced_accuracy', 0):.3f}")

                    with metric_cols[2]:
                        st.metric("F1 (Macro)", f"{overall.get('f1_macro', 0):.3f}")

                    with metric_cols[3]:
                        if "roc_auc" in overall and "macro" in overall["roc_auc"]:
                            st.metric("ROC AUC (Macro)", f"{overall['roc_auc']['macro']:.3f}")
                        else:
                            st.metric("ROC AUC", "N/A")

                    # Per-class metrics
                    if "per_class" in overall:
                        st.markdown("**Per-Class Metrics:**")
                        per_class_df = pd.DataFrame(overall["per_class"])
                        st.dataframe(per_class_df, use_container_width=True)

                    # Confusion matrix
                    if "confusion_matrix" in overall:
                        st.markdown("**Confusion Matrix:**")
                        cm = overall["confusion_matrix"]
                        cm_df = pd.DataFrame(
                            cm["matrix"],
                            index=cm["labels"],
                            columns=cm["labels"],
                        )
                        st.dataframe(cm_df, use_container_width=True)
            else:
                st.info("No metrics computed (ground-truth labels not provided)")

        with tab3:
            st.markdown("**Generated Files:**")

            output_files = results.get("output_files", {})

            for name, path in output_files.items():
                st.markdown(f"- **{name}**: `{path}`")

                # Download button for key files
                if path.exists():
                    if path.suffix == ".csv":
                        with open(path, "rb") as f:
                            st.download_button(
                                label=f"⬇️ Download {path.name}",
                                data=f.read(),
                                file_name=path.name,
                                mime="text/csv",
                            )
                    elif path.suffix in [".xlsx", ".xls"]:
                        with open(path, "rb") as f:
                            st.download_button(
                                label=f"⬇️ Download {path.name}",
                                data=f.read(),
                                file_name=path.name,
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            )
                    elif path.suffix in [".png", ".jpg", ".jpeg"]:
                        st.image(str(path), width=600)

        # Warnings
        if results.get("warnings"):
            with st.expander(f"⚠️ Warnings ({len(results['warnings'])})"):
                for i, warning in enumerate(results["warnings"], 1):
                    st.warning(f"{i}. {warning}")

    except Exception as e:
        st.error(f"❌ Inference failed: {e}")
        with st.expander("🐛 Error details"):
            import traceback
            st.code(traceback.format_exc())

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray; font-size: 0.9em;'>
    <p><strong>Inference Pipeline</strong> | Part of the MLSubtype Package</p>
    <p>For CLI usage: <code>classiflow infer --help</code></p>
    </div>
    """,
    unsafe_allow_html=True,
)
