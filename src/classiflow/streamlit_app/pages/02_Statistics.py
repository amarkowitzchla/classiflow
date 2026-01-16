"""② Statistics - Statistical analysis and visualization page."""

from pathlib import Path
import streamlit as st
import pandas as pd

# Import the new stats API
from classiflow.stats import run_stats, run_visualizations

st.title("② Statistics")

# Initialize paths
root = Path.cwd()
DATA = root / "data"
DERIVED = root / "derived"
DATA.mkdir(parents=True, exist_ok=True)
DERIVED.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────────────────────────────────────
# File Upload
# ──────────────────────────────────────────────────────────────────────────────

up = st.file_uploader("Upload CSV for statistics", type=["csv"], key="stat_csv")
if up is None:
    st.info("Upload a CSV to continue.")
    st.stop()

# Save uploaded file
file_name = Path(up.name).name or "uploaded.csv"
csv_path = DATA / file_name
csv_path.write_bytes(up.getvalue())

# Load and preview data
try:
    df = pd.read_csv(csv_path)
except Exception as e:
    st.error(f"Error loading CSV: {e}")
    st.stop()

st.caption(f"Using file: {file_name}")
st.dataframe(df.head(20), use_container_width=True)

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

st.divider()
st.subheader("Configuration")

col1, col2 = st.columns(2)

with col1:
    label_col = st.selectbox(
        "Label column (target)",
        options=list(df.columns),
        index=0,  # first column as default
        key="stats_label_col",
    )

    alpha = st.number_input("Significance level (α)", min_value=0.001, max_value=0.5, value=0.05, step=0.01)

    min_n = st.number_input("Minimum n per class for Shapiro-Wilk", min_value=2, max_value=20, value=3, step=1)

with col2:
    dunn_adjust = st.selectbox(
        "P-value adjustment (Dunn test)",
        options=["holm", "bonferroni", "fdr_bh", "fdr_by", "sidak"],
        index=0,
    )

    top_n_features = st.number_input("Top N features in summary", min_value=10, max_value=100, value=30, step=10)

# Class distribution preview
with st.expander("Preview label distribution"):
    vc = df[label_col].value_counts(dropna=False).rename_axis(label_col).reset_index(name="count")
    st.dataframe(vc, use_container_width=True)

    # Optional class subset/order
    all_classes = df[label_col].dropna().unique().tolist()
    use_class_subset = st.checkbox("Restrict to specific classes?", value=False)

    if use_class_subset:
        selected_classes = st.multiselect(
            "Select classes (in desired order)",
            options=all_classes,
            default=all_classes,
        )
    else:
        selected_classes = None

# ──────────────────────────────────────────────────────────────────────────────
# Run Analysis
# ──────────────────────────────────────────────────────────────────────────────

st.divider()

col_btn1, col_btn2 = st.columns(2)

run_stats_btn = col_btn1.button("Run Statistical Tests", type="primary", use_container_width=True)
run_viz_btn = col_btn2.button("Generate Visualizations", use_container_width=True)

if run_stats_btn:
    with st.spinner("Running statistical analysis..."):
        try:
            results = run_stats(
                data_csv=csv_path,
                label_col=label_col,
                outdir=DERIVED,
                classes=selected_classes,
                alpha=alpha,
                min_n=min_n,
                dunn_adjust=dunn_adjust,
                top_n_features=top_n_features,
                write_legacy_csv=True,
                write_legacy_xlsx=True,
            )

            st.success("✓ Statistical analysis complete!")
            st.write(f"**Classes:** {len(results['classes'])} → {', '.join(results['classes'])}")
            st.write(f"**Features:** {results['n_features']}")
            st.write(f"**Samples:** {results['n_samples']}")

            # Store results in session state for visualization
            st.session_state["stats_results"] = results

        except Exception as e:
            st.error(f"Analysis failed: {e}")
            import traceback

            st.code(traceback.format_exc())

if run_viz_btn:
    stats_dir = DERIVED / "stats_results"

    if not stats_dir.exists():
        st.warning("Please run statistical tests first to generate stats results.")
    else:
        with st.spinner("Generating visualizations..."):
            try:
                viz_results = run_visualizations(
                    data_csv=csv_path,
                    label_col=label_col,
                    outdir=DERIVED,
                    stats_dir=stats_dir,
                    classes=selected_classes,
                    alpha=alpha,
                )

                st.success("✓ Visualizations complete!")
                st.write(f"**Output directory:** {viz_results['viz_dir']}")

            except Exception as e:
                st.error(f"Visualization failed: {e}")
                import traceback

                st.code(traceback.format_exc())

# ──────────────────────────────────────────────────────────────────────────────
# Display Results
# ──────────────────────────────────────────────────────────────────────────────

st.divider()

stats_dir = DERIVED / "stats_results"
if stats_dir.exists():
    st.subheader("Results & Downloads")

    # Publication workbook
    pub_xlsx = stats_dir / "publication_stats.xlsx"
    if pub_xlsx.exists():
        st.markdown("**📊 Publication-Ready Workbook**")
        st.download_button(
            "Download publication_stats.xlsx",
            data=pub_xlsx.read_bytes(),
            file_name="publication_stats.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

    # Legacy workbook
    legacy_xlsx = stats_dir / "stats_results.xlsx"
    if legacy_xlsx.exists():
        st.markdown("**📋 Legacy Workbook (backward compatible)**")
        st.download_button(
            "Download stats_results.xlsx",
            data=legacy_xlsx.read_bytes(),
            file_name="stats_results.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

    # Top features preview
    if "stats_results" in st.session_state:
        st.divider()
        st.subheader("Top Features Preview")

        top_feat = st.session_state["stats_results"]["top_features_overall"]
        st.dataframe(top_feat, use_container_width=True)

    # CSV downloads
    st.divider()
    st.subheader("Individual CSV Tables")

    csv_files = [
        "Normality_Summary.csv",
        "Normality_By_Class.csv",
        "Parametric_Overall.csv",
        "Parametric_PostHoc.csv",
        "Nonparametric_Overall.csv",
        "Nonparametric_PostHoc.csv",
    ]

    cols = st.columns(3)
    for i, name in enumerate(csv_files):
        p = stats_dir / name
        if p.exists():
            with cols[i % 3]:
                st.markdown(f"**{name}**")
                try:
                    preview = pd.read_csv(p)
                    with st.expander(f"Preview {name}"):
                        st.dataframe(preview.head(10), use_container_width=True)
                    st.download_button(
                        f"Download",
                        data=p.read_bytes(),
                        file_name=name,
                        mime="text/csv",
                        key=f"download_{name}",
                    )
                except Exception as e:
                    st.error(f"Error loading {name}: {e}")

else:
    st.info("Run the statistical tests to populate results.")
