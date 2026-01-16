"""Train Models page - refactored to use classiflow library."""

from pathlib import Path
import streamlit as st
import pandas as pd

from classiflow.config import TrainConfig, MetaConfig
from classiflow.training import train_binary_task, train_meta_classifier
from classiflow.streamlit_app.ui.helpers import list_outputs

st.title("① Train Models")

root = Path.cwd()
DATA = root / "data"
DERIVED = root / "derived"
DATA.mkdir(parents=True, exist_ok=True)
DERIVED.mkdir(parents=True, exist_ok=True)

st.subheader("Upload / select ML-ready CSV")
up = st.file_uploader("CSV with features + label column", type=["csv"], key="train_csv")

csv_path = None
df = None

if up is not None:
    df = pd.read_csv(up)
    uploaded_dir = DATA / "uploaded"
    uploaded_dir.mkdir(parents=True, exist_ok=True)
    safe_name = Path(up.name).name
    csv_path = uploaded_dir / safe_name
    up.seek(0)
    with open(csv_path, "wb") as f:
        f.write(up.read())
    st.session_state["train_csv_path"] = str(csv_path)
elif "train_csv_path" in st.session_state and Path(st.session_state["train_csv_path"]).exists():
    csv_path = Path(st.session_state["train_csv_path"])
    df = pd.read_csv(csv_path)
else:
    example = DATA / "iris_data.csv"
    if example.exists():
        csv_path = example
        df = pd.read_csv(csv_path)
    else:
        st.warning("Provide a CSV to continue (no upload detected and no example found).")
        st.stop()

st.caption(f"Using: {csv_path}")
st.dataframe(df.head(20), use_container_width=True)

# Label column choice
label_cols = [col for col in df.columns if df[col].dtype == object or df[col].nunique() <= 20]
label_col = st.selectbox("Label column", options=label_cols or list(df.columns), index=0)

# Determine problem type
unique_vals = pd.Series(df[label_col].dropna().astype(str).unique())
n_classes = unique_vals.nunique()
is_binary = n_classes == 2

# Binary-only UI: positive class override
pos_label = None
if is_binary:
    st.subheader("Binary options")
    uniq_labels = sorted(unique_vals.tolist())
    pos_label_opt = st.selectbox(
        "Positive class (optional; default = minority detected)",
        options=["<infer minority>"] + uniq_labels,
        index=0,
    )
    pos_label = None if pos_label_opt == "<infer minority>" else pos_label_opt

# Multiclass-only UI: tasks definition
tasks_json_path = None
if not is_binary:
    st.subheader("Tasks definition (multiclass)")
    col1, col2 = st.columns(2)
    use_tasks_json = col1.checkbox("Use tasks.json (optional)")
    if use_tasks_json:
        up_tasks = st.file_uploader("tasks.json", type=["json"], key="tasks_json")
        if up_tasks is not None:
            tasks_json_path = root / "data" / "tasks_uploaded.json"
            with open(tasks_json_path, "wb") as f:
                f.write(up_tasks.read())
        st.caption("If omitted, auto OvR + pairwise tasks will be used.")

# Common training options
st.subheader("Training options")
colA, colB, colC, colD = st.columns(4)
smote = colA.toggle("Use SMOTE", value=True)
folds = colB.number_input("CV folds", min_value=2, max_value=10, value=3, step=1)
seed = colC.number_input("Random seed", min_value=0, max_value=9999, value=42, step=1)

# Run
button_label = "Run training (binary)" if is_binary else "Run training (multiclass + meta)"
if st.button(button_label, type="primary"):
    if is_binary:
        with st.spinner("Training binary classifier (nested CV)…"):
            config = TrainConfig(
                data_csv=csv_path,
                label_col=label_col,
                pos_label=pos_label,
                outdir=DERIVED,
                outer_folds=int(folds),
                random_state=int(seed),
                smote_mode="on" if smote else "off",
            )
            try:
                train_binary_task(config)
                st.success("Binary training complete.")
            except Exception as e:
                st.error(f"Training failed: {e}")
    else:
        with st.spinner("Training multiclass tasks + meta-classifier (nested CV)…"):
            smote_mode = "both" if smote else "off"
            config = MetaConfig(
                data_csv=csv_path,
                label_col=label_col,
                tasks_json=tasks_json_path,
                outdir=DERIVED,
                outer_folds=int(folds),
                random_state=int(seed),
                smote_mode=smote_mode,
            )
            try:
                train_meta_classifier(config)
                st.success("Meta-classifier training complete.")
            except Exception as e:
                st.error(f"Training failed: {e}")

    st.divider()
    st.subheader("Outputs")
    outputs = list_outputs(DERIVED)
    if outputs:
        for group, files in outputs.items():
            with st.expander(group):
                for f in files[:10]:  # Show first 10
                    st.write(f)
