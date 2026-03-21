"""Train Models page - refactored to use classiflow library."""

from pathlib import Path
import streamlit as st
import pandas as pd

from classiflow.config import TrainConfig, MetaConfig, default_torch_num_workers
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
backend = colD.selectbox("Backend", options=["sklearn", "torch"], index=0)

device = "auto"
model_set = None
torch_dtype = "float32"
torch_num_workers = default_torch_num_workers()
require_torch_device = False
expanded_mlp_tuning_grid = False

if backend == "torch":
    st.caption("Torch backend enables MLP candidates and expanded MLP tuning.")
    colE, colF, colG, colH = st.columns(4)
    device = colE.selectbox("Device", options=["auto", "cpu", "cuda", "mps"], index=0)
    model_set = colF.selectbox("Torch model set", options=["torch_basic", "torch_fast"], index=0)
    torch_dtype = colG.selectbox("Torch dtype", options=["float32", "float16"], index=0)
    torch_num_workers = int(
        colH.number_input(
            "Torch workers",
            min_value=0,
            max_value=32,
            value=default_torch_num_workers(),
            step=1,
        )
    )
    colI, colJ = st.columns(2)
    require_torch_device = colI.checkbox("Require requested device", value=False)
    expanded_mlp_tuning_grid = colJ.checkbox(
        "Expanded MLP tuning grid",
        value=False,
        help="Include CCIX-style MLP hyperparameter axes and nearby values.",
    )
else:
    st.caption("Expanded MLP tuning grid applies only when using the torch backend.")

final_estimator_strategy = "single"
bagging_n_estimators = 10
bagging_max_samples = 1.0
bagging_max_features = 1.0
bagging_bootstrap = True
bagging_bootstrap_features = False

if is_binary:
    st.subheader("Final estimator")
    final_estimator_strategy = st.selectbox(
        "Final estimator strategy",
        options=["single", "bagged"],
        index=0,
        help="Bagged wraps the selected final estimator in a bootstrap ensemble.",
    )
    if final_estimator_strategy == "bagged":
        colK, colL, colM = st.columns(3)
        bagging_n_estimators = int(
            colK.number_input("Bagging estimators", min_value=1, max_value=200, value=10, step=1)
        )
        bagging_max_samples = float(
            colL.number_input(
                "Bagging max samples",
                min_value=0.05,
                max_value=1.0,
                value=1.0,
                step=0.05,
            )
        )
        bagging_max_features = float(
            colM.number_input(
                "Bagging max features",
                min_value=0.05,
                max_value=1.0,
                value=1.0,
                step=0.05,
            )
        )
        colN, colO = st.columns(2)
        bagging_bootstrap = colN.checkbox("Bootstrap rows", value=True)
        bagging_bootstrap_features = colO.checkbox("Bootstrap features", value=False)
else:
    st.info(
        "This page runs the meta workflow for non-binary labels. "
        "Bagging is not exposed here because meta final estimators remain single-model."
    )

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
                backend=backend,
                device=device,
                model_set=model_set,
                torch_num_workers=torch_num_workers,
                torch_dtype=torch_dtype,
                require_torch_device=require_torch_device,
                expanded_mlp_tuning_grid=expanded_mlp_tuning_grid,
                final_estimator_strategy=final_estimator_strategy,
                bagging_n_estimators=bagging_n_estimators,
                bagging_max_samples=bagging_max_samples,
                bagging_max_features=bagging_max_features,
                bagging_bootstrap=bagging_bootstrap,
                bagging_bootstrap_features=bagging_bootstrap_features,
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
                backend=backend,
                device=device,
                model_set=model_set,
                torch_num_workers=torch_num_workers,
                torch_dtype=torch_dtype,
                require_torch_device=require_torch_device,
                expanded_mlp_tuning_grid=expanded_mlp_tuning_grid,
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
