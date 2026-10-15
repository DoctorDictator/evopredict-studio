from __future__ import annotations

import hashlib
import json
import pickle
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.datasets import load_iris, load_diabetes
from engine import read_table, train_experiment, Settings, MAX_ROWS

st.set_page_config(page_title="Data Modeling Studio", page_icon="🧬", layout="wide")
st.markdown("""<style>
.block-container {max-width: 1200px; padding-top: 5rem;}
[data-testid="stSidebarUserContent"] {padding-top: 3rem;}
h1 {letter-spacing: -1.5px;}
h1, h2, h3 {scroll-margin-top: 5rem;}
[data-testid="stMetric"] {background: white; padding: 18px; border: 1px solid #e2e8f0; border-radius: 12px;}
.eyebrow {color: #2563eb; font-weight: 700; letter-spacing: 2px; font-size: 12px;}
@media (max-width: 640px) {.block-container {padding-top: 5rem;}}
</style>""", unsafe_allow_html=True)
st.markdown('<p class="eyebrow">AISC · NEURAL NETWORKS + GENETIC ALGORITHMS</p>', unsafe_allow_html=True)
st.title("Data Modeling Studio")
st.write("Turn your table into a prediction model. Explore the data, train a neural network, and watch genetic search tune it.")

with st.sidebar:
    st.header("Your workspace")
    source = st.radio("Dataset", ["Upload a file", "Iris · classification demo", "Diabetes · regression demo"])
    separator = st.selectbox("CSV separator", ["Auto", "Comma", "Semicolon", "Tab"], help="Ignored for Excel files.")
    uploaded = st.file_uploader("CSV or Excel dataset", type=["csv", "xlsx"], key="training_upload") if source == "Upload a file" else None
    st.caption("Local CPU training · up to 12,000 rows · 50 input columns · 20 MB files")
    st.divider()
    st.write("**How it works**")
    st.caption("Training learns weights. Validation guides early stopping and genetic search. The test partition measures the selected model only after search finishes.")
    st.caption("The demos are public datasets bundled with scikit-learn. The diabetes demo is an educational regression benchmark.")

if source == "Upload a file" and uploaded is None:
    st.info("Upload a CSV or Excel file in the sidebar, or choose a demo to try the full workflow.")
    st.markdown("""
**One system, different prediction problems**

Choose a numerical target to predict a value, or a categorical target to predict a class. Each dataset gets its own trained model.

**Supported inputs:** numerical and categorical table columns. Exclude identifiers, raw dates, free text, and information unavailable at prediction time. Images, audio, and document text need a separate workflow.
""")
    st.stop()

try:
    if source == "Upload a file":
        raw_bytes = uploaded.getvalue()
        frame = read_table(raw_bytes, uploaded.name, separator)
        dataset_name = uploaded.name
        fingerprint = hashlib.sha256(raw_bytes + separator.encode()).hexdigest()
    else:
        dataset = load_iris(as_frame=True) if source.startswith("Iris") else load_diabetes(as_frame=True)
        frame = dataset.frame.copy()
        if source.startswith("Iris"):
            frame["target"] = frame["target"].map(dict(enumerate(dataset.target_names)))
        dataset_name = source
        fingerprint = source
except Exception as exc:
    st.error(f"Could not read the dataset: {exc}")
    st.stop()

if st.session_state.get("dataset_fingerprint") != fingerprint:
    st.session_state["dataset_fingerprint"] = fingerprint
    st.session_state.pop("experiment", None)
    st.session_state.pop("experiment_signature", None)
    st.session_state.pop("prediction_output", None)

cols = st.columns(4)
cols[0].metric("Rows", f"{len(frame):,}")
cols[1].metric("Columns", len(frame.columns))
cols[2].metric("Missing cells", f"{int(frame.isna().sum().sum()):,}")
cols[3].metric("Data memory", f"{frame.memory_usage(deep=True).sum() / 1024 ** 2:.2f} MB")

explore, train_tab, results_tab, predict_tab = st.tabs(["01  Explore", "02  Train & optimize", "03  Results", "04  Predict"])

with explore:
    st.subheader(dataset_name)
    st.dataframe(frame.head(100), width="stretch")
    st.caption("Preview shows the first 100 rows. Duplicate rows are reported but not automatically removed.")
    profile = pd.DataFrame({"Type": frame.dtypes.astype(str), "Missing": frame.isna().sum(), "Distinct values": frame.nunique()})
    st.dataframe(profile, width="stretch")
    st.write(f"**Duplicate rows:** {int(frame.duplicated().sum()):,}")
    numeric = list(frame.select_dtypes(include="number").columns)
    if numeric:
        plot_col = st.selectbox("Inspect a numerical column", numeric)
        values = frame[plot_col].replace([np.inf, -np.inf], np.nan).dropna()
        if len(values):
            counts, bins = np.histogram(values, bins=20)
            st.bar_chart(pd.DataFrame({"Count": counts}, index=np.round((bins[:-1] + bins[1:]) / 2, 3)))
    st.download_button("Download this dataset", frame.to_csv(index=False).encode(), "dataset.csv", "text/csv")

with train_tab:
    st.subheader("Define your prediction problem")
    c1, c2 = st.columns(2)
    target = c1.selectbox("Column to predict", list(frame.columns), index=len(frame.columns) - 1, key=f"target_{fingerprint}")
    suggestion = "Regression" if pd.api.types.is_numeric_dtype(frame[target]) and frame[target].nunique() > 20 else "Classification"
    task = c2.selectbox("Task", ["Classification", "Regression"], index=int(suggestion == "Regression"), key=f"task_{fingerprint}_{target}")
    st.caption("Classification predicts a label. Regression predicts a number. Confirm the task yourself—numeric codes can represent categories.")
    candidates = [c for c in frame.columns if c != target]
    defaults = [c for c in candidates if pd.api.types.is_numeric_dtype(frame[c]) or frame[c].nunique() <= 30]
    features = st.multiselect("Input columns", candidates, default=defaults, key=f"features_{fingerprint}_{target}")
    excluded = set(candidates) - set(defaults)
    if excluded:
        st.caption("High-cardinality text excluded by default: " + ", ".join(sorted(excluded)))
    st.warning("Exclude IDs, target-derived columns, and anything you would not know when making a new prediction. For repeated people/devices, this version does not provide group-aware splits.")
    high_card = [c for c in features if not pd.api.types.is_numeric_dtype(frame[c]) and frame[c].nunique() > 30]
    if high_card:
        st.info("Categorical encoding keeps the 30 most common training values per column; remaining and unseen values use OTHER: " + ", ".join(high_card))
    split = st.radio("Data split", ["Random", "Chronological"], horizontal=True)
    time_column = st.selectbox("Date/time column used to sort rows", candidates) if split == "Chronological" else None
    if time_column in features and split == "Chronological":
        st.info("Remove the raw date/time column from input columns before training. Numeric calendar features can be added to your CSV separately.")
    st.caption("Random split: 60% training / 20% validation / 20% test. Chronological split keeps equal timestamps together, so row proportions can vary.")
    with st.expander("CPU budget and search settings", expanded=True):
        left, right = st.columns(2)
        epochs = left.slider("Maximum epochs per candidate", 10, 150, 50, 10)
        optimize = right.checkbox("Run genetic algorithm optimization", value=True)
        population = left.slider("Population size", 4, 10, 6)
        generations = right.slider("Generations", 1, 5, 3)
        seed = int(left.number_input("Random seed", min_value=0, max_value=100000, value=42, step=1))
        st.caption(f"Up to {1 + population * generations if optimize else 1} training runs; identical candidates are cached. Candidates run sequentially. Early stopping limits wasted work.")
        st.caption("Search space: input feature subsets, 16/32/64 hidden units, and learning rates 0.001/0.003/0.01. The initial network uses all selected inputs and 32 hidden units.")
    settings = Settings(target, features, task, split, time_column, epochs, population, generations, optimize, seed)
    signature = fingerprint + json.dumps(settings.__dict__, sort_keys=True)
    if st.button("Train my model", type="primary", disabled=not features, width="stretch"):
        progress = st.progress(0., text="Preparing the dataset")
        status = st.empty()
        def update(stage, fraction, message):
            progress.progress(float(np.clip(fraction, 0, 1)), text=stage)
            status.caption(message)
        try:
            st.session_state.pop("experiment", None)
            st.session_state.pop("prediction_output", None)
            result = train_experiment(frame, settings, update)
            st.session_state["experiment"] = result
            st.session_state["experiment_signature"] = signature
            st.success("Your model is ready. Open Results to inspect it or Predict to use it.")
        except Exception as exc:
            progress.empty()
            st.error(f"Training could not finish: {exc}")

result = st.session_state.get("experiment")
if result is not None and signature != st.session_state.get("experiment_signature"):
    st.warning("Configuration changed. Results and predictions below belong to the previous completed run. Train again to apply your changes.")

with results_tab:
    if result is None:
        st.info("Train a model first. Test results, learning curves, and genetic search history will appear here.")
    else:
        report = result["report"]
        st.subheader("Performance on untouched test rows")
        st.caption(f"Target: {report['settings']['target']} · task: {report['settings']['task']} · completed in {report['seconds']} seconds")
        st.dataframe(result["comparison"], hide_index=True, width="stretch")
        st.caption("Test metrics are for final evaluation. Repeatedly changing settings after viewing test results can overfit this test partition.")
        c1, c2, c3 = st.columns(3)
        for col, (name, count) in zip([c1, c2, c3], report["split_rows"].items()):
            col.metric(name.title() + " rows", count)
        best = report["selected_configuration"]
        st.write(f"**Selected network:** {best['hidden_units']} hidden units · learning rate {best['learning_rate']} · {len(best['features'])} input columns")
        st.write("**Selected inputs:** " + ", ".join(best["features"]))
        st.caption(f"Encoded columns: {report['encoded_features']} · estimated full encoded matrix: {report['estimated_encoded_memory_mb']} MB. This estimate excludes parsing, temporary arrays, and model overhead.")
        if report["settings"]["optimize"]:
            improvement = report["selected_validation_fitness"] - report["initial_validation_fitness"]
            st.write(f"**Validation fitness change:** {improvement:+.4f}. " + ("The initial network was retained because no candidate scored better." if improvement <= 0 else "Genetic search selected a better validation candidate."))
        left, right = st.columns(2)
        with left:
            st.write("**Selected model learning curve**")
            st.line_chart(result["selected_history"].set_index("epoch"))
        with right:
            if not result["search"].empty:
                st.write("**Genetic search progress**")
                st.caption(report["validation_fitness_definition"])
                st.line_chart(result["search"][["validation_fitness", "best_fitness"]].reset_index(drop=True))
            else:
                st.write("**Initial model learning curve**")
                st.line_chart(result["initial_history"].set_index("epoch"))
        if result["confusion"] is not None:
            labels = result["bundle"].classes
            st.write("**Confusion matrix: actual rows × predicted columns**")
            st.dataframe(pd.DataFrame(result["confusion"], index=labels, columns=labels), width="stretch")
        else:
            st.write("**Actual vs predicted test values**")
            st.scatter_chart(result["predictions"], x="actual", y="prediction")
        if not result["search"].empty:
            with st.expander("Inspect every genetic candidate"):
                st.dataframe(result["search"], hide_index=True, width="stretch")
        c1, c2, c3 = st.columns(3)
        c1.download_button("Download report", json.dumps(report, indent=2).encode(), "model_report.json", "application/json")
        c2.download_button("Download test predictions", result["predictions"].to_csv(index=False).encode(), "test_predictions.csv", "text/csv")
        c3.download_button("Download trained model", pickle.dumps(result["bundle"]), "model.pkl", "application/octet-stream")
        st.caption("The model export includes preprocessing and the selected network. Use it with this project's engine.py and matching dependencies. Only load pickle files you trust; the app does not accept model uploads.")

with predict_tab:
    if result is None:
        st.info("Train a model first to predict new rows.")
    else:
        st.subheader("Predict with your selected model")
        required = result["bundle"].preprocessor.columns
        st.caption("Provide all input columns from this training run. Missing values are imputed; unknown categories use OTHER.")
        st.download_button("Download an input template", frame[required].head(0).to_csv(index=False).encode(), "input_template.csv", "text/csv")
        method = st.radio("New data", ["Enter rows", "Upload prediction file"], horizontal=True)
        prediction_frame = None
        if method == "Enter rows":
            example = frame[required].head(1).copy().reset_index(drop=True)
            prediction_frame = st.data_editor(example, num_rows="dynamic", width="stretch", key="prediction_editor_" + st.session_state["experiment_signature"])
        else:
            prediction_upload = st.file_uploader("New CSV or Excel file", type=["csv", "xlsx"], key="prediction_upload")
            pred_sep = st.selectbox("Prediction CSV separator", ["Auto", "Comma", "Semicolon", "Tab"])
            if prediction_upload:
                try:
                    # Single-input prediction files are valid too.
                    from io import BytesIO
                    data = prediction_upload.getvalue()
                    if len(data) > 20 * 1024 ** 2:
                        raise ValueError("Prediction files must be 20 MB or smaller.")
                    if prediction_upload.name.lower().endswith(".xlsx"):
                        prediction_frame = pd.read_excel(BytesIO(data), nrows=MAX_ROWS + 1)
                    else:
                        sep = None if pred_sep == "Auto" else {"Comma": ",", "Semicolon": ";", "Tab": "\t"}[pred_sep]
                        prediction_frame = pd.read_csv(BytesIO(data), sep=sep, engine="python", nrows=MAX_ROWS + 1)
                    prediction_frame.columns = prediction_frame.columns.map(str)
                    if len(prediction_frame) > MAX_ROWS or prediction_frame.columns.duplicated().any():
                        raise ValueError("Use at most 12,000 prediction rows and unique column names.")
                    st.dataframe(prediction_frame.head(20), width="stretch")
                except Exception as exc:
                    st.error(f"Could not read predictions: {exc}")
                    prediction_frame = None
        prediction_signature = None
        if prediction_frame is not None:
            prediction_signature = hashlib.sha256(prediction_frame.to_csv(index=False).encode()).hexdigest() + st.session_state["experiment_signature"]
        if st.button("Generate predictions", type="primary", disabled=prediction_frame is None):
            try:
                if prediction_frame.empty:
                    raise ValueError("Add at least one row.")
                output = prediction_frame.copy()
                label = "prediction"
                while label in output.columns:
                    label = "model_" + label
                output[label] = result["bundle"].predict(prediction_frame)
                st.session_state["prediction_output"] = {"signature": prediction_signature, "data": output}
            except Exception as exc:
                st.error(f"Prediction failed: {exc}")
        saved_predictions = st.session_state.get("prediction_output")
        if saved_predictions and saved_predictions["signature"] == prediction_signature:
            output = saved_predictions["data"]
            st.success(f"Predictions ready for {len(output):,} rows.")
            st.dataframe(output, width="stretch")
            st.download_button("Download predictions", output.to_csv(index=False).encode(), "predictions.csv", "text/csv")
