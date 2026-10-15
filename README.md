# Data Modeling Studio

A local web application for reusable tabular prediction, built for Artificial Intelligence and Soft Computing. Upload your own CSV/Excel data, choose a target, train a NumPy neural network, and use a genetic algorithm to search feature subsets and network settings.

## Start on Windows

The ready-to-run `.runtime` environment in this folder uses your existing Anaconda packages (NumPy 2.3.5, pandas 2.3.3, scikit-learn 1.7.2, Streamlit 1.51.0). Run:

```powershell
.\run.ps1
```

Then open http://localhost:8501. Alternatively:

```powershell
.\.runtime\Scripts\python.exe -m streamlit run app.py
```

For a fully isolated setup on another computer, install the dependencies with uv. An empty `.venv` was created during setup here; the app currently uses `.runtime` because package downloads were slow:

```powershell
uv venv --python 3.13 .venv
uv pip install --python .venv/Scripts/python.exe -r requirements.txt
```

If uv is unavailable, use Python 3.11+:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Workflow

The frontend uses the original blue theme, sidebar upload, and four-stage tab workspace. Top spacing keeps headings below Streamlit's fixed toolbar. Predictions remain available across page reruns until the input rows or training run change. An input template can be downloaded from the Predict tab.

1. Choose a bundled Iris classification or Diabetes regression demo, or upload CSV/XLSX. Excel reads the first worksheet. CSV separator is configurable.
2. Inspect the data and choose a target and inputs. Confirm classification vs regression.
3. Exclude IDs, raw dates, long text, target-derived columns, and fields unavailable at prediction time.
4. Use a random stratified split for ordinary classification, or a date column for chronological splitting. Raw dates must be excluded from model inputs.
5. Train; review test metrics, learning curves, and candidate history.
6. Enter new rows or upload a prediction file, then download the output. Download the report and model if needed.

## Algorithm implementation

`engine.py` implements a one-hidden-layer ReLU neural network from scratch: forward propagation, softmax/cross-entropy for classification or linear output/MSE for regression, explicit backpropagation, mini-batch Adam, gradient clipping, and early stopping. Regression targets are standardized using training statistics.

Genetic search uses a chromosome with a raw-column feature mask, hidden-unit count, and learning rate. It applies tournament selection, uniform crossover, mutation, and elitism. Candidates run sequentially. Validation macro F1 is maximized for classification; negative standardized RMSE is maximized for regression. Identical chromosomes reuse cached fitness. If none beats the initial network, the initial network is retained. No improvement is promised.

Preprocessing is learned only from training rows: numerical median imputation and standardization; categorical encoding with the 30 most common values plus an OTHER bucket per column. The same mapping is saved for inference. Numerical standardized values are clipped to [-20, 20]. Numerical infinity in training is treated as missing; invalid non-numeric inference values are rejected.

Splits are approximately 60/20/20. Equal timestamps stay together in chronological mode. The selected model restores its best validation checkpoint. The test partition does not influence candidate scoring. This version keeps the selected checkpoint rather than retraining on training+validation rows.

## Scope and resource limits

- CSV/XLSX tables only; numerical and categorical inputs. No raw images, audio, text embeddings, multi-target prediction, or group-aware splits.
- 20 MB upload limit, 12,000 rows, 50 inputs, 512 encoded features, 2–20 classes, at least 50 labeled rows. Random classification requires at least five examples per class; class coverage is checked for chronological splits.
- At most 150 epochs/candidate, 10 candidates/generation, and 5 generations. Mini-batches of 128 and one BLAS thread reduce CPU/memory pressure.
- File and feature limits bound typical training cost but are not a guarantee of a fixed RAM ceiling, especially for compressed Excel parsing. Sample large datasets outside this app.
- The app runs locally and sends no datasets to a cloud service. Sessions are in memory; restarting clears them. A downloaded model preserves the trained state.
- Results may be poor on small, noisy, shifted, or unsuitable data. Duplicate records, related entities, and target leakage require user review. Repeated use of test results for tuning compromises the holdout.

## Reuse a model in Python

Only deserialize a trusted model you exported yourself. Pickle can execute code; this app intentionally has no model import UI.

```python
import pickle
import pandas as pd
from engine import ModelBundle  # engine.py must be importable

with open("model.pkl", "rb") as file:
    bundle = pickle.load(file)
new_rows = pd.read_csv("new_rows.csv")
new_rows["prediction"] = bundle.predict(new_rows)
new_rows.to_csv("predictions.csv", index=False)
```

## Tests

```powershell
.\.runtime\Scripts\python.exe -m unittest discover -s tests -v
```

Tests cover learning on classification/regression, leakage-resistant preprocessing and splits, genetic selection, model export round-trips, input validation, and the Streamlit demo workflows.

## Public data and reference documentation

- [Iris](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_iris.html) and [Diabetes](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_diabetes.html), bundled demo loaders.
- [UCI Bike Sharing](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset): regression. Exclude casual and registered counts when predicting total count.
- [UCI Dry Bean](https://archive.ics.uci.edu/dataset/602/dry+bean+dataset): classification on extracted numerical features.
- [Streamlit file upload](https://docs.streamlit.io/develop/api-reference/widgets/st.file_uploader).

For your viva, explain why the training/validation/test sets serve different purposes, how backpropagation changes weights, and how chromosome selection/crossover/mutation changes configurations. The neural network and GA are implemented here, rather than hidden behind a pretrained service.
