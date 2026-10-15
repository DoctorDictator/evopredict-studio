"""CPU-sized tabular learning: explicit backpropagation and genetic search."""
from __future__ import annotations

from dataclasses import dataclass, asdict
from io import BytesIO
import time
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from threadpoolctl import threadpool_limits

MAX_ROWS = 12000
MAX_COLUMNS = 50
MAX_ENCODED = 512
MAX_BYTES = 20 * 1024 * 1024


def read_table(data: bytes, name: str, separator: str = "Auto") -> pd.DataFrame:
    if len(data) > MAX_BYTES:
        raise ValueError("Files must be 20 MB or smaller.")
    stream = BytesIO(data)
    if name.lower().endswith(".xlsx"):
        frame = pd.read_excel(stream, nrows=MAX_ROWS + 1)
    else:
        sep = None if separator == "Auto" else {"Comma": ",", "Semicolon": ";", "Tab": "\t"}[separator]
        frame = pd.read_csv(stream, sep=sep, engine="python", nrows=MAX_ROWS + 1)
    frame.columns = frame.columns.map(str)
    if frame.empty or len(frame.columns) < 2:
        raise ValueError("Use a table with at least two columns and some rows.")
    if frame.columns.duplicated().any():
        raise ValueError("Column names must be unique.")
    if len(frame) > MAX_ROWS:
        raise ValueError(f"This local version supports up to {MAX_ROWS:,} rows. Upload a smaller sample.")
    if len(frame.columns) > MAX_COLUMNS + 1:
        raise ValueError(f"Use no more than {MAX_COLUMNS} input columns plus the target.")
    return frame


class Preprocessor:
    """Fit only on training rows. Categories are bounded; unseen values share OTHER."""
    def fit(self, frame):
        self.columns = list(frame.columns)
        self.specs, self.groups, self.names = {}, {}, []
        offset = 0
        for col in self.columns:
            series = frame[col]
            if pd.api.types.is_numeric_dtype(series):
                values = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
                median = float(values.median()) if values.notna().any() else 0.0
                filled = values.fillna(median)
                mean, std = float(filled.mean()), float(filled.std(ddof=0))
                self.specs[col] = ("numeric", median, mean, max(std, 1e-6))
                width = 1
                self.names.append(col)
            else:
                values = series.astype("string").fillna("<MISSING>")
                categories = values.value_counts().head(30).index.tolist()
                self.specs[col] = ("category", categories)
                width = len(categories) + 1
                self.names.extend([f"{col}={v}" for v in categories] + [f"{col}=<OTHER>"])
            self.groups[col] = np.arange(offset, offset + width)
            offset += width
        if offset > MAX_ENCODED:
            raise ValueError(f"Encoding needs {offset} features; limit is {MAX_ENCODED}. Select fewer categorical columns.")
        return self

    def transform(self, frame):
        missing = set(self.columns) - set(frame.columns)
        if missing:
            raise ValueError("Missing input columns: " + ", ".join(sorted(missing)))
        out = np.zeros((len(frame), len(self.names)), dtype=np.float32)
        for col, spec in self.specs.items():
            ix = self.groups[col]
            if spec[0] == "numeric":
                values = pd.to_numeric(frame[col], errors="coerce")
                invalid = frame[col].notna() & values.isna()
                if invalid.any():
                    raise ValueError(f"Column '{col}' contains invalid numeric values.")
                values = values.replace([np.inf, -np.inf], np.nan)
                out[:, ix[0]] = np.clip((values.fillna(spec[1]).to_numpy() - spec[2]) / spec[3], -20, 20)
            else:
                mapping = {v: i for i, v in enumerate(spec[1])}
                values = frame[col].astype("string").fillna("<MISSING>")
                codes = values.map(mapping).fillna(len(mapping)).to_numpy(dtype=int)
                out[np.arange(len(frame)), ix[codes]] = 1
        return out


class NeuralNetwork:
    """One ReLU hidden layer, softmax/MSE output, mini-batch Adam and early stopping."""
    def __init__(self, inputs, outputs, hidden=32, learning_rate=0.003, task="Classification", seed=42):
        self.task, self.learning_rate = task, learning_rate
        self.rng = np.random.default_rng(seed)
        self.params = [
            (self.rng.normal(size=(inputs, hidden)) * np.sqrt(2 / inputs)).astype(np.float32),
            np.zeros(hidden, dtype=np.float32),
            (self.rng.normal(size=(hidden, outputs)) * np.sqrt(1 / hidden)).astype(np.float32),
            np.zeros(outputs, dtype=np.float32),
        ]

    def forward(self, x):
        w1, b1, w2, b2 = self.params
        hidden = np.maximum(x @ w1 + b1, 0)
        out = hidden @ w2 + b2
        if self.task == "Classification":
            exp = np.exp(out - out.max(axis=1, keepdims=True))
            out = exp / exp.sum(axis=1, keepdims=True)
        return hidden, out

    def loss(self, x, y):
        _, out = self.forward(x)
        if self.task == "Classification":
            return float(-np.log(np.clip(out[np.arange(len(y)), y], 1e-8, 1)).mean())
        return float(np.mean((out[:, 0] - y) ** 2))

    def fit(self, x, y, xv, yv, epochs=60, callback=None):
        first = [np.zeros_like(p) for p in self.params]
        second = [np.zeros_like(p) for p in self.params]
        best, stale, step = float("inf"), 0, 0
        saved = [p.copy() for p in self.params]
        self.history = []
        with threadpool_limits(limits=1):
            for epoch in range(epochs):
                order = self.rng.permutation(len(x))
                for start in range(0, len(x), 128):
                    indices = order[start:start + 128]
                    xb, yb = x[indices], y[indices]
                    hidden, output = self.forward(xb)
                    if self.task == "Classification":
                        delta = output.copy()
                        delta[np.arange(len(yb)), yb] -= 1
                        delta /= len(yb)
                    else:
                        delta = 2 * (output - yb[:, None]) / len(yb)
                    d_hidden = (delta @ self.params[2].T) * (hidden > 0)
                    gradients = [xb.T @ d_hidden, d_hidden.sum(axis=0), hidden.T @ delta, delta.sum(axis=0)]
                    step += 1
                    for i, grad in enumerate(gradients):
                        grad = np.clip(grad, -5, 5)
                        first[i] = .9 * first[i] + .1 * grad
                        second[i] = .999 * second[i] + .001 * grad * grad
                        self.params[i] -= self.learning_rate * (first[i] / (1 - .9 ** step)) / (np.sqrt(second[i] / (1 - .999 ** step)) + 1e-8)
                train_loss, val_loss = self.loss(x, y), self.loss(xv, yv)
                if not np.isfinite(train_loss + val_loss):
                    raise ValueError("Training became unstable. Try a smaller learning rate or check the data.")
                self.history.append({"epoch": epoch + 1, "training_loss": train_loss, "validation_loss": val_loss})
                if val_loss < best - 1e-5:
                    best, stale, saved = val_loss, 0, [p.copy() for p in self.params]
                else:
                    stale += 1
                if callback:
                    callback(epoch + 1, epochs, val_loss)
                if stale >= 10:
                    break
        self.params = saved
        return self

    def predict(self, x):
        with threadpool_limits(limits=1):
            out = self.forward(x)[1]
        return out.argmax(axis=1) if self.task == "Classification" else out[:, 0]


@dataclass
class Settings:
    target: str
    features: list[str]
    task: str = "Classification"
    split: str = "Random"
    time_column: str | None = None
    epochs: int = 60
    population: int = 6
    generations: int = 3
    optimize: bool = True
    seed: int = 42


def metrics(task, actual, predicted):
    if task == "Classification":
        return {"Accuracy": float(accuracy_score(actual, predicted)), "Macro F1": float(f1_score(actual, predicted, average="macro", zero_division=0))}
    return {"MAE": float(mean_absolute_error(actual, predicted)), "RMSE": float(np.sqrt(mean_squared_error(actual, predicted))), "R²": float(r2_score(actual, predicted))}


def fitness(task, actual, predicted):
    return metrics(task, actual, predicted)["Macro F1"] if task == "Classification" else -float(np.sqrt(mean_squared_error(actual, predicted)))


def split_data(frame, settings):
    frame = frame.loc[frame[settings.target].notna()].copy()
    if len(frame) < 50:
        raise ValueError("At least 50 rows with a non-missing target are required.")
    if settings.split == "Chronological":
        if not settings.time_column:
            raise ValueError("Choose a date/time column for chronological splitting.")
        dates = pd.to_datetime(frame[settings.time_column], errors="coerce", utc=True)
        if dates.isna().any():
            raise ValueError("The chosen time column contains missing or invalid dates.")
        frame = frame.assign(_split_date=dates).sort_values("_split_date", kind="stable").drop(columns="_split_date")
        # Keep equal timestamps in the same partition.
        ordered = pd.to_datetime(frame[settings.time_column], utc=True)
        unique = ordered.drop_duplicates().to_numpy()
        if len(unique) < 5:
            raise ValueError("Chronological splitting needs at least five distinct timestamps.")
        cut1, cut2 = unique[int(len(unique) * .6)], unique[int(len(unique) * .8)]
        train, val, test = frame[ordered < cut1], frame[(ordered >= cut1) & (ordered < cut2)], frame[ordered >= cut2]
        if min(map(len, (train, val, test))) < 5:
            raise ValueError("Timestamp groups leave a partition too small. Choose another split.")
        return train, val, test
    stratify = frame[settings.target].astype(str) if settings.task == "Classification" else None
    if stratify is not None and stratify.value_counts().min() < 5:
        raise ValueError("Each class needs at least five rows for a stratified train/validation/test split.")
    try:
        train, rest = train_test_split(frame, test_size=.4, random_state=settings.seed, stratify=stratify)
        val, test = train_test_split(rest, test_size=.5, random_state=settings.seed, stratify=rest[settings.target].astype(str) if stratify is not None else None)
    except ValueError as exc:
        raise ValueError("There are too few examples per class for this split. Add rows or combine rare classes.") from exc
    return train, val, test


class ModelBundle:
    def predict(self, frame):
        x = self.preprocessor.transform(frame)
        prediction = self.network.predict(x[:, self.indices])
        if self.task == "Classification":
            return self.classes[prediction]
        return prediction * self.y_std + self.y_mean


def train_experiment(frame, settings, callback=None):
    started = time.monotonic()
    if len(frame) > MAX_ROWS or len(settings.features) > MAX_COLUMNS:
        raise ValueError("Dataset exceeds local resource limits.")
    if not settings.features or settings.target in settings.features:
        raise ValueError("Select input columns and exclude the target.")
    if settings.split == "Chronological" and settings.time_column in settings.features:
        raise ValueError("Exclude the raw date column from inputs; engineer numeric date features separately.")
    if not 5 <= settings.epochs <= 150 or not 4 <= settings.population <= 10 or not 1 <= settings.generations <= 5:
        raise ValueError("Training settings exceed the supported CPU budget.")
    train, val, test = split_data(frame, settings)
    prep = Preprocessor().fit(train[settings.features])
    xt, xv = prep.transform(train), prep.transform(val)
    if settings.task == "Classification":
        classes = np.array(sorted(train[settings.target].astype(str).unique()))
        if not 2 <= len(classes) <= 20:
            raise ValueError("Classification supports 2–20 classes. For continuous numbers, choose regression.")
        mapping = {v: i for i, v in enumerate(classes)}
        def encode(part):
            encoded = part[settings.target].astype(str).map(mapping)
            if encoded.isna().any():
                raise ValueError("A validation/test class is absent from training. Use a random split or more historical examples.")
            return encoded.to_numpy(dtype=int)
        yt, yv, ytest = encode(train), encode(val), encode(test)
        mean, std, outputs = 0., 1., len(classes)
        baseline = np.full(len(test), np.bincount(yt).argmax())
    else:
        def encode(part):
            values = pd.to_numeric(part[settings.target], errors="coerce").to_numpy(dtype=np.float64)
            if not np.isfinite(values).all():
                raise ValueError("Regression target must contain finite numbers.")
            return values
        raw = encode(train)
        mean, std = float(raw.mean()), max(float(raw.std()), 1e-6)
        yt, yv, ytest = (raw - mean) / std, (encode(val) - mean) / std, encode(test)
        classes, outputs = None, 1
        baseline = np.full(len(test), mean)
    all_indices = np.arange(xt.shape[1])
    def train_candidate(mask, hidden, rate, notify=False):
        indices = np.concatenate([prep.groups[c] for c, keep in zip(settings.features, mask) if keep])
        network = NeuralNetwork(len(indices), outputs, hidden, rate, settings.task, settings.seed)
        network.fit(xt[:, indices], yt, xv[:, indices], yv, settings.epochs,
                    (lambda e, total, loss: callback("Initial network", e / total, f"Epoch {e}/{total} · validation loss {loss:.4f}")) if notify and callback else None)
        prediction = network.predict(xv[:, indices])
        return fitness(settings.task, yv, prediction), network, indices
    mask = np.ones(len(settings.features), dtype=bool)
    score, initial, _ = train_candidate(mask, 32, .003, True)
    selected, selected_indices = initial, all_indices
    best_config = {"features": settings.features.copy(), "hidden_units": 32, "learning_rate": .003}
    initial_score = score
    records, cache = [], {}
    rng = np.random.default_rng(settings.seed)
    best_score = score
    def random_candidate():
        candidate_mask = rng.random(len(mask)) > .25
        if not candidate_mask.any():
            candidate_mask[rng.integers(len(mask))] = True
        return (candidate_mask, int(rng.choice([16, 32, 64])), float(rng.choice([.001, .003, .01])))
    population = [(mask.copy(), 32, .003)] + [random_candidate() for _ in range(settings.population - 1)]
    cache[(tuple(mask), 32, .003)] = score
    if settings.optimize:
        for generation in range(settings.generations):
            ranked = []
            for number, (candidate_mask, hidden, rate) in enumerate(population):
                key = (tuple(candidate_mask), hidden, rate)
                if callback:
                    callback("Genetic search", (generation * settings.population + number) / (settings.population * settings.generations), f"Generation {generation + 1}/{settings.generations} · candidate {number + 1}/{settings.population}")
                if key in cache:
                    candidate_score = cache[key]
                else:
                    candidate_score, network, indices = train_candidate(candidate_mask, hidden, rate)
                    cache[key] = candidate_score
                    if candidate_score > best_score:
                        best_score, selected, selected_indices = candidate_score, network, indices
                        best_config = {"features": [c for c, keep in zip(settings.features, candidate_mask) if keep], "hidden_units": hidden, "learning_rate": rate}
                ranked.append((candidate_score, candidate_mask, hidden, rate))
                records.append({"generation": generation + 1, "candidate": number + 1, "validation_fitness": candidate_score, "best_fitness": best_score, "feature_count": int(candidate_mask.sum()), "hidden_units": hidden, "learning_rate": rate, "features": ", ".join(c for c, keep in zip(settings.features, candidate_mask) if keep)})
            ranked.sort(key=lambda item: item[0], reverse=True)
            population = [(ranked[0][1].copy(), ranked[0][2], ranked[0][3])]
            def tournament():
                picks = rng.integers(len(ranked), size=3)
                return max((ranked[i] for i in picks), key=lambda item: item[0])
            while len(population) < settings.population:
                a, b = tournament(), tournament()
                child = np.where(rng.random(len(mask)) < .5, a[1], b[1]).copy()
                child ^= rng.random(len(mask)) < .12
                if not child.any():
                    child[rng.integers(len(mask))] = True
                hidden = a[2] if rng.random() < .5 else b[2]
                rate = a[3] if rng.random() < .5 else b[3]
                if rng.random() < .2:
                    hidden = int(rng.choice([16, 32, 64]))
                if rng.random() < .2:
                    rate = float(rng.choice([.001, .003, .01]))
                population.append((child, hidden, rate))
    # Test data is transformed and evaluated only after selection has finished.
    x_test = prep.transform(test)
    initial_pred, selected_pred = initial.predict(x_test), selected.predict(x_test[:, selected_indices])
    if settings.task == "Regression":
        initial_pred, selected_pred = initial_pred * std + mean, selected_pred * std + mean
    comparisons = pd.DataFrame([{"Model": name, **metrics(settings.task, ytest, pred)} for name, pred in [("Constant baseline", baseline), ("Initial neural network", initial_pred), ("Selected neural network", selected_pred)]])
    bundle = ModelBundle()
    bundle.preprocessor, bundle.network, bundle.indices = prep, selected, selected_indices
    bundle.task, bundle.classes, bundle.y_mean, bundle.y_std = settings.task, classes, mean, std
    actual = classes[ytest] if classes is not None else ytest
    pred = classes[selected_pred] if classes is not None else selected_pred
    test_predictions = pd.DataFrame({"actual": actual, "prediction": pred})
    report = {"settings": asdict(settings), "split_rows": {"training": len(train), "validation": len(val), "test": len(test)}, "encoded_features": len(prep.names), "estimated_encoded_memory_mb": round(len(frame) * len(prep.names) * 4 / 1024 ** 2, 2), "selected_configuration": best_config, "initial_validation_fitness": initial_score, "selected_validation_fitness": best_score, "validation_fitness_definition": "Macro F1 (higher is better)" if classes is not None else "Negative RMSE in standardized target units (higher is better)", "test_metrics": comparisons.to_dict(orient="records"), "seconds": round(time.monotonic() - started, 2), "seed": settings.seed}
    matrix = confusion_matrix(ytest, selected_pred, labels=np.arange(len(classes))) if classes is not None else None
    if callback:
        callback("Complete", 1., "Training and evaluation complete")
    return {"bundle": bundle, "report": report, "comparison": comparisons, "initial_history": pd.DataFrame(initial.history), "selected_history": pd.DataFrame(selected.history), "search": pd.DataFrame(records), "predictions": test_predictions, "confusion": matrix}
