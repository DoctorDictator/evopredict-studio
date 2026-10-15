import pickle
import unittest
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from engine import Preprocessor, NeuralNetwork, Settings, split_data, train_experiment, read_table


class EngineTests(unittest.TestCase):
    def test_preprocessing_uses_training_statistics_and_handles_unseen(self):
        prep = Preprocessor().fit(pd.DataFrame({"n": [1., 3., np.nan], "cat": ["a", "b", "a"]}))
        self.assertEqual(prep.specs["n"][1], 2.)
        before = pickle.dumps(prep)
        encoded = prep.transform(pd.DataFrame({"n": [1000., np.nan], "cat": ["new", None]}))
        self.assertTrue(np.isfinite(encoded).all())
        self.assertEqual(encoded[0, -1], 1)
        self.assertEqual(pickle.dumps(prep), before)
        with self.assertRaisesRegex(ValueError, "invalid numeric"):
            prep.transform(pd.DataFrame({"n": ["wrong"], "cat": ["a"]}))
        with self.assertRaisesRegex(ValueError, "Missing input"):
            prep.transform(pd.DataFrame({"cat": ["a"]}))

    def test_backprop_learns_regression(self):
        rng = np.random.default_rng(1)
        x = rng.normal(size=(500, 3)).astype(np.float32)
        y = 2 * x[:, 0] - x[:, 1] + .4
        model = NeuralNetwork(3, 1, task="Regression", seed=1)
        initial = model.loss(x[400:], y[400:])
        model.fit(x[:400], y[:400], x[400:], y[400:], epochs=80)
        self.assertLess(model.loss(x[400:], y[400:]), initial * .1)

    def test_classification_and_export(self):
        data = load_iris(as_frame=True).frame
        result = train_experiment(data, Settings("target", list(data.columns[:-1]), epochs=50, optimize=False))
        selected = result["comparison"].iloc[2]
        self.assertGreater(selected["Accuracy"], .75)
        bundle = pickle.loads(pickle.dumps(result["bundle"]))
        np.testing.assert_array_equal(bundle.predict(data.head()), result["bundle"].predict(data.head()))
        self.assertEqual(result["confusion"].sum(), result["report"]["split_rows"]["test"])

    def test_genetic_search_retains_best_validation_candidate(self):
        rng = np.random.default_rng(8)
        data = pd.DataFrame(rng.normal(size=(240, 3)), columns=["a", "b", "noise"])
        data["target"] = 3 * data.a - data.b
        result = train_experiment(data, Settings("target", ["a", "b", "noise"], "Regression", epochs=20, population=4, generations=2))
        report = result["report"]
        self.assertGreaterEqual(report["selected_validation_fitness"], report["initial_validation_fitness"])
        self.assertEqual(len(result["search"]), 8)
        self.assertTrue(np.isfinite(result["bundle"].predict(data)).all())
        self.assertGreater(result["comparison"].iloc[2]["R²"], .4)

    def test_holdout_cannot_change_selected_model(self):
        rng = np.random.default_rng(5)
        data = pd.DataFrame({"x": rng.normal(size=120), "target": rng.normal(size=120)})
        settings = Settings("target", ["x"], "Regression", epochs=10, population=4, generations=1)
        _, _, test = split_data(data, settings)
        modified = data.copy()
        modified.loc[test.index, "x"] = 999
        modified.loc[test.index, "target"] = -999
        a, b = train_experiment(data, settings), train_experiment(modified, settings)
        self.assertEqual(a["report"]["selected_configuration"], b["report"]["selected_configuration"])
        for first, second in zip(a["bundle"].network.params, b["bundle"].network.params):
            np.testing.assert_array_equal(first, second)

    def test_chronological_split_keeps_equal_dates_together(self):
        data = pd.DataFrame({"date": np.repeat(pd.date_range("2020-01-01", periods=10).astype(str), 10), "x": range(100), "target": range(100)})
        train, val, test = split_data(data, Settings("target", ["x"], "Regression", "Chronological", "date"))
        self.assertLess(train.date.max(), val.date.min())
        self.assertLess(val.date.max(), test.date.min())
        self.assertEqual(len(train) + len(val) + len(test), 100)

    def test_csv_and_invalid_target(self):
        data = read_table(b"x;y\n1;2\n3;4\n", "data.csv", "Semicolon")
        self.assertEqual(list(data.columns), ["x", "y"])
        frame = pd.DataFrame({"x": range(60), "target": ["bad"] * 60})
        with self.assertRaisesRegex(ValueError, "finite numbers"):
            train_experiment(frame, Settings("target", ["x"], "Regression", epochs=10, optimize=False))


if __name__ == "__main__":
    unittest.main()
