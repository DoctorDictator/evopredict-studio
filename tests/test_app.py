import unittest
from streamlit.testing.v1 import AppTest


class AppTests(unittest.TestCase):
    def test_empty_upload_and_demo_training(self):
        app = AppTest.from_file("app.py", default_timeout=90).run()
        self.assertFalse(app.exception)
        app.sidebar.radio[0].set_value("Iris · classification demo").run()
        self.assertFalse(app.exception)
        app.checkbox[0].uncheck().run()
        next(button for button in app.button if button.label == "Train my model").click().run()
        self.assertFalse(app.exception)
        self.assertFalse(app.error)
        self.assertIn("experiment", app.session_state)
        next(button for button in app.button if button.label == "Generate predictions").click().run()
        self.assertFalse(app.exception)
        self.assertFalse(app.error)
        app.run()
        self.assertIn("prediction_output", app.session_state)
        self.assertTrue(any("Predictions ready" in item.value for item in app.success))

    def test_regression_demo_with_ga(self):
        app = AppTest.from_file("app.py", default_timeout=120).run()
        app.sidebar.radio[0].set_value("Diabetes · regression demo").run()
        app.slider[0].set_value(10)
        app.slider[1].set_value(4)
        app.slider[2].set_value(1)
        app.run()
        next(button for button in app.button if button.label == "Train my model").click().run()
        self.assertFalse(app.exception)
        self.assertFalse(app.error)
        self.assertEqual(app.session_state["experiment"]["report"]["settings"]["task"], "Regression")


if __name__ == "__main__":
    unittest.main()
