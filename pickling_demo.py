import argparse
import pickle
from pathlib import Path

import joblib
import numpy as np
from matplotlib.figure import Figure
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

from grade_tracker import GradeTracker


DEFAULT_OUTPUT = Path(__file__).resolve().parent / "artifacts"


def write_pickle(path, value):
    with path.open("wb") as file:
        pickle.dump(value, file, protocol=pickle.HIGHEST_PROTOCOL)


def read_pickle(path):
    with path.open("rb") as file:
        return pickle.load(file)


def save_plot(path, study_hours, exam_scores, model):
    line_hours = np.linspace(1, 25, 100).reshape(-1, 1)
    figure = Figure(figsize=(8.4, 5), layout="constrained")
    axes = figure.subplots()
    axes.scatter(study_hours.ravel(), exam_scores, color="#235b91", label="Observed scores")
    axes.plot(line_hours.ravel(), model.predict(line_hours), color="#ab3c2c", label="Fitted line")
    axes.set(
        title="Study hours and exam scores",
        xlabel="Study hours per week",
        ylabel="Exam score",
    )
    axes.grid(alpha=0.2)
    axes.set_axisbelow(True)
    axes.legend()
    figure.savefig(path, dpi=160, backend="agg")


def save_examples(directory):
    directory.mkdir(parents=True, exist_ok=True)
    fruits = ["apple", "banana", "mango", "strawberry"]
    scores = {"Alice": 92, "Bob": 85, "Charlie": 78, "Diana": 95}
    combined = {"fruits": fruits, "scores": scores, "metrics": [1.2, 3.4, 5.6, 7.8], "active": True}
    write_pickle(directory / "fruits.pkl", fruits)
    write_pickle(directory / "scores.pkl", scores)
    write_pickle(directory / "complex_data.pkl", combined)

    tracker = GradeTracker("Introduction to Machine Learning")
    for student, score in {"Alice": 95, "Bob": 87, "Charlie": 91, "Diana": 84, "Eve": 78}.items():
        tracker.record(student, score)
    write_pickle(directory / "ml_course_grades.pkl", tracker)
    print(f"Saved {len(tracker.grades)} sample grades. Average: {tracker.average():.1f}.")

    rng = np.random.RandomState(42)
    study_hours = rng.uniform(1, 25, size=(30, 1))
    exam_scores = 50 + 2 * study_hours.ravel() + rng.normal(0, 5, size=30)
    model = LinearRegression().fit(study_hours, exam_scores)
    fitted_scores = model.predict(study_hours)
    print(f"Slope: {model.coef_[0]:.2f}. Intercept: {model.intercept_:.2f}.")
    print(f"Training MSE: {mean_squared_error(exam_scores, fitted_scores):.2f}.")
    print(f"Training R^2: {r2_score(exam_scores, fitted_scores):.3f}.")
    write_pickle(directory / "study_hours_model.pkl", model)
    joblib.dump(model, directory / "study_hours_model.joblib")
    save_plot(directory / "study_hours_model.png", study_hours, exam_scores, model)


def predict_score(model, hours):
    if not np.isfinite(hours) or hours < 0:
        raise ValueError("Study hours must be finite and nonnegative.")
    return float(model.predict([[hours]])[0])


def load_examples(directory):
    fruits = read_pickle(directory / "fruits.pkl")
    scores = read_pickle(directory / "scores.pkl")
    combined = read_pickle(directory / "complex_data.pkl")
    if combined["fruits"] != fruits or combined["scores"] != scores:
        raise RuntimeError("The reloaded containers contain different values.")
    print(f"Reloaded {len(fruits)} fruits and {len(scores)} scores.")

    tracker = read_pickle(directory / "ml_course_grades.pkl")
    print(f"Reloaded grade tracker. Average: {tracker.average():.1f}.")
    for student, score in {"Frank": 89, "Grace": 92, "Hannah": 79}.items():
        tracker.record(student, score)
    write_pickle(directory / "ml_course_grades_updated.pkl", tracker)
    student, score = tracker.top_student()
    print(f"Updated {len(tracker.grades)} grades. Average: {tracker.average():.1f}. Top: {student} ({score}).")

    pickle_model = read_pickle(directory / "study_hours_model.pkl")
    joblib_model = joblib.load(directory / "study_hours_model.joblib")
    hours = np.array([[5.0], [10.0], [12.5], [15.0], [17.5], [20.0]])
    pickle_scores = pickle_model.predict(hours)
    np.testing.assert_allclose(pickle_scores, joblib_model.predict(hours), rtol=0, atol=0)
    for hour, score in zip(hours.ravel(), pickle_scores):
        print(f"{hour:4.1f} hours -> {score:5.1f} points")
    print(f"pickle and joblib match on all {len(hours)} inputs.")
    for hour in (7.8, 12.5, 22.0):
        print(f"predict_score({hour:g}) = {predict_score(pickle_model, hour):.1f}")


def main():
    parser = argparse.ArgumentParser(description="Save and reload Python objects and a scikit-learn model.")
    parser.add_argument("mode", choices=("save", "load"))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    action = save_examples if args.mode == "save" else load_examples
    action(args.output_dir)


if __name__ == "__main__":
    main()
