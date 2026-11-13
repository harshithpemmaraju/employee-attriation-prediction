"""
app.py - Streamlit app for Employee Attrition Prediction

The app only LOADS the trained model (model.pkl) - it never trains anything.
Create/refresh model.pkl first with:  python train.py
Run the app with:                     streamlit run app.py
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

MODEL_PATH = Path(__file__).resolve().parent / "model.pkl"

FIELD_GROUPS = {
    " Personal": ["Age", "Gender", "MaritalStatus", "Education", "DistanceFromHome"],
    " Job": ["Department", "JobRole", "JobLevel", "BusinessTravel", "OverTime", "PercentRemoteWork"],
    " Experience": [
        "TotalWorkingYears", "NumCompaniesWorked", "YearsAtCompany",
        "YearsInCurrentRole", "YearsSinceLastPromotion", "TrainingTimesLastYear",
    ],
    " Pay": ["MonthlyIncome", "DailyRate", "HourlyRate", "MonthlyRate", "PercentSalaryHike", "StockOptionLevel"],
    " Satisfaction & Performance": [
        "JobSatisfaction", "EnvironmentSatisfaction", "WorkLifeBalance",
        "RelationshipSatisfaction", "PerformanceRating",
    ],
}

HELP_TEXT = {
    "Age": "Employee age in years.",
    "DistanceFromHome": "Distance from home to work, as recorded in the dataset.",
    "JobLevel": "Job level, 1 to 5 as in the dataset.",
    "PercentRemoteWork": "Percentage of work done remotely (0-100).",
    "TotalWorkingYears": "Total years of work experience (all employers).",
    "YearsAtCompany": "Years at the current company. Cannot exceed total working years.",
    "YearsInCurrentRole": "Years in the current role. Cannot exceed years at the company.",
    "YearsSinceLastPromotion": "Years since last promotion. Cannot exceed years at the company.",
    "TrainingTimesLastYear": "Number of training sessions attended last year.",
    "MonthlyIncome": "Monthly income, in the units used by the dataset.",
    "PercentSalaryHike": "Last salary hike in percent.",
    "StockOptionLevel": "Stock option level, 0 to 3 as in the dataset.",
    "JobSatisfaction": "Rating on the dataset's 1-4 scale.",
    "EnvironmentSatisfaction": "Rating on the dataset's 1-4 scale.",
    "WorkLifeBalance": "Rating on the dataset's 1-4 scale.",
    "RelationshipSatisfaction": "Rating on the dataset's 1-4 scale.",
    "PerformanceRating": "Performance rating, 2 to 4 as in the dataset.",
}


def pretty(name):
    """Turn 'YearsAtCompany' into 'Years At Company'."""
    result = ""
    for i, letter in enumerate(name):
        if i > 0 and letter.isupper() and not name[i - 1].isupper():
            result += " "
        result += letter
    return result


@st.cache_resource(show_spinner="Loading model...")
def load_bundle(path):
    """Read model.pkl from disk (cached so it is only loaded once)."""
    return joblib.load(path)


def load_model_or_stop():
    """Load model.pkl. If anything is wrong, show a friendly message and stop the app."""
    if not MODEL_PATH.exists():
        st.error(
            "**model.pkl was not found.**\n\n"
            "Open a terminal in this project folder and run:\n\n`python train.py`\n\n"
            "Then restart the app."
        )
        st.stop()

    try:
        bundle = load_bundle(str(MODEL_PATH))
    except Exception as error:
        st.error(
            "**model.pkl could not be loaded.** This usually happens when the file was created "
            "with a different scikit-learn version.\n\n"
            "Fix: run `python train.py` on this computer to recreate it, then restart the app.\n\n"
            f"Technical details: `{type(error).__name__}: {error}`"
        )
        st.stop()

    needed_keys = ["pipeline", "feature_columns", "numeric_info", "categories"]
    if not isinstance(bundle, dict) or any(key not in bundle for key in needed_keys):
        st.error("model.pkl does not have the expected content. Please run `python train.py` again.")
        st.stop()

    return bundle



def number_field(col, info, key):
    """Create the right widget for one numeric column (limited to the range seen in training)."""
    label = pretty(col)
    help_text = HELP_TEXT.get(col)
    smallest = info["min"]
    largest = info["max"]
    middle = info["median"]

    
    if info["is_integer"]:
        smallest, largest, middle = int(smallest), int(largest), int(round(middle))
        if largest - smallest <= 10:
            
            return st.slider(label, smallest, largest, middle, step=1, help=help_text, key=key)
        return st.number_input(label, min_value=smallest, max_value=largest, value=middle,
                               step=1, help=help_text, key=key)

    return st.number_input(label, min_value=float(smallest), max_value=float(largest),
                           value=float(round(middle, 1)), step=0.1, format="%.1f",
                           help=help_text, key=key)


def category_field(col, bundle, values):
    """Create a dropdown for one categorical column."""
    options = bundle["categories"][col]

    department_roles = bundle.get("department_roles", {})
    if col == "JobRole" and values.get("Department") in department_roles:
        options = department_roles[values["Department"]]

    return st.selectbox(pretty(col), options, key=f"in_{col}", help=HELP_TEXT.get(col))


def get_tab_groups(feature_columns):
    """Decide which fields go into which tab (only fields the model really uses)."""
    groups = {}
    placed = []
    for tab_name, members in FIELD_GROUPS.items():
        fields = [col for col in members if col in feature_columns]
        if fields:
            groups[tab_name] = fields
            placed += fields

    leftovers = [col for col in feature_columns if col not in placed]
    if leftovers:
        groups["Other"] = leftovers
    return groups


def collect_inputs(bundle):
    """Draw all input widgets and return their values as {column_name: value}."""
    groups = get_tab_groups(bundle["feature_columns"])
    values = {}

    tabs = st.tabs(list(groups))
    for tab, tab_name in zip(tabs, groups):
        with tab:
            left, right = st.columns(2)
            for i, col in enumerate(groups[tab_name]):
                box = left if i % 2 == 0 else right
                with box:
                    if col in bundle["numeric_info"]:
                        values[col] = number_field(col, bundle["numeric_info"][col], key=f"in_{col}")
                    else:
                        values[col] = category_field(col, bundle, values)
    return values


def is_greater(values, first, second):
    """True if both fields exist and `first` is bigger than `second`."""
    return first in values and second in values and values[first] > values[second]


def validate(values, bundle):
    """Return a list of problems. An empty list means the inputs are fine."""
    problems = []

    for col in bundle["feature_columns"]:
        value = values.get(col)
        if value is None or (isinstance(value, str) and value.strip() == ""):
            problems.append(f"'{pretty(col)}' is missing.")

    for col, info in bundle["numeric_info"].items():
        value = values.get(col)
        if value is not None and not (info["min"] <= value <= info["max"]):
            problems.append(f"{pretty(col)} must be between {info['min']:g} and {info['max']:g}.")

    for col, allowed in bundle["categories"].items():
        if values.get(col) not in allowed:
            problems.append(f"'{values.get(col)}' is not a known value for {pretty(col)}.")

    if is_greater(values, "YearsAtCompany", "TotalWorkingYears"):
        problems.append("Years At Company cannot be greater than Total Working Years.")
    if is_greater(values, "YearsInCurrentRole", "YearsAtCompany"):
        problems.append("Years In Current Role cannot be greater than Years At Company.")
    if is_greater(values, "YearsSinceLastPromotion", "YearsAtCompany"):
        problems.append("Years Since Last Promotion cannot be greater than Years At Company.")
    if "Age" in values and "TotalWorkingYears" in values:
        if values["TotalWorkingYears"] > values["Age"] - 14:
            problems.append("Total Working Years looks too high for this Age (the data has at least 14 years "
                            "between age and years worked).")

    department_roles = bundle.get("department_roles", {})
    department = values.get("Department")
    if department in department_roles and values.get("JobRole") not in department_roles[department]:
        problems.append(f"Job Role '{values.get('JobRole')}' does not belong to the "
                        f"'{department}' department.")

    return problems


def find_original_column(clean_name, original_columns):
    """'Department_Sales' -> 'Department', 'Age' -> 'Age'."""
    for col in sorted(original_columns, key=len, reverse=True):
        if clean_name == col or clean_name.startswith(col + "_"):
            return col
    return clean_name


def explain_prediction(pipeline, row, top_n=8):
    """
    Show how much each input pushed the prediction toward Leave (+) or Stay (-).
    contribution = model coefficient x (scaled/encoded input value)
    Returns None if the model cannot be explained this way.
    """
    try:
        model = pipeline.named_steps["model"]
        preprocessor = pipeline.named_steps["preprocessor"]

        if not hasattr(model, "coef_"):
            return None

        transformed = preprocessor.transform(row)
        if hasattr(transformed, "toarray"):
            transformed = transformed.toarray()
        transformed = np.asarray(transformed)

        contributions = transformed[0] * model.coef_[0]
        names = preprocessor.get_feature_names_out()

        totals = {}
        for name, value in zip(names, contributions):
            clean_name = name.split("__", 1)[1]          
            owner = find_original_column(clean_name, list(row.columns))
            totals[owner] = totals.get(owner, 0.0) + float(value)

        series = pd.Series(totals)
        strongest_first = series.abs().sort_values(ascending=False).index
        series = series.reindex(strongest_first).head(top_n)
        series.index = [pretty(name) for name in series.index]
        return series
    except Exception:
        return None



def main():
    bundle = load_model_or_stop()
    pipeline = bundle["pipeline"]

    st.title(" Employee Attrition Predictor")
    st.markdown(
        "Predict whether an employee is **likely to leave** or **likely to stay**, based on the same "
        "28 employee features used to train the model. Fill in the tabs below and press **Predict**."
    )

    st.subheader("Employee details")
    values = collect_inputs(bundle)

    st.divider()
    if not st.button(" Predict", type="primary"):
        return

    problems = validate(values, bundle)
    if problems:
        st.error("Please fix the following before predicting:")
        for problem in problems:
            st.write(f"- {problem}")
        return

    row = pd.DataFrame([values], columns=bundle["feature_columns"])
    try:
        predicted = int(pipeline.predict(row)[0])
        probabilities = pipeline.predict_proba(row)[0] if hasattr(pipeline, "predict_proba") else None
    except Exception as error:
        st.error(f"The model could not make a prediction: `{type(error).__name__}: {error}`")
        return

    labels = bundle.get("class_labels", {0: "Stay", 1: "Leave"})
    classes = list(pipeline.classes_)

    st.subheader("Result")
    if predicted == 1:
        st.error("### ⚠️ Likely to LEAVE")
    else:
        st.success("### ✅ Likely to STAY")

    if probabilities is not None:
        p_leave = float(probabilities[classes.index(1)])
        p_stay = float(probabilities[classes.index(0)])

        left_col, right_col = st.columns(2)
        left_col.metric("Probability of leaving", f"{p_leave:.1%}")
        right_col.metric("Probability of staying", f"{p_stay:.1%}")
        st.progress(min(max(p_leave, 0.0), 1.0), text=f"Attrition risk: {p_leave:.1%}")
        st.caption("The predicted class uses the model's standard 50% threshold.")

    contributions = explain_prediction(pipeline, row)
main()
