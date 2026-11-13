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
import sklearn
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


def pretty(name: str) -> str:
    """'YearsAtCompany' -> 'Years At Company'"""
    out = ""
    for i, ch in enumerate(name):
        if i and ch.isupper() and not name[i - 1].isupper():
            out += " "
        out += ch
    return out


@st.cache_resource(show_spinner="Loading model...")
def load_bundle(path: str):
    return joblib.load(path)


def load_model_or_stop():
    if not MODEL_PATH.exists():
        st.error(
            "**model.pkl was not found.**\n\n"
            "Open a terminal in this project folder and run:\n\n`python train.py`\n\n"
            "Then restart the app."
        )
        st.stop()
    try:
        bundle = load_bundle(str(MODEL_PATH))
    except Exception as exc:  
        st.error(
            "**model.pkl could not be loaded.** This usually happens when the file was created "
            "with a different scikit-learn version.\n\n"
            "Fix: run `python train.py` on this computer to recreate it, then restart the app.\n\n"
            f"Technical details: `{type(exc).__name__}: {exc}`"
        )
        st.stop()

    required = {"pipeline", "feature_columns", "numeric_columns", "categorical_columns",
                "numeric_info", "categories", "metrics"}
    if not isinstance(bundle, dict) or not required <= set(bundle):
        st.error("model.pkl does not have the expected content. Please run `python train.py` again.")
        st.stop()
    return bundle



def number_field(col: str, info: dict, key: str):
    """Create the right numeric widget for one column, limited to the range seen in training."""
    label = pretty(col)
    help_txt = HELP_TEXT.get(col)
    lo, hi, med = info["min"], info["max"], info["median"]

    if info["is_integer"]:
        lo, hi, med = int(lo), int(hi), int(round(med))
        if hi - lo <= 10:  
            return st.slider(label, lo, hi, med, step=1, help=help_txt, key=key)
        return st.number_input(label, min_value=lo, max_value=hi, value=med, step=1,
                               help=help_txt, key=key)
    return st.number_input(label, min_value=float(lo), max_value=float(hi), value=float(round(med, 1)),
                           step=0.1, format="%.1f", help=help_txt, key=key)


def collect_inputs(bundle: dict) -> dict:
    """Draw all input widgets and return {column: value}."""
    cols = bundle["feature_columns"]
    num_info = bundle["numeric_info"]
    categories = bundle["categories"]
    dept_roles = bundle.get("department_roles", {})

    groups = {g: [c for c in members if c in cols] for g, members in FIELD_GROUPS.items()}
    placed = {c for members in groups.values() for c in members}
    leftovers = [c for c in cols if c not in placed]
    if leftovers:
        groups["Other"] = leftovers
    groups = {g: m for g, m in groups.items() if m}

    values = {}
    tabs = st.tabs(list(groups))
    for tab, (_, members) in zip(tabs, groups.items()):
        with tab:
            left, right = st.columns(2)
            for i, col in enumerate(members):
                with (left if i % 2 == 0 else right):
                    key = f"in_{col}"
                    if col in num_info:
                        values[col] = number_field(col, num_info[col], key)
                    else:
                        options = categories[col]
                        
                        if col == "JobRole" and dept_roles and values.get("Department") in dept_roles:
                            options = dept_roles[values["Department"]]
                        values[col] = st.selectbox(pretty(col), options, key=key,
                                                   help=HELP_TEXT.get(col))
    return values


def validate(values: dict, bundle: dict) -> list:
    """Return a list of problems. Empty list = inputs are fine."""
    problems = []
    cols = bundle["feature_columns"]

    for c in cols:
        if c not in values or values[c] is None or (isinstance(values[c], str) and not values[c].strip()):
            problems.append(f"'{pretty(c)}' is missing.")

    for c, info in bundle["numeric_info"].items():
        v = values.get(c)
        if v is not None and not (info["min"] <= v <= info["max"]):
            problems.append(f"{pretty(c)} must be between {info['min']:g} and {info['max']:g}.")
    for c, allowed in bundle["categories"].items():
        if values.get(c) not in allowed:
            problems.append(f"'{values.get(c)}' is not a known value for {pretty(c)}.")

    def le(a, b, msg):
        if a in values and b in values and values[a] > values[b] + 1e-9:
            problems.append(msg)

    le("YearsAtCompany", "TotalWorkingYears", "Years At Company cannot be greater than Total Working Years.")
    le("YearsInCurrentRole", "YearsAtCompany", "Years In Current Role cannot be greater than Years At Company.")
    le("YearsSinceLastPromotion", "YearsAtCompany",
       "Years Since Last Promotion cannot be greater than Years At Company.")
    if "Age" in values and "TotalWorkingYears" in values and values["TotalWorkingYears"] > values["Age"] - 14:
        problems.append("Total Working Years looks too high for this Age (the data has at least 14 years "
                        "between age and years worked).")

    roles = bundle.get("department_roles", {})
    if roles and values.get("Department") in roles and values.get("JobRole") not in roles[values["Department"]]:
        problems.append(f"Job Role '{values.get('JobRole')}' does not belong to the "
                        f"'{values.get('Department')}' department.")
    return problems


def explain_prediction(pipeline, row: pd.DataFrame, top_n: int = 8):
    """For linear models: how much each input pushes the prediction toward Leave (+) or Stay (-)."""
    try:
        model = pipeline.named_steps["model"]
        pre = pipeline.named_steps["preprocessor"]
        if not hasattr(model, "coef_"):
            return None
        transformed = pre.transform(row)
        transformed = transformed.toarray() if hasattr(transformed, "toarray") else np.asarray(transformed)
        contrib = transformed[0] * model.coef_[0]
        names = pre.get_feature_names_out()
        original = list(row.columns)
        totals = {}
        for name, value in zip(names, contrib):
            clean = name.split("__", 1)[1]                     
            owner = next((c for c in sorted(original, key=len, reverse=True)
                          if clean == c or clean.startswith(c + "_")), clean)
            totals[owner] = totals.get(owner, 0.0) + float(value)
        series = pd.Series(totals)
        series = series.reindex(series.abs().sort_values(ascending=False).index).head(top_n)
        series.index = [pretty(i) for i in series.index]
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
    if st.button(" Predict", type="primary"):
        problems = validate(values, bundle)
        if problems:
            st.error("Please fix the following before predicting:")
            for p in problems:
                st.write(f"- {p}")
            return

        row = pd.DataFrame([values], columns=bundle["feature_columns"])
        try:
            predicted = int(pipeline.predict(row)[0])
            probabilities = pipeline.predict_proba(row)[0] if hasattr(pipeline, "predict_proba") else None
        except Exception as exc:
            st.error(f"The model could not make a prediction: `{type(exc).__name__}: {exc}`")
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
            chart = pd.DataFrame({"Probability": [p_stay, p_leave]}, index=[labels[0], labels[1]])
            st.bar_chart(chart)
            st.caption("The predicted class uses the model's standard 50% threshold.")

        contributions = explain_prediction(pipeline, row)
        if contributions is not None:
            st.subheader("What influenced this prediction most?")
            st.caption("Positive values push toward **Leave**, negative values toward **Stay** "
                       "(contribution to the model's log-odds).")
            st.bar_chart(contributions.rename("Contribution"))

        with st.expander("Show the exact input row sent to the model"):
            st.dataframe(row)


main()
