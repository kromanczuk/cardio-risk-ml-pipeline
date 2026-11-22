"""Streamlit front-end for the Heart Disease Risk API.

Run locally (from the frontend/ folder, so .streamlit/config.toml is picked up):
    API_URL=http://127.0.0.1:8000 streamlit run app.py
"""

import os

import plotly.graph_objects as go
import requests
import streamlit as st

API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000").rstrip("/")
TIMEOUT = 60  # seconds; the first request after idle waits for a Cloud Run cold start

# Readable labels for the UCI codes. Codes and ranges must match src/schemas.py.
SEX = {0: "Female", 1: "Male"}
YES_NO = {0: "No", 1: "Yes"}
CP = {1: "Typical angina", 2: "Atypical angina", 3: "Non-anginal pain", 4: "Asymptomatic"}
RESTECG = {0: "Normal", 1: "ST-T wave abnormality", 2: "Left ventricular hypertrophy"}
SLOPE = {1: "Upsloping", 2: "Flat", 3: "Downsloping"}
THAL = {3: "Normal", 6: "Fixed defect", 7: "Reversible defect"}

LABELS = {
    "age": "Age (years)",
    "sex": "Sex",
    "trestbps": "Resting blood pressure (mm Hg)",
    "chol": "Cholesterol (mg/dl)",
    "fbs": "Fasting blood sugar > 120 mg/dl",
    "cp": "Chest pain type",
    "restecg": "Resting ECG",
    "thalach": "Max heart rate achieved",
    "exang": "Exercise-induced angina",
    "oldpeak": "ST depression (oldpeak)",
    "slope": "Slope of peak exercise ST segment",
    "ca": "Major vessels colored by fluoroscopy",
    "thal": "Thalassemia",
}
FEATURES = list(LABELS)
EMPTY_PATIENT = dict.fromkeys(FEATURES)  # every input starts blank
EXAMPLES = {
    "high": {
        "age": 54, "sex": 1, "cp": 4, "trestbps": 130, "chol": 250, "fbs": 0, "restecg": 2,
        "thalach": 150, "exang": 1, "oldpeak": 1.5, "slope": 2, "ca": 0, "thal": 7,
    },
    "low": {
        "age": 38, "sex": 0, "cp": 3, "trestbps": 118, "chol": 190, "fbs": 0, "restecg": 0,
        "thalach": 178, "exang": 0, "oldpeak": 0.0, "slope": 1, "ca": 0, "thal": 3,
    },
}  # fmt: skip


# ---------------------------------------------------------------- API calls
@st.cache_data(ttl=30, show_spinner=False)
def get_health() -> dict | None:
    try:
        r = requests.get(f"{API_URL}/health", timeout=TIMEOUT)
        r.raise_for_status()
        return r.json()
    except requests.RequestException:
        return None


def predict(patient: dict) -> tuple[dict | None, str | None]:
    """Return (result, error message)."""
    try:
        r = requests.post(f"{API_URL}/predict", json=patient, timeout=TIMEOUT)
    except requests.Timeout:
        return None, "The API took too long to respond. Please try again in a moment."
    except requests.ConnectionError:
        return None, f"Can't reach the API at {API_URL}."

    if r.status_code == 200:
        return r.json(), None
    if r.status_code == 503:
        return None, "The API is running, but no model is loaded yet. Please try again later."
    if r.status_code == 422:
        fields = ", ".join(
            LABELS.get(e["loc"][-1], str(e["loc"][-1])) for e in r.json().get("detail", [])
        )
        return None, f"Some inputs were rejected by the API: {fields}."
    return None, f"Unexpected API error ({r.status_code})."


# ---------------------------------------------------------------- UI helpers
def load_patient(values: dict) -> None:
    for key, value in values.items():
        st.session_state[key] = value
    st.session_state.pop("result", None)
    st.session_state.pop("error", None)


def gauge(probability: float) -> go.Figure:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=probability * 100,
            number={"suffix": "%", "valueformat": ".0f"},
            gauge={
                "axis": {"range": [0, 100], "ticksuffix": "%"},
                "bar": {"color": "#475569", "thickness": 0.3},
                "steps": [
                    {"range": [0, 30], "color": "#bbf7d0"},
                    {"range": [30, 60], "color": "#fde68a"},
                    {"range": [60, 100], "color": "#fecaca"},
                ],
            },
        )
    )
    fig.update_layout(height=250, margin={"l": 45, "r": 45, "t": 30, "b": 0})
    return fig


def show_result(result: dict) -> None:
    p, level = result["probability"], result["risk_level"]
    st.plotly_chart(gauge(p), use_container_width=True, config={"displayModeBar": False})
    message = {
        "low": st.success,
        "medium": st.warning,
        "high": st.error,
    }[level]
    message(f"**{level.capitalize()} risk** of heart disease")
    verdict = "heart disease" if result["prediction"] == 1 else "no heart disease"
    st.markdown(
        f"The model estimates a **{p:.0%}** probability of heart disease, "
        f"so it predicts **{verdict}** (decision threshold: 50%)."
    )
    st.caption("Risk bands: low < 30% ≤ medium < 60% ≤ high")


# ---------------------------------------------------------------- page
st.set_page_config(page_title="Heart Disease Risk", page_icon="🫀", layout="wide")
# Hide the -/+ step buttons on number inputs (Streamlit has no option for this).
st.markdown(
    "<style>[data-testid='stNumberInputStepDown'], [data-testid='stNumberInputStepUp']"
    " {display: none;}</style>",
    unsafe_allow_html=True,
)
for key, value in EMPTY_PATIENT.items():
    st.session_state.setdefault(key, value)

st.title("🫀 Heart Disease Risk")
st.caption(
    "Estimates the probability of heart disease from 13 clinical measurements, using a model "
    "trained on the UCI Heart Disease (Cleveland) dataset. "
    "**Educational project — not medical advice.**"
)

inputs, output = st.columns([3, 2], gap="large")

with inputs:
    b1, b2, b3 = st.columns(3)
    b1.button("High-risk example", on_click=load_patient, args=(EXAMPLES["high"],),
              use_container_width=True)  # fmt: skip
    b2.button("Low-risk example", on_click=load_patient, args=(EXAMPLES["low"],),
              use_container_width=True)  # fmt: skip
    b3.button("Clear", on_click=load_patient, args=(EMPTY_PATIENT,), use_container_width=True)

    def number(col, key, lo, hi, step=1, fmt=None):
        col.number_input(LABELS[key], min_value=lo, max_value=hi, step=step, format=fmt,
                         placeholder=f"{lo}–{hi}", key=key)  # fmt: skip

    def choice(col, key, options, radio=False):
        if radio:
            col.radio(LABELS[key], options=list(options), format_func=options.get,
                      horizontal=True, key=key)  # fmt: skip
        else:
            col.selectbox(LABELS[key], options=list(options), format_func=options.get,
                          placeholder="Select…", key=key)  # fmt: skip

    with st.form("patient"):
        with st.container(border=True):
            st.subheader("Demographics")
            c1, c2 = st.columns(2)
            number(c1, "age", 18, 100)
            choice(c2, "sex", SEX, radio=True)

        with st.container(border=True):
            st.subheader("Vitals & labs")
            c1, c2, c3 = st.columns(3)
            number(c1, "trestbps", 80, 220)
            number(c2, "chol", 100, 600)
            choice(c3, "fbs", YES_NO, radio=True)

        with st.container(border=True):
            st.subheader("Symptoms, ECG & exercise test")
            c1, c2 = st.columns(2)
            choice(c1, "cp", CP)
            choice(c2, "restecg", RESTECG)
            number(c1, "thalach", 60, 220)
            choice(c2, "exang", YES_NO, radio=True)
            number(c1, "oldpeak", 0.0, 7.0, step=0.1, fmt="%.1f")
            choice(c2, "slope", SLOPE)
            choice(c1, "ca", {n: str(n) for n in range(4)})
            choice(c2, "thal", THAL)

        submitted = st.form_submit_button("Predict risk", type="primary", use_container_width=True)

    if submitted:
        patient = {f: st.session_state[f] for f in FEATURES}
        missing = [LABELS[f] for f, v in patient.items() if v is None]
        if missing:
            st.session_state["result"] = None
            st.session_state["error"] = "Please fill in all fields. Missing: " + ", ".join(missing)
        else:
            with st.spinner("Asking the model… (the first request can take a few seconds)"):
                st.session_state["result"], st.session_state["error"] = predict(patient)

with output:
    st.subheader("Result")
    if st.session_state.get("error"):
        st.error(st.session_state["error"])
    elif st.session_state.get("result"):
        show_result(st.session_state["result"])
    else:
        st.info("Fill in the patient's data and click **Predict risk**.")

# ---------------------------------------------------------------- footer
st.divider()
health = get_health()
if health is None:
    st.caption(f"🔴 API unreachable at {API_URL}")
elif not health["model_loaded"]:
    st.caption(f"🟡 API online, no model loaded · [API docs]({API_URL}/docs)")
else:
    st.caption(f"🟢 API online · model `{health['model_version']}` · [API docs]({API_URL}/docs)")
