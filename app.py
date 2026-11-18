from pathlib import Path
import math

import gradio as gr
import joblib
import numpy as np
import pandas as pd

MODEL_PATH = Path(__file__).with_name("diabetes_model.pkl")
FEATURE_NAMES = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age",
]

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model file not found: {MODEL_PATH}. Keep diabetes_model.pkl beside app.py."
    )

model = joblib.load(MODEL_PATH)


def predict_diabetes(pregnancies, glucose, blood_pressure, skin_thickness,
                     insulin, bmi, dpf, age):
    """Validate user inputs and return a model estimate with clear limitations."""
    values = {
        "Pregnancies": pregnancies,
        "Glucose": glucose,
        "BloodPressure": blood_pressure,
        "SkinThickness": skin_thickness,
        "Insulin": insulin,
        "BMI": bmi,
        "DiabetesPedigreeFunction": dpf,
        "Age": age,
    }

    cleaned = {}
    for name, value in values.items():
        if value is None:
            return "Please enter a value for every field."
        try:
            number = float(value)
        except (TypeError, ValueError):
            return f"{name} must be a number."
        if not math.isfinite(number):
            return f"{name} must be a finite number."
        if number < 0:
            return f"{name} cannot be negative."
        cleaned[name] = number

    # Basic input sanity checks; these are not clinical diagnostic limits.
    limits = {
        "Pregnancies": 20,
        "Glucose": 300,
        "BloodPressure": 200,
        "SkinThickness": 100,
        "Insulin": 1000,
        "BMI": 80,
        "DiabetesPedigreeFunction": 3,
        "Age": 120,
    }
    for name, maximum in limits.items():
        if cleaned[name] > maximum:
            return f"{name} is outside the supported input range (0–{maximum})."

    row = pd.DataFrame([[cleaned[name] for name in FEATURE_NAMES]],
                       columns=FEATURE_NAMES)
    prediction = int(model.predict(row)[0])
    probability = float(model.predict_proba(row)[0][1])

    label = "Higher model-estimated likelihood" if prediction == 1 else "Lower model-estimated likelihood"
    return (
        f"### {label}\n"
        f"**Model-estimated probability for the positive class: {probability:.1%}**\n\n"
        "This is an educational machine-learning estimate based on a small "
        "historical dataset. It is not a diagnosis, screening test, or medical "
        "advice. A qualified healthcare professional must interpret health "
        "measurements and arrange appropriate testing."
    )


with gr.Blocks(title="Diabetes Risk Prediction — Educational Demo") as demo:
    gr.Markdown(
        "# Diabetes Prediction — Educational Demo\n"
        "Enter the eight features used by the model. Values of zero for "
        "Glucose, Blood Pressure, Skin Thickness, Insulin, or BMI are treated "
        "as missing by the trained preprocessing pipeline."
    )
    with gr.Row():
        with gr.Column():
            pregnancies = gr.Number(label="Pregnancies", value=0, precision=0)
            glucose = gr.Number(label="Glucose", value=120)
            blood_pressure = gr.Number(label="Blood Pressure", value=70)
            skin_thickness = gr.Number(label="Skin Thickness", value=20)
        with gr.Column():
            insulin = gr.Number(label="Insulin", value=80)
            bmi = gr.Number(label="BMI", value=30)
            dpf = gr.Number(label="Diabetes Pedigree Function", value=0.5)
            age = gr.Number(label="Age", value=33, precision=0)
    submit = gr.Button("Estimate", variant="primary")
    result = gr.Markdown()
    submit.click(
        fn=predict_diabetes,
        inputs=[pregnancies, glucose, blood_pressure, skin_thickness,
                insulin, bmi, dpf, age],
        outputs=result,
    )
    gr.Markdown(
        "---\n"
        "**Important:** This project is for coursework and demonstration only. "
        "It must not be used to make healthcare decisions."
    )

if __name__ == "__main__":
    demo.launch()
