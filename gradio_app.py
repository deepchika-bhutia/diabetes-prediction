import pandas as pd
import joblib
import gradio as gr
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_score, recall_score, f1_score
from PIL import Image


model = joblib.load("diabetes_model.pkl")
model_columns = joblib.load("model_columns.pkl")
X_test = pd.read_csv("X_test.csv")
y_test = pd.read_csv("y_test.csv").values.ravel()


feature_importance_img = Image.open("feature_importance.png")


y_pred = model.predict(X_test)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

metrics_md = (
    "### Model Performance on Held-out Test Set\n\n"
    f"- **Precision (class 1 – diabetic)**: `{precision:.3f}`\n"
    f"- **Recall (class 1 – diabetic)**: `{recall:.3f}`\n"
    f"- **F1-score (class 1 – diabetic)**: `{f1:.3f}`\n"
)


def build_explanation(contrib_df):
    if contrib_df.empty:
        return "No meaningful feature contributions could be computed for this example."
    lines = []
    for _, row in contrib_df.head(3).iterrows():
        feat = row["feature"]
        contrib = row["contribution"]
        direction = "increased" if contrib > 0 else "decreased"
        lines.append(f"- **{feat}** {direction} the predicted diabetes risk.")
    return "### Why this prediction?\n" + "\n".join(lines)


def predict_and_explain(
    age,
    gender,
    hypertension,
    heart_disease,
    smoking_history,
    bmi,
    HbA1c,
    glucose,
):

    input_dict = {
        "age": [age],
        "gender": [gender],
        "hypertension": [1 if hypertension == "Yes" else 0],
        "heart_disease": [1 if heart_disease == "Yes" else 0],
        "smoking_history": [smoking_history],
        "bmi": [bmi],
        "HbA1c_level": [HbA1c],
        "blood_glucose_level": [glucose],
    }
    input_df = pd.DataFrame(input_dict)


    input_df = pd.get_dummies(input_df, drop_first=True)


    for col in model_columns:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[model_columns]

    proba = model.predict_proba(input_df)[0]
    pred_class = int(np.argmax(proba))

    result = "🔴 **Diabetic (1)**" if pred_class == 1 else "🟢 **Not Diabetic (0)**"
    confidence = f"{proba[pred_class] * 100:.1f}%"

    proba_not_diab = proba[0] * 100
    proba_diab = proba[1] * 100

    proba_text = (
        f"- Probability of **Not Diabetic (0)**: `{proba_not_diab:.1f}%`\n"
        f"- Probability of **Diabetic (1)**: `{proba_diab:.1f}%`"
    )


    feature_values = input_df.values[0]
    feature_importances = model.feature_importances_
    raw_contrib = feature_values * feature_importances

    contrib_df = pd.DataFrame(
        {
            "feature": model_columns,
            "contribution": raw_contrib,
        }
    )

    contrib_df = contrib_df[contrib_df["contribution"] != 0]
    contrib_df = contrib_df.sort_values("contribution", ascending=False).head(10)


    fig, ax = plt.subplots(figsize=(7, 4))
    if not contrib_df.empty:
        sns.barplot(
            data=contrib_df,
            x="contribution",
            y="feature",
            ax=ax,
        )
    ax.set_title("Top 10 Feature Contributions (approximate)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Contribution score")
    ax.set_ylabel("Feature")
    plt.tight_layout()


    explanation_md = build_explanation(contrib_df)

    return (
        result,
        confidence,
        proba_text,
        fig,
        explanation_md,
    )


gender_choices = ["Male", "Female", "Other"]
smoking_choices = ["never", "No Info", "current", "former", "ever", "not current"]

with gr.Blocks(title="Diabetes Prediction & Analysis") as demo:
    gr.Markdown("# Diabetes Prediction & Feature Analysis")

    gr.Markdown(
        "Enter basic health information on the **left**. "
        "The model prediction and explanation will appear on the **right**.\n\n"
        "**Ranges used in this app:**\n"
        "- Age: `0 – 80` years\n"
        "- BMI: `10 – 72`\n"
        "- HbA1c level: `3.5 – 15.0`\n"
        "- Blood glucose level: `50 – 400` mg/dL\n"
    )

    with gr.Row():
        # ------------------------
        # LEFT: Input Form
        # ------------------------
        with gr.Column(scale=1):
            gr.Markdown("## Input Form")

            age_input = gr.Number(
                label="Age (years, 0–80)",
                value=50,
                minimum=0,
                maximum=80,
            )
            gender_input = gr.Dropdown(
                gender_choices,
                label="Gender",
                value="Male",
            )
            hypertension_input = gr.Radio(
                ["No", "Yes"],
                label="Hypertension",
                value="No",
            )
            heart_disease_input = gr.Radio(
                ["No", "Yes"],
                label="Heart Disease",
                value="No",
            )
            smoking_input = gr.Dropdown(
                smoking_choices,
                label="Smoking History",
                value="never",
            )
            bmi_input = gr.Number(
                label="BMI (10–72)",
                value=25.0,
                minimum=10.0,
                maximum=72.0,
            )
            hba1c_input = gr.Number(
                label="HbA1c Level (3.5–15.0)",
                value=5.5,
                minimum=3.5,
                maximum=15.0,
            )
            glucose_input = gr.Number(
                label="Blood Glucose Level (mg/dL, 50–400)",
                value=120,
                minimum=50,
                maximum=400,
            )

            predict_btn = gr.Button("🔍 Predict", variant="primary")

        with gr.Column(scale=1):
            gr.Markdown("## Prediction & Explanation")

            result_output = gr.Markdown(label="Prediction Result")
            confidence_output = gr.Markdown(label="Model Confidence")
            proba_output = gr.Markdown(label="Prediction Probabilities")

            gr.Markdown("### Individual Prediction Contributions")
            contribution_plot = gr.Plot(label="Feature Contributions")

            gr.Markdown("### Explanation (natural language)")
            explanation_output = gr.Markdown()

    # ------------------------
    # BOTTOM: Global metrics & importance
    # ------------------------
    gr.Markdown("---")
    gr.Markdown("## Global Feature Importance & Model Performance")

    with gr.Row():
        with gr.Column(scale=1):
            static_heatmap = gr.Image(
                label="Global Feature Importance (Random Forest)",
                type="pil",
            )
        with gr.Column(scale=1):
            metrics_output = gr.Markdown()

    # Connect button
    predict_btn.click(
        fn=predict_and_explain,
        inputs=[
            age_input,
            gender_input,
            hypertension_input,
            heart_disease_input,
            smoking_input,
            bmi_input,
            hba1c_input,
            glucose_input,
        ],
        outputs=[
            result_output,
            confidence_output,
            proba_output,
            contribution_plot,
            explanation_output,
        ],
    )

    # Load static image + metrics on startup
    def load_static():
        return feature_importance_img, metrics_md

    demo.load(
        fn=load_static,
        inputs=None,
        outputs=[static_heatmap, metrics_output],
    )

demo.launch()
