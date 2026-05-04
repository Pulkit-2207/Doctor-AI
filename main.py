from flask import Flask, render_template, request, send_file, session, redirect, url_for
from flask_cors import CORS
from chatbot import chatbot_bp
import pandas as pd
import numpy as np
import logging
import os, re, uuid

import matplotlib
matplotlib.use("Agg")  # use non-interactive backend
import matplotlib.pyplot as plt
from io import BytesIO
from xhtml2pdf import pisa

from app_models import decode_label
from ensemble_utils import soft_vote
from xai_utils import shap_explain_instance

# -------------------------------
# Flask setup
# -------------------------------
app = Flask(__name__, template_folder="templates", static_folder="static")
app.secret_key = "83b12e51de3d4d371eb9352fd316245498969f9e7d39c3e02cc6c55b2e521c1f"
CORS(app)
app.register_blueprint(chatbot_bp)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# -------------------------------
# Data files
# -------------------------------
DATA_FILES = {
    'symptoms': r"C:\Users\yashd\OneDrive\Desktop\Doctor AI\data\symtoms_df.csv",
    'precautions': r"C:\Users\yashd\OneDrive\Desktop\Doctor AI\data\precautions_df.csv",
    'workout': r"C:\Users\yashd\OneDrive\Desktop\Doctor AI\data\workout_df.csv",
    'description': r"C:\Users\yashd\OneDrive\Desktop\Doctor AI\data\description.csv",
    'medications': r"C:\Users\yashd\OneDrive\Desktop\Doctor AI\data\medications.csv",
    'diets': r"C:\Users\yashd\OneDrive\Desktop\Doctor AI\data\diets.csv"
}

for path in DATA_FILES.values():
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing data file: {path}")

def load_data():
    data = {}
    for key, path in DATA_FILES.items():
        df = pd.read_csv(path)
        df.columns = df.columns.str.strip().str.lower()
        data[key] = df
        logger.info(f"Loaded {key} ({len(df)} rows)")
    return data

data = load_data()

# -------------------------------
# Symptom dictionary (same as before)
# -------------------------------
SYMPTOMS_DICT = {
    'itching': 0, 'skin_rash': 1, 'nodal_skin_eruptions': 2, 'continuous_sneezing': 3,
    'shivering': 4, 'chills': 5, 'joint_pain': 6, 'stomach_pain': 7, 'acidity': 8,
    'ulcers_on_tongue': 9, 'muscle_wasting': 10, 'vomiting': 11, 'burning_micturition': 12,
    'spotting_ urination': 13, 'fatigue': 14, 'weight_gain': 15, 'anxiety': 16,
    'cold_hands_and_feets': 17, 'mood_swings': 18, 'weight_loss': 19, 'restlessness': 20,
    'lethargy': 21, 'patches_in_throat': 22, 'irregular_sugar_level': 23, 'cough': 24,
    'high_fever': 25, 'sunken_eyes': 26, 'breathlessness': 27, 'sweating': 28,
    'dehydration': 29, 'indigestion': 30, 'headache': 31, 'yellowish_skin': 32,
    'dark_urine': 33, 'nausea': 34, 'loss_of_appetite': 35, 'pain_behind_the_eyes': 36,
    'back_pain': 37, 'constipation': 38, 'abdominal_pain': 39, 'diarrhoea': 40,
    'mild_fever': 41, 'yellow_urine': 42, 'yellowing_of_eyes': 43, 'acute_liver_failure': 44,
    'fluid_overload': 45, 'swelling_of_stomach': 46, 'swelled_lymph_nodes': 47, 'malaise': 48,
    'blurred_and_distorted_vision': 49, 'phlegm': 50, 'throat_irritation': 51,
    'redness_of_eyes': 52, 'sinus_pressure': 53, 'runny_nose': 54, 'congestion': 55,
    'chest_pain': 56, 'weakness_in_limbs': 57, 'fast_heart_rate': 58, 'pain_during_bowel_movements': 59,
    'pain_in_anal_region': 60, 'bloody_stool': 61, 'irritation_in_anus': 62, 'neck_pain': 63,
    'dizziness': 64, 'cramps': 65, 'bruising': 66, 'obesity': 67, 'swollen_legs': 68,
    'swollen_blood_vessels': 69, 'puffy_face_and_eyes': 70, 'enlarged_thyroid': 71,
    'brittle_nails': 72, 'swollen_extremeties': 73, 'excessive_hunger': 74,
    'extra_marital_contacts': 75, 'drying_and_tingling_lips': 76, 'slurred_speech': 77,
    'knee_pain': 78, 'hip_joint_pain': 79, 'muscle_weakness': 80, 'stiff_neck': 81,
    'swelling_joints': 82, 'movement_stiffness': 83, 'spinning_movements': 84,
    'loss_of_balance': 85, 'unsteadiness': 86, 'weakness_of_one_body_side': 87,
    'loss_of_smell': 88, 'bladder_discomfort': 89, 'foul_smell_of urine': 90,
    'continuous_feel_of_urine': 91, 'passage_of_gases': 92, 'internal_itching': 93,
    'toxic_look_(typhos)': 94, 'depression': 95, 'irritability': 96, 'muscle_pain': 97,
    'altered_sensorium': 98, 'red_spots_over_body': 99, 'belly_pain': 100,
    'abnormal_menstruation': 101, 'dischromic _patches': 102, 'watering_from_eyes': 103,
    'increased_appetite': 104, 'polyuria': 105, 'family_history': 106, 'mucoid_sputum': 107,
    'rusty_sputum': 108, 'lack_of_concentration': 109, 'visual_disturbances': 110,
    'receiving_blood_transfusion': 111, 'receiving_unsterile_injections': 112, 'coma': 113,
    'stomach_bleeding': 114, 'distention_of_abdomen': 115, 'history_of_alcohol_consumption': 116,
    'fluid_overload.1': 117, 'blood_in_sputum': 118, 'prominent_veins_on_calf': 119,
    'palpitations': 120, 'painful_walking': 121, 'pus_filled_pimples': 122, 'blackheads': 123,
    'scurring': 124, 'skin_peeling': 125, 'silver_like_dusting': 126, 'small_dents_in_nails': 127,
    'inflammatory_nails': 128, 'blister': 129, 'red_sore_around_nose': 130, 'yellow_crust_ooze': 131
}

# -------------------------------
# Helpers
# -------------------------------
def sanitize_input(text: str) -> str:
    return re.sub(r"[^\w\s,-]", "", text).strip().lower()

def helper(disease: str):
    disease = sanitize_input(disease)
    if not disease:
        return ("Unknown condition", [], [], [], [])

    desc_df = data["description"][data["description"]["disease"].str.strip().str.lower() == disease]
    desc = " ".join(desc_df["description"].astype(str)) if not desc_df.empty else "No description available"

    pre_df = data["precautions"][data["precautions"]["disease"].str.strip().str.lower() == disease]
    pre = pre_df.iloc[0][["precaution_1", "precaution_2", "precaution_3", "precaution_4"]].dropna().tolist() if not pre_df.empty else []

    med_df = data["medications"][data["medications"]["disease"].str.strip().str.lower() == disease]
    med = med_df["medication"].dropna().tolist() if not med_df.empty else []

    diet_df = data["diets"][data["diets"]["disease"].str.strip().str.lower() == disease]
    diet = diet_df["diet"].dropna().tolist() if not diet_df.empty else []

    workout_df = data["workout"][data["workout"]["disease"].str.strip().str.lower() == disease]
    workout = workout_df["workout"].dropna().tolist() if not workout_df.empty else []

    return desc, pre, med, diet, workout

def generate_pdf(predicted_disease, symptoms, description, precautions, medications, diet, workout):
    html = render_template(
        "report.html",
        prediction=predicted_disease,
        symptoms=symptoms,
        description=description,
        precautions=precautions or ["No precautions"],
        medications=medications or ["No medications"],
        diet=diet or ["No diet recommendations"],
        workout=workout or ["No workout advice"],
        current_date=pd.Timestamp.now().strftime("%Y-%m-%d"),
        current_time=pd.Timestamp.now().strftime("%H:%M:%S"),
        patient_id=f"PID-{pd.Timestamp.now().strftime('%Y%m%d%H%M%S')}",
        report_id=f"RPT-{pd.Timestamp.now().strftime('%Y%m%d%H%M%S')}",
    )
    pdf = BytesIO()
    pisa.CreatePDF(html, dest=pdf, encoding="UTF-8")
    pdf.seek(0)
    return pdf

# -------------------------------
# Routes
# -------------------------------
@app.route("/")
def index():
    return render_template("index.html", symptoms_list=list(SYMPTOMS_DICT.keys()))

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # 1️⃣ Collect user input
        symptoms = request.form.get("symptoms", "")
        symptom_list = [sanitize_input(s).replace(" ", "_") for s in symptoms.split(",") if s.strip()]

        # 2️⃣ Build full input vector (132 features)
        input_vector = np.zeros(len(SYMPTOMS_DICT), dtype=np.float32)
        for s in symptom_list:
            if s in SYMPTOMS_DICT:
                input_vector[SYMPTOMS_DICT[s]] = 1

        # 3️⃣ Ensure correct shape for model inputs
        input_vector = input_vector.reshape(1, -1).astype(np.float32)

        # 4️⃣ Ensemble prediction
        probs_final, pred_idx, predicted_disease = soft_vote(input_vector, weights=(0.5, 0.5))
        confidence = float(np.max(probs_final))

        # 5️⃣ Prepare SHAP background
        try:
            training_df = pd.read_csv(DATA_FILES["symptoms"])
            X_bg = training_df.drop(columns=[training_df.columns[-1]]).values.astype(np.float32)
        except Exception:
            X_bg = np.zeros((1, len(SYMPTOMS_DICT)), dtype=np.float32)

        bg_for_shap = X_bg[np.random.choice(len(X_bg), min(50, len(X_bg)), replace=False)]

        # 6️⃣ Compute SHAP values
        shap_vals = shap_explain_instance(bg_for_shap, input_vector, weights=(0.5, 0.5), nsamples=200)
        shap_for_pred = np.array(shap_vals).flatten()  # Fixed: do NOT index with pred_idx

        # 7️⃣ SHAP plot (top 6 features)
        topk = 6
        feat_names = list(SYMPTOMS_DICT.keys())
        shap_len = len(shap_for_pred)  
        topk = min(topk, shap_len)
        idx_sorted = np.argsort(np.abs(shap_for_pred))[-topk:][::-1]
        top_feats = [(feat_names[i], float(shap_for_pred[i])) for i in idx_sorted if i < len(feat_names)]

        # 8️⃣ Generate plot
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.barh(range(len(top_feats)), [v for _, v in top_feats])
        ax.set_yticks(range(len(top_feats)))
        ax.set_yticklabels([n for n, _ in top_feats])
        ax.set_xlabel("SHAP value")
        plt.tight_layout()
        shap_img_name = f"static/shap_{uuid.uuid4().hex[:8]}.png"
        fig.savefig(shap_img_name, bbox_inches="tight")
        plt.close("all")

        # 9️⃣ Recommendations
        description, precautions, medications, diet, workout = helper(predicted_disease)

        # 🔟 Save prediction data to session
        session["prediction_data"] = {
            "predicted_disease": predicted_disease,
            "symptoms": symptoms,
            "confidence": confidence,
            "shap_image": shap_img_name,
            "shap_top_features": top_feats,
            "description": description,
            "precautions": precautions,
            "medications": medications,
            "diet": diet,
            "workout": workout,
        }

        # 1️⃣1️⃣ Render template
        return render_template(
            "index.html",
            predicted_disease=predicted_disease,
            dis_des=description,
            my_precautions=precautions,
            medications=medications,
            my_diet=diet,
            workout=workout,
            symptoms_list=list(SYMPTOMS_DICT.keys()),
            shap_image=shap_img_name,
            shap_top_features=top_feats,
            confidence=confidence,
        )

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}", exc_info=True)
        return render_template("error.html", message="Prediction failed"), 500


@app.route("/download_pdf")
def download_pdf():
    if "prediction_data" not in session:
        return redirect(url_for("index"))
    data = session["prediction_data"]
    pdf = generate_pdf(data["predicted_disease"], data["symptoms"], data["description"],
                       data["precautions"], data["medications"], data["diet"], data["workout"])
    return send_file(pdf, as_attachment=True,
                     download_name=f"medical_report_{pd.Timestamp.now().strftime('%Y%m%d')}.pdf",
                     mimetype="application/pdf")

# Static pages
@app.route("/chatbot")
def chatbot(): return render_template("chatbot.html")

@app.route("/about")
def about(): return render_template("about.html")

@app.route("/contact")
def contact(): return render_template("contact.html")

@app.route("/developer")
def developer(): return render_template("developer.html")

@app.route("/blog")
def blog(): return render_template("blog.html")

@app.errorhandler(404)
def page_not_found(e):
    return render_template("error.html", message="Page not found", error_code="404"), 404

@app.errorhandler(500)
def internal_server_error(e):
    return render_template("error.html", message="Internal server error", error_code="500"), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
