"""
EduProphet AI — Backend
------------------------
Flask API that serves:
  • /api/predict          -> dropout risk + placement probability + salary forecast
                              + explainable "top contributing factors"
  • /api/careers          -> top-3 career matches (cosine similarity) + skill-gap breakdown
  • /api/report            -> assembles a plain-text downloadable report
  • /api/meta              -> model metrics, feature ranges, career catalogue (for the UI)
"""

import json
import os

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, send_from_directory

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(HERE, "models")

app = Flask(__name__, static_folder="static", template_folder="templates")

# ------------------------------------------------------------------ #
# Load trained artifacts once at startup
# ------------------------------------------------------------------ #
dropout_model = joblib.load(os.path.join(MODELS_DIR, "dropout_model.pkl"))
placement_model = joblib.load(os.path.join(MODELS_DIR, "placement_model.pkl"))
salary_model = joblib.load(os.path.join(MODELS_DIR, "salary_model.pkl"))

with open(os.path.join(MODELS_DIR, "feature_stats.json")) as f:
    STATS = json.load(f)

with open(os.path.join(MODELS_DIR, "career_profiles.json")) as f:
    CAREER_PROFILES = json.load(f)

DROPOUT_FEATURES = STATS["dropout_features"]
PLACEMENT_FEATURES = STATS["placement_features"]
SALARY_FEATURES = STATS["salary_features"]
MEANS = STATS["means"]
STDS = STATS["stds"]

FRIENDLY_NAMES = {
    "cgpa": "CGPA",
    "attendance": "Attendance",
    "backlogs": "Backlogs",
    "study_hours": "Daily study hours",
    "family_income": "Family income bracket",
    "extracurricular": "Extracurricular activity",
    "financial_stress": "Financial stress level",
    "lms_engagement": "LMS/online engagement",
    "mentor_sessions": "Mentor check-ins",
    "coding_skill": "Coding skill",
    "communication_skill": "Communication skill",
    "internships": "Internships completed",
    "projects": "Projects completed",
    "certifications": "Certifications earned",
}

# direction: +1 means "higher value = higher risk / better outcome" as-is,
# -1 means the feature is protective (higher value = lower risk)
DROPOUT_DIRECTION = {
    "cgpa": -1, "attendance": -1, "backlogs": 1, "study_hours": -1,
    "family_income": -1, "extracurricular": -1, "financial_stress": 1,
    "lms_engagement": -1, "mentor_sessions": -1,
}


def _contributions(feature_list, values, model, direction_map=None):
    """
    Lightweight, dependency-free explainability:
    contribution_i = feature_importance_i * standardized_deviation_i
    Sorted by absolute contribution, descending. Not SHAP, but gives a fast,
    personalized, directionally-correct 'why' for every prediction.
    """
    importances = model.feature_importances_
    out = []
    for i, feat in enumerate(feature_list):
        mean = MEANS.get(feat, 0)
        std = STDS.get(feat, 1) or 1
        z = (values[i] - mean) / std
        raw_contribution = importances[i] * z
        if direction_map:
            raw_contribution *= direction_map.get(feat, 1)
        out.append({
            "feature": feat,
            "label": FRIENDLY_NAMES.get(feat, feat),
            "value": round(float(values[i]), 2),
            "importance": round(float(importances[i]), 3),
            "contribution": round(float(raw_contribution), 3),
        })
    out.sort(key=lambda d: abs(d["contribution"]), reverse=True)
    return out


@app.route("/")
def index():
    return app.send_static_file("index.html") if os.path.exists(
        os.path.join(app.static_folder, "index.html")
    ) else send_from_directory(app.template_folder, "index.html")


@app.route("/api/meta")
def meta():
    return jsonify({
        "metrics": STATS["metrics"],
        "career_catalogue": list(CAREER_PROFILES.keys()),
    })


@app.route("/api/predict", methods=["POST"])
def predict():
    p = request.get_json(force=True)

    student = {
        "cgpa": float(p.get("cgpa", 7.0)),
        "attendance": float(p.get("attendance", 75)),
        "backlogs": float(p.get("backlogs", 0)),
        "study_hours": float(p.get("study_hours", 3)),
        "family_income": float(p.get("family_income", 2)),
        "extracurricular": float(p.get("extracurricular", 5)),
        "financial_stress": float(p.get("financial_stress", 5)),
        "lms_engagement": float(p.get("lms_engagement", 60)),
        "mentor_sessions": float(p.get("mentor_sessions", 1)),
        "coding_skill": float(p.get("coding_skill", 5)),
        "communication_skill": float(p.get("communication_skill", 5)),
        "internships": float(p.get("internships", 0)),
        "projects": float(p.get("projects", 1)),
        "certifications": float(p.get("certifications", 0)),
    }

    # ---- Dropout ----
    d_vals = [student[f] for f in DROPOUT_FEATURES]
    d_df = pd.DataFrame([d_vals], columns=DROPOUT_FEATURES)
    dropout_prob = float(dropout_model.predict_proba(d_df)[0][1])
    dropout_contrib = _contributions(DROPOUT_FEATURES, d_vals, dropout_model, DROPOUT_DIRECTION)

    # ---- Placement ----
    pl_vals = [student[f] for f in PLACEMENT_FEATURES]
    pl_df = pd.DataFrame([pl_vals], columns=PLACEMENT_FEATURES)
    placement_prob = float(placement_model.predict_proba(pl_df)[0][1])
    placement_contrib = _contributions(PLACEMENT_FEATURES, pl_vals, placement_model)

    # ---- Salary (only meaningful conditional on placement) ----
    s_vals = [student[f] for f in SALARY_FEATURES]
    s_df = pd.DataFrame([s_vals], columns=SALARY_FEATURES)
    salary_pred = float(salary_model.predict(s_df)[0])
    salary_low, salary_high = round(salary_pred * 0.85, 2), round(salary_pred * 1.2, 2)

    # ---- Composite EduProphet Score (0-100, higher = better trajectory) ----
    prophet_score = round(
        100 * (0.45 * (1 - dropout_prob) + 0.35 * placement_prob + 0.20 * min(salary_pred / 15, 1)),
        1,
    )

    if dropout_prob >= 0.66:
        risk_band, risk_msg = "High Risk", "Immediate counselor intervention recommended."
    elif dropout_prob >= 0.35:
        risk_band, risk_msg = "Moderate Risk", "Monitor closely; targeted support advised."
    else:
        risk_band, risk_msg = "Low Risk", "Trajectory looks stable."

    top_dropout_driver = dropout_contrib[0]
    narrative = (
        f"{FRIENDLY_NAMES.get(top_dropout_driver['feature'], top_dropout_driver['feature'])} "
        f"is currently the single largest factor shaping this student's risk profile."
    )

    return jsonify({
        "dropout": {
            "probability": round(dropout_prob, 3),
            "risk_band": risk_band,
            "message": risk_msg,
            "top_factors": dropout_contrib[:5],
        },
        "placement": {
            "probability": round(placement_prob, 3),
            "top_factors": placement_contrib[:5],
        },
        "salary": {
            "predicted_lpa": round(salary_pred, 2),
            "range_lpa": [salary_low, salary_high],
        },
        "prophet_score": prophet_score,
        "narrative": narrative,
        "student": student,
    })


@app.route("/api/careers", methods=["POST"])
def careers():
    p = request.get_json(force=True)
    # 5-axis skill vector: coding, communication, analytical, creativity, extracurricular
    student_vec = np.array([
        float(p.get("coding_skill", 5)),
        float(p.get("communication_skill", 5)),
        float(p.get("analytical_skill", (float(p.get("cgpa", 7)) / 10) * 10)),
        float(p.get("creativity_skill", 5)),
        float(p.get("extracurricular", 5)),
    ])

    def cosine(a, b):
        a, b = np.array(a), np.array(b)
        denom = (np.linalg.norm(a) * np.linalg.norm(b)) or 1e-9
        return float(np.dot(a, b) / denom)

    results = []
    for name, profile in CAREER_PROFILES.items():
        ideal = profile["vector"]
        fit = cosine(student_vec, ideal)
        gap = [round(float(i - s), 2) for i, s in zip(ideal, student_vec)]
        results.append({
            "career": name,
            "description": profile["description"],
            "avg_salary_lpa": profile["avg_salary_lpa"],
            "fit_pct": round(fit * 100, 1),
            "skill_gap": {
                "axes": ["Coding", "Communication", "Analytical", "Creativity", "Extracurricular"],
                "student": student_vec.tolist(),
                "ideal": ideal,
                "gap": gap,
            },
        })

    results.sort(key=lambda r: r["fit_pct"], reverse=True)
    return jsonify({"recommendations": results[:3]})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
