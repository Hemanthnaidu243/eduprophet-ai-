"""
EduProphet AI — Model Training Pipeline
----------------------------------------
Generates a realistic synthetic student dataset and trains three models:
  1. Dropout Risk Classifier      (RandomForestClassifier)
  2. Placement Probability Model  (RandomForestClassifier)
  3. Salary Forecast Model        (RandomForestRegressor)

Also builds a career-profile bank used by the content-based
career-recommendation + skill-gap engine.

Run:  python train_models.py
"""

import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_absolute_error, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

RNG = np.random.default_rng(42)
N = 4000

HERE = os.path.dirname(os.path.abspath(__file__))


def generate_dataset(n=N):
    cgpa = np.clip(RNG.normal(7.0, 1.3, n), 3.5, 10.0)
    attendance = np.clip(RNG.normal(78, 14, n), 30, 100)
    backlogs = np.clip(RNG.poisson(1.0, n) - (cgpa - 6).clip(min=0).astype(int), 0, 8)
    study_hours = np.clip(RNG.normal(3.2, 1.6, n), 0, 10)
    family_income = RNG.choice([1, 2, 3, 4], size=n, p=[0.30, 0.35, 0.25, 0.10])  # 1=low .. 4=high
    extracurricular = np.clip(RNG.normal(5.5, 2.2, n), 0, 10)
    internships = RNG.poisson(0.6, n).clip(0, 4)
    coding_skill = np.clip(RNG.normal(5.5, 2.3, n), 0, 10)
    communication_skill = np.clip(RNG.normal(5.8, 2.0, n), 0, 10)
    projects = RNG.poisson(1.8, n).clip(0, 8)
    certifications = RNG.poisson(1.0, n).clip(0, 6)
    financial_stress = np.clip(RNG.normal(5, 2.3, n) - (family_income - 2), 0, 10)
    lms_engagement = np.clip(RNG.normal(60, 20, n), 0, 100)  # % of LMS activities completed
    mentor_sessions = RNG.poisson(1.2, n).clip(0, 10)

    # ---- Dropout risk (latent score -> probability) --------------------
    dropout_logit = (
        -0.55 * (cgpa - 7)
        - 0.045 * (attendance - 75)
        + 0.42 * backlogs
        - 0.25 * study_hours
        + 0.30 * financial_stress
        - 0.25 * (family_income - 2)
        - 0.02 * (lms_engagement - 60)
        - 0.15 * mentor_sessions
        + RNG.normal(0, 0.9, n)
    )
    dropout_prob_true = 1 / (1 + np.exp(-dropout_logit))
    dropout = (RNG.uniform(0, 1, n) < dropout_prob_true).astype(int)

    # ---- Placement probability ------------------------------------------
    placement_logit = (
        0.55 * (cgpa - 6.5)
        - 0.55 * backlogs
        + 0.30 * coding_skill
        + 0.22 * communication_skill
        + 0.55 * internships
        + 0.28 * projects
        + 0.22 * certifications
        + 0.015 * (attendance - 75)
        - 0.9 * dropout  # at-risk students rarely get placed
        + RNG.normal(0, 1.0, n)
    )
    placement_prob_true = 1 / (1 + np.exp(-placement_logit))
    placed = (RNG.uniform(0, 1, n) < placement_prob_true).astype(int)

    # ---- Salary (LPA - lakhs per annum) only meaningful if placed --------
    base_salary = (
        3.0
        + 0.55 * (cgpa - 6.5)
        + 0.42 * coding_skill
        + 0.20 * communication_skill
        + 0.65 * internships
        + 0.30 * projects
        + 0.25 * certifications
        + RNG.normal(0, 1.1, n)
    )
    salary = np.clip(base_salary, 2.2, 45)

    df = pd.DataFrame(
        {
            "cgpa": cgpa.round(2),
            "attendance": attendance.round(1),
            "backlogs": backlogs,
            "study_hours": study_hours.round(1),
            "family_income": family_income,
            "extracurricular": extracurricular.round(1),
            "internships": internships,
            "coding_skill": coding_skill.round(1),
            "communication_skill": communication_skill.round(1),
            "projects": projects,
            "certifications": certifications,
            "financial_stress": financial_stress.round(1),
            "lms_engagement": lms_engagement.round(1),
            "mentor_sessions": mentor_sessions,
            "dropout": dropout,
            "placed": placed,
            "salary_lpa": salary.round(2),
        }
    )
    return df


DROPOUT_FEATURES = [
    "cgpa", "attendance", "backlogs", "study_hours", "family_income",
    "extracurricular", "financial_stress", "lms_engagement", "mentor_sessions",
]
PLACEMENT_FEATURES = [
    "cgpa", "backlogs", "coding_skill", "communication_skill", "internships",
    "projects", "certifications", "attendance",
]
SALARY_FEATURES = [
    "cgpa", "coding_skill", "communication_skill", "internships",
    "projects", "certifications",
]


def train_and_save():
    df = generate_dataset()
    df.to_csv(os.path.join(HERE, "..", "data", "students_dataset.csv"), index=False)

    # ---------------- Dropout model ----------------
    X = df[DROPOUT_FEATURES]
    y = df["dropout"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    dropout_scaler = StandardScaler().fit(X_train)
    dropout_model = RandomForestClassifier(n_estimators=300, max_depth=8, random_state=42, class_weight="balanced")
    dropout_model.fit(X_train, y_train)
    dropout_auc = roc_auc_score(y_test, dropout_model.predict_proba(X_test)[:, 1])
    print(f"[Dropout Model]   AUC = {dropout_auc:.3f}")

    # ---------------- Placement model ----------------
    X = df[PLACEMENT_FEATURES]
    y = df["placed"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    placement_model = RandomForestClassifier(n_estimators=300, max_depth=8, random_state=42, class_weight="balanced")
    placement_model.fit(X_train, y_train)
    placement_auc = roc_auc_score(y_test, placement_model.predict_proba(X_test)[:, 1])
    print(f"[Placement Model] AUC = {placement_auc:.3f}")

    # ---------------- Salary model (only placed students) ----------------
    placed_df = df[df["placed"] == 1]
    X = placed_df[SALARY_FEATURES]
    y = placed_df["salary_lpa"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    salary_model = RandomForestRegressor(n_estimators=300, max_depth=8, random_state=42)
    salary_model.fit(X_train, y_train)
    salary_mae = mean_absolute_error(y_test, salary_model.predict(X_test))
    print(f"[Salary Model]    MAE  = {salary_mae:.2f} LPA")

    # ---------------- Save artifacts ----------------
    joblib.dump(dropout_model, os.path.join(HERE, "dropout_model.pkl"))
    joblib.dump(placement_model, os.path.join(HERE, "placement_model.pkl"))
    joblib.dump(salary_model, os.path.join(HERE, "salary_model.pkl"))

    # feature means/stds used for personalized "contribution" explanations
    stats = {
        "dropout_features": DROPOUT_FEATURES,
        "placement_features": PLACEMENT_FEATURES,
        "salary_features": SALARY_FEATURES,
        "means": df[DROPOUT_FEATURES + list(set(PLACEMENT_FEATURES + SALARY_FEATURES) - set(DROPOUT_FEATURES))].mean().to_dict(),
        "stds": df[DROPOUT_FEATURES + list(set(PLACEMENT_FEATURES + SALARY_FEATURES) - set(DROPOUT_FEATURES))].std().to_dict(),
        "metrics": {
            "dropout_auc": round(dropout_auc, 3),
            "placement_auc": round(placement_auc, 3),
            "salary_mae": round(salary_mae, 2),
            "trained_on_rows": int(len(df)),
        },
    }
    with open(os.path.join(HERE, "feature_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)

    print("\nAll models + feature_stats.json saved to /models")


# ----------------------------------------------------------------------
# Career profile bank (content-based recommendation + skill-gap engine)
# Each profile: ideal normalized (0-10) skill vector across 5 axes:
#   [coding, communication, analytical, creativity, extracurricular]
# ----------------------------------------------------------------------
CAREER_PROFILES = {
    "Software Development Engineer": {
        "vector": [9.0, 5.5, 7.0, 5.0, 3.5],
        "avg_salary_lpa": 9.5,
        "description": "Builds and ships production software; strongest fit for high coding + analytical skill.",
    },
    "Data Scientist / ML Engineer": {
        "vector": [8.0, 5.5, 9.0, 6.0, 3.0],
        "avg_salary_lpa": 11.0,
        "description": "Model building, statistics and experimentation; needs strong analytical + coding skill.",
    },
    "Product Manager": {
        "vector": [4.5, 9.0, 6.5, 7.0, 6.5],
        "avg_salary_lpa": 12.5,
        "description": "Bridges tech and business; thrives on communication, prioritization and market sense.",
    },
    "Business / Data Analyst": {
        "vector": [5.0, 7.5, 8.0, 4.5, 4.0],
        "avg_salary_lpa": 7.5,
        "description": "Turns data into decisions for stakeholders; needs analytical rigor + clear communication.",
    },
    "Core / Systems Engineer": {
        "vector": [7.0, 4.5, 8.0, 4.0, 3.0],
        "avg_salary_lpa": 7.0,
        "description": "Hardware-adjacent or infrastructure-heavy roles; deep technical depth over breadth.",
    },
    "UI/UX & Product Design": {
        "vector": [4.0, 7.0, 5.0, 9.0, 5.5],
        "avg_salary_lpa": 8.0,
        "description": "Designs usable, delightful interfaces; creativity + communication are the core drivers.",
    },
    "Higher Studies / Research (MS/PhD)": {
        "vector": [6.5, 6.0, 9.5, 6.5, 3.0],
        "avg_salary_lpa": 0.0,
        "description": "Deep specialization via postgraduate research; analytical curiosity is the dominant trait.",
    },
    "Government / Public Sector (Competitive Exams)": {
        "vector": [3.0, 7.0, 7.5, 3.0, 6.0],
        "avg_salary_lpa": 6.0,
        "description": "Stable public-sector careers via competitive exams; consistency + discipline matter most.",
    },
    "Entrepreneurship / Startup Founder": {
        "vector": [6.5, 8.5, 6.5, 8.0, 8.0],
        "avg_salary_lpa": 0.0,
        "description": "Builds something from zero; needs a versatile blend across every axis, especially grit.",
    },
}

if __name__ == "__main__":
    train_and_save()
    with open(os.path.join(HERE, "career_profiles.json"), "w") as f:
        json.dump(CAREER_PROFILES, f, indent=2)
    print("career_profiles.json saved to /models")
