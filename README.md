# EduProphet AI

A single dashboard that "casts a reading" on a student's academic trajectory:
**dropout risk, placement probability, salary forecast, and career fit** —
all from one profile, updating live as you move the dials.

This is a fully working, self-contained demo built on synthetic (but
realistically-correlated) data — trained models, a Flask API, and a
dark "night-almanac" themed dashboard.

## What makes it different from a typical dropout-prediction project

Most student-risk projects stop at "here's a risk score." EduProphet AI adds:

1. **One composite "Prophet Score"** (0–100) that blends dropout risk,
   placement probability, and salary potential into a single trajectory
   reading, with a plain-English narrative sentence.
2. **Personalized explainability without SHAP/LIME dependencies** — a
   lightweight `importance × standardized-deviation` contribution engine
   (see `_contributions()` in `app.py`) that tells you *why* a specific
   student's score looks the way it does, not just a global feature ranking.
3. **Live what-if simulator** — every slider re-runs all three models with
   ~350ms debounce, so a counselor can see in real time how raising
   attendance or adding an internship shifts risk, placement, and salary
   together.
4. **Career-fit engine with skill-gap radar** — cosine similarity between
   a 5-axis student skill vector and 9 career archetype profiles
   (Software Engineer, Data Scientist, PM, Analyst, Core Engineering,
   UI/UX, Higher Studies, Government exams, Entrepreneurship), plus a
   radar chart showing exactly which skills to close the gap on for the
   top-recommended path.
5. **Placement-conditioned salary model** — salary is predicted from a
   model trained only on placed students, then shown as a confidence
   range (conservative/predicted/optimistic), not a single misleading number.
6. **One-click downloadable report** (plain-text) that a counselor can
   attach to a student file, generated client-side from the live reading.

## Project structure

```
eduprophet-ai/
├── app.py                     Flask backend (API + serves the dashboard)
├── requirements.txt
├── models/
│   ├── train_models.py        Synthetic data generation + model training
│   ├── dropout_model.pkl       (generated on first run)
│   ├── placement_model.pkl     (generated on first run)
│   ├── salary_model.pkl        (generated on first run)
│   ├── feature_stats.json      (generated on first run)
│   └── career_profiles.json    (generated on first run)
├── data/
│   └── students_dataset.csv    (generated synthetic dataset, 4000 rows)
├── templates/
│   └── index.html              Dashboard markup
└── static/
    ├── style.css                "Night-almanac" design system
    └── script.js                Live prediction + chart rendering
```

## Setup

```bash
cd eduprophet-ai
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 1. Train the models (only needed once — artifacts are already included,
#    but re-run any time you want to regenerate the synthetic dataset)
python3 models/train_models.py

# 2. Launch the dashboard
python3 app.py
```

Then open **http://127.0.0.1:5000** in your browser.

## API reference

| Endpoint         | Method | Body                          | Returns                                              |
|------------------|--------|--------------------------------|-------------------------------------------------------|
| `/api/predict`   | POST   | student profile JSON          | dropout risk, placement probability, salary forecast, Prophet Score, top contributing factors |
| `/api/careers`   | POST   | student profile JSON          | top-3 career matches with fit % and skill-gap vectors |
| `/api/meta`      | GET    | —                              | model AUC/MAE metrics + career catalogue              |

## Notes on the data

All data is **synthetically generated** (`models/train_models.py`) using
correlated-but-noisy relationships that mimic real student risk factors
(CGPA, attendance, backlogs, financial stress, coding/communication skill,
internships, etc). Swap in your institution's real (anonymized) dataset by
replacing `generate_dataset()` with a CSV loader — the rest of the
pipeline (training, API, dashboard) works unchanged as long as column
names match `DROPOUT_FEATURES`, `PLACEMENT_FEATURES`, and `SALARY_FEATURES`
in `train_models.py` / `app.py`.

## Disclaimer

This is a demonstration project built on synthetic data. It is **not**
validated for real institutional decision-making and should not be used
to make actual academic, admissions, or employment decisions without
proper fairness auditing, consent processes, and domain-expert review.
