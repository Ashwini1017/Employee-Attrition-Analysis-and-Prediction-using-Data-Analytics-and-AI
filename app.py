"""
Employee Attrition Analysis & AI Prediction Web Application
Flask backend providing real-time model inference, EDA API, and dynamic dataset exploration.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template_string, request

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "dataset" / "employee_data.csv"

# Global data and model container
DATA_STORE = {
    "df": None,
    "model": None,
    "model_type": "LogisticRegression",
    "feature_names": [],
    "metrics": {}
}


def load_and_preprocess_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    df = df.drop_duplicates().copy()

    numeric_cols = [
        "Age", "MonthlyIncome", "YearsAtCompany", "JobSatisfaction",
        "WorkLifeBalance", "DistanceFromHomeKm", "TrainingHours",
        "NumCompaniesWorked"
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    categorical_cols = ["Department", "JobLevel", "OverTime", "Attrition"]
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].mode()[0])

    DATA_STORE["df"] = df
    train_model("LogisticRegression")
    return df


def train_model(model_type="LogisticRegression"):
    df = DATA_STORE["df"]
    if df is None:
        return

    X = df.drop(columns=["Attrition"])
    y = df["Attrition"].map({"No": 0, "Yes": 1})

    categorical_features = ["Department", "JobLevel", "OverTime"]
    numeric_features = [c for c in X.columns if c not in categorical_features]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ]), numeric_features),
            ("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore"))
            ]), categorical_features)
        ]
    )

    if model_type == "RandomForest":
        clf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
    else:
        clf = LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")

    model_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", clf)
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    model_pipeline.fit(X_train, y_train)
    y_pred = model_pipeline.predict(X_test)
    y_prob = model_pipeline.predict_proba(X_test)[:, 1] if hasattr(model_pipeline, "predict_proba") else y_pred

    acc = accuracy_score(y_test, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="binary")
    cm = confusion_matrix(y_test, y_pred).tolist()

    # Extract feature names after one-hot encoding
    ohe = model_pipeline.named_steps["preprocessor"].named_transformers_["cat"].named_steps["onehot"]
    cat_cols_encoded = list(ohe.get_feature_names_out(categorical_features))
    all_feature_names = numeric_features + cat_cols_encoded

    # Extract coefficients / feature importances
    classifier = model_pipeline.named_steps["classifier"]
    if hasattr(classifier, "coef_"):
        importances = classifier.coef_[0].tolist()
    elif hasattr(classifier, "feature_importances_"):
        importances = classifier.feature_importances_.tolist()
    else:
        importances = [0] * len(all_feature_names)

    feature_importance_list = sorted(
        [{"feature": f, "importance": round(imp, 4)} for f, imp in zip(all_feature_names, importances)],
        key=lambda x: abs(x["importance"]),
        reverse=True
    )

    DATA_STORE["model"] = model_pipeline
    DATA_STORE["model_type"] = model_type
    DATA_STORE["feature_names"] = all_feature_names
    DATA_STORE["metrics"] = {
        "accuracy": round(acc * 100, 2),
        "precision": round(prec * 100, 2),
        "recall": round(rec * 100, 2),
        "f1": round(f1 * 100, 2),
        "confusion_matrix": cm,
        "feature_importances": feature_importance_list[:10]
    }


# Initial data load
load_and_preprocess_data()


@app.route("/api/stats", methods=["GET"])
def get_stats():
    df = DATA_STORE["df"]
    if df is None:
        return jsonify({"error": "Data not loaded"}), 500

    total_employees = int(len(df))
    attrition_yes = int((df["Attrition"] == "Yes").sum())
    attrition_rate = round((attrition_yes / total_employees) * 100, 2)
    avg_income = round(float(df["MonthlyIncome"].mean()), 2)
    avg_satisfaction = round(float(df["JobSatisfaction"].mean()), 2)
    avg_worklife = round(float(df["WorkLifeBalance"].mean()), 2)

    # Department Attrition Breakdown
    dept_stats = []
    for dept, group in df.groupby("Department"):
        tot = len(group)
        att = (group["Attrition"] == "Yes").sum()
        dept_stats.append({
            "department": dept,
            "total": int(tot),
            "attrition": int(att),
            "rate": round((att / tot) * 100, 1)
        })

    # Overtime Breakdown
    overtime_stats = []
    for ot, group in df.groupby("OverTime"):
        tot = len(group)
        att = (group["Attrition"] == "Yes").sum()
        overtime_stats.append({
            "overtime": ot,
            "total": int(tot),
            "attrition": int(att),
            "rate": round((att / tot) * 100, 1)
        })

    # Job Satisfaction Breakdown
    sat_stats = []
    for sat, group in df.groupby("JobSatisfaction"):
        tot = len(group)
        att = (group["Attrition"] == "Yes").sum()
        sat_stats.append({
            "satisfaction": int(sat),
            "total": int(tot),
            "attrition": int(att),
            "rate": round((att / tot) * 100, 1)
        })

    # Job Level Breakdown
    level_stats = []
    for lvl, group in df.groupby("JobLevel"):
        tot = len(group)
        att = (group["Attrition"] == "Yes").sum()
        level_stats.append({
            "job_level": lvl,
            "total": int(tot),
            "attrition": int(att),
            "rate": round((att / tot) * 100, 1)
        })

    return jsonify({
        "total_employees": total_employees,
        "attrition_count": attrition_yes,
        "attrition_rate": attrition_rate,
        "avg_income": avg_income,
        "avg_satisfaction": avg_satisfaction,
        "avg_worklife": avg_worklife,
        "dept_stats": dept_stats,
        "overtime_stats": overtime_stats,
        "sat_stats": sat_stats,
        "level_stats": level_stats,
        "model_metrics": DATA_STORE["metrics"],
        "model_type": DATA_STORE["model_type"]
    })


@app.route("/api/dataset", methods=["GET"])
def get_dataset():
    df = DATA_STORE["df"]
    if df is None:
        return jsonify([])

    search = request.args.get("search", "").lower()
    dept = request.args.get("dept", "")
    attrition = request.args.get("attrition", "")
    limit = int(request.args.get("limit", 100))

    filtered_df = df.copy()

    if dept:
        filtered_df = filtered_df[filtered_df["Department"] == dept]
    if attrition:
        filtered_df = filtered_df[filtered_df["Attrition"] == attrition]
    if search:
        filtered_df = filtered_df[
            filtered_df["Department"].str.lower().str.contains(search) |
            filtered_df["JobLevel"].str.lower().str.contains(search) |
            filtered_df["OverTime"].str.lower().str.contains(search)
        ]

    records = filtered_df.head(limit).to_dict(orient="records")
    return jsonify({
        "total_filtered": len(filtered_df),
        "records": records
    })


@app.route("/api/predict", methods=["POST"])
def predict():
    model = DATA_STORE["model"]
    if model is None:
        return jsonify({"error": "Model not trained"}), 500

    data = request.json or {}
    try:
        input_data = pd.DataFrame([{
            "Age": float(data.get("Age", 35)),
            "Department": str(data.get("Department", "Research & Development")),
            "JobLevel": str(data.get("JobLevel", "Mid")),
            "MonthlyIncome": float(data.get("MonthlyIncome", 4500)),
            "YearsAtCompany": float(data.get("YearsAtCompany", 4)),
            "JobSatisfaction": float(data.get("JobSatisfaction", 3)),
            "WorkLifeBalance": float(data.get("WorkLifeBalance", 3)),
            "OverTime": str(data.get("OverTime", "No")),
            "DistanceFromHomeKm": float(data.get("DistanceFromHomeKm", 10)),
            "TrainingHours": float(data.get("TrainingHours", 40)),
            "NumCompaniesWorked": float(data.get("NumCompaniesWorked", 2))
        }])

        prob = float(model.predict_proba(input_data)[0][1])
        prediction = int(model.predict(input_data)[0])

        risk_score = round(prob * 100, 1)
        if risk_score > 65:
            risk_level = "High Risk"
            badge_color = "danger"
        elif risk_score > 35:
            risk_level = "Moderate Risk"
            badge_color = "warning"
        else:
            risk_level = "Low Risk"
            badge_color = "success"

        # Key drivers
        drivers = []
        if data.get("OverTime") == "Yes":
            drivers.append("Frequent Overtime increases burnout risk significantly.")
        if float(data.get("JobSatisfaction", 3)) <= 2:
            drivers.append("Low Job Satisfaction rating (<= 2).")
        if float(data.get("WorkLifeBalance", 3)) <= 2:
            drivers.append("Low Work-Life Balance score (<= 2).")
        if float(data.get("MonthlyIncome", 4500)) < 3000:
            drivers.append("Below-average Monthly Income for job level.")
        if float(data.get("DistanceFromHomeKm", 10)) > 20:
            drivers.append("Long commute distance (> 20 km).")

        if not drivers:
            drivers.append("Balanced workload, competitive compensation, and high job satisfaction.")

        # Actionable recommendations
        recs = []
        if data.get("OverTime") == "Yes":
            recs.append("Review workload distribution and reduce mandatory overtime.")
        if float(data.get("JobSatisfaction", 3)) <= 2:
            recs.append("Schedule a 1-on-1 feedback session to explore career pathing.")
        if float(data.get("WorkLifeBalance", 3)) <= 2:
            recs.append("Offer flexible working hours or remote work options.")
        if float(data.get("MonthlyIncome", 4500)) < 3000:
            recs.append("Conduct a market salary benchmark review.")

        if not recs:
            recs.append("Continue current engagement and recognition strategies.")

        return jsonify({
            "prediction": "Yes" if prediction == 1 else "No",
            "probability": risk_score,
            "risk_level": risk_level,
            "badge_color": badge_color,
            "key_drivers": drivers,
            "recommendations": recs
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/api/retrain", methods=["POST"])
def retrain():
    data = request.json or {}
    model_type = data.get("model_type", "LogisticRegression")
    train_model(model_type)
    return jsonify({
        "message": f"Model retrained using {model_type}",
        "metrics": DATA_STORE["metrics"],
        "model_type": model_type
    })


@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)


HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Employee Attrition Analysis & AI Predictor</title>
    <!-- Google Fonts & Font Awesome -->
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <!-- Chart.js -->
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        :root {
            --bg-dark: #0f172a;
            --bg-card: rgba(30, 41, 59, 0.7);
            --bg-card-border: rgba(255, 255, 255, 0.1);
            --accent-primary: #6366f1;
            --accent-secondary: #06b6d4;
            --accent-pink: #ec4899;
            --accent-success: #10b981;
            --accent-warning: #f59e0b;
            --accent-danger: #ef4444;
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: 'Inter', sans-serif;
        }

        body {
            background-color: var(--bg-dark);
            color: var(--text-main);
            min-height: 100vh;
            background-image: 
                radial-gradient(circle at 15% 15%, rgba(99, 102, 241, 0.15) 0%, transparent 40%),
                radial-gradient(circle at 85% 85%, rgba(6, 182, 212, 0.15) 0%, transparent 40%);
            background-attachment: fixed;
        }

        /* Top Navbar */
        header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 1.25rem 2.5rem;
            background: rgba(15, 23, 42, 0.85);
            backdrop-filter: blur(12px);
            border-bottom: 1px solid var(--bg-card-border);
            position: sticky;
            top: 0;
            z-index: 100;
        }

        .brand {
            display: flex;
            align-items: center;
            gap: 0.8rem;
            font-size: 1.35rem;
            font-weight: 700;
            background: linear-gradient(135deg, #6366f1, #06b6d4);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .brand i {
            -webkit-text-fill-color: initial;
            color: #6366f1;
            font-size: 1.5rem;
        }

        nav {
            display: flex;
            gap: 1rem;
        }

        .nav-btn {
            background: transparent;
            border: 1px solid transparent;
            color: var(--text-muted);
            padding: 0.6rem 1.2rem;
            border-radius: 8px;
            font-weight: 500;
            cursor: pointer;
            transition: all 0.3s ease;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        .nav-btn.active, .nav-btn:hover {
            color: var(--text-main);
            background: rgba(99, 102, 241, 0.15);
            border-color: rgba(99, 102, 241, 0.4);
        }

        /* Layout */
        .container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 2rem;
        }

        .tab-content {
            display: none;
            animation: fadeIn 0.4s ease-in-out;
        }

        .tab-content.active {
            display: block;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }

        /* Stats KPI Grid */
        .kpi-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 1.25rem;
            margin-bottom: 2rem;
        }

        .kpi-card {
            background: var(--bg-card);
            backdrop-filter: blur(16px);
            border: 1px solid var(--bg-card-border);
            border-radius: 14px;
            padding: 1.5rem;
            display: flex;
            align-items: center;
            gap: 1.25rem;
            box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3);
            transition: transform 0.3s ease;
        }

        .kpi-card:hover {
            transform: translateY(-4px);
        }

        .kpi-icon {
            width: 54px;
            height: 54px;
            border-radius: 12px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.5rem;
        }

        .kpi-icon.indigo { background: rgba(99, 102, 241, 0.2); color: #818cf8; }
        .kpi-icon.pink { background: rgba(236, 72, 153, 0.2); color: #f472b6; }
        .kpi-icon.cyan { background: rgba(6, 182, 212, 0.2); color: #38bdf8; }
        .kpi-icon.emerald { background: rgba(16, 185, 129, 0.2); color: #34d399; }

        .kpi-info h4 {
            font-size: 0.85rem;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 0.3rem;
        }

        .kpi-info p {
            font-size: 1.6rem;
            font-weight: 700;
        }

        /* Charts Section */
        .chart-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(420px, 1fr));
            gap: 1.5rem;
            margin-bottom: 2rem;
        }

        .card {
            background: var(--bg-card);
            backdrop-filter: blur(16px);
            border: 1px solid var(--bg-card-border);
            border-radius: 16px;
            padding: 1.75rem;
            box-shadow: 0 10px 30px -5px rgba(0, 0, 0, 0.3);
        }

        .card-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 1.5rem;
        }

        .card-title {
            font-size: 1.1rem;
            font-weight: 600;
            display: flex;
            align-items: center;
            gap: 0.6rem;
        }

        .card-title i {
            color: var(--accent-secondary);
        }

        .chart-container {
            position: relative;
            height: 300px;
            width: 100%;
        }

        /* Form Controls */
        .form-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 1.25rem;
        }

        .form-group {
            display: flex;
            flex-direction: column;
            gap: 0.5rem;
        }

        .form-group label {
            font-size: 0.85rem;
            font-weight: 500;
            color: var(--text-muted);
        }

        .form-control {
            background: rgba(15, 23, 42, 0.6);
            border: 1px solid var(--bg-card-border);
            border-radius: 8px;
            padding: 0.75rem 1rem;
            color: var(--text-main);
            font-size: 0.95rem;
            outline: none;
            transition: border-color 0.3s ease;
        }

        .form-control:focus {
            border-color: var(--accent-primary);
        }

        .btn-submit {
            grid-column: 1 / -1;
            background: linear-gradient(135deg, var(--accent-primary), var(--accent-secondary));
            color: white;
            border: none;
            padding: 1rem;
            border-radius: 10px;
            font-weight: 600;
            font-size: 1rem;
            cursor: pointer;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(99, 102, 241, 0.4);
            margin-top: 0.5rem;
        }

        .btn-submit:hover {
            opacity: 0.95;
            transform: translateY(-2px);
        }

        /* Result Modal / Box */
        .result-box {
            margin-top: 1.5rem;
            padding: 1.5rem;
            border-radius: 14px;
            background: rgba(15, 23, 42, 0.8);
            border: 1px solid var(--bg-card-border);
            display: none;
        }

        .risk-badge {
            display: inline-block;
            padding: 0.4rem 1rem;
            border-radius: 30px;
            font-size: 0.9rem;
            font-weight: 700;
            margin-top: 0.5rem;
        }

        .badge-danger { background: rgba(239, 68, 68, 0.2); color: #f87171; border: 1px solid #ef4444; }
        .badge-warning { background: rgba(245, 158, 11, 0.2); color: #fbbf24; border: 1px solid #f59e0b; }
        .badge-success { background: rgba(16, 185, 129, 0.2); color: #34d399; border: 1px solid #10b981; }

        .meter-bar {
            width: 100%;
            height: 12px;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 6px;
            overflow: hidden;
            margin: 1rem 0;
        }

        .meter-fill {
            height: 100%;
            width: 0%;
            transition: width 0.8s ease-in-out;
        }

        /* Data Table */
        .table-responsive {
            overflow-x: auto;
        }

        table {
            width: 100%;
            border-collapse: collapse;
            font-size: 0.9rem;
        }

        th, td {
            padding: 0.9rem 1rem;
            text-align: left;
            border-bottom: 1px solid var(--bg-card-border);
        }

        th {
            color: var(--text-muted);
            font-weight: 600;
            background: rgba(15, 23, 42, 0.4);
        }

        tr:hover {
            background: rgba(255, 255, 255, 0.03);
        }

        .pill {
            padding: 0.25rem 0.6rem;
            border-radius: 12px;
            font-size: 0.75rem;
            font-weight: 600;
        }

        .pill-yes { background: rgba(239, 68, 68, 0.2); color: #f87171; }
        .pill-no { background: rgba(16, 185, 129, 0.2); color: #34d399; }

        .search-bar {
            display: flex;
            gap: 1rem;
            margin-bottom: 1.25rem;
        }

        @media (max-width: 768px) {
            header { flex-direction: column; gap: 1rem; padding: 1rem; }
            .chart-grid { grid-template-columns: 1fr; }
            .container { padding: 1rem; }
        }
    </style>
</head>
<body>

    <header>
        <div class="brand">
            <i class="fa-solid fa-brain"></i>
            <span>Attrition Predictor AI</span>
        </div>
        <nav>
            <button class="nav-btn active" onclick="switchTab('dashboard')">
                <i class="fa-solid fa-chart-pie"></i> Executive EDA
            </button>
            <button class="nav-btn" onclick="switchTab('predictor')">
                <i class="fa-solid fa-user-gear"></i> AI Risk Predictor
            </button>
            <button class="nav-btn" onclick="switchTab('explorer')">
                <i class="fa-solid fa-database"></i> Dataset Explorer
            </button>
            <button class="nav-btn" onclick="switchTab('model-info')">
                <i class="fa-solid fa-microchip"></i> Model Performance
            </button>
        </nav>
    </header>

    <div class="container">

        <!-- TAB 1: EXECUTIVE EDA DASHBOARD -->
        <div id="dashboard" class="tab-content active">
            <!-- KPI Cards -->
            <div class="kpi-grid">
                <div class="kpi-card">
                    <div class="kpi-icon indigo"><i class="fa-solid fa-users"></i></div>
                    <div class="kpi-info">
                        <h4>Total Workforce</h4>
                        <p id="kpi-total">1,000</p>
                    </div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-icon pink"><i class="fa-solid fa-user-minus"></i></div>
                    <div class="kpi-info">
                        <h4>Attrition Rate</h4>
                        <p id="kpi-rate">0.0%</p>
                    </div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-icon cyan"><i class="fa-solid fa-dollar-sign"></i></div>
                    <div class="kpi-info">
                        <h4>Avg Monthly Salary</h4>
                        <p id="kpi-salary">$0</p>
                    </div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-icon emerald"><i class="fa-solid fa-smile"></i></div>
                    <div class="kpi-info">
                        <h4>Avg Job Satisfaction</h4>
                        <p id="kpi-satisfaction">0.0 / 4</p>
                    </div>
                </div>
            </div>

            <!-- Chart Grid -->
            <div class="chart-grid">
                <div class="card">
                    <div class="card-header">
                        <div class="card-title"><i class="fa-solid fa-business-time"></i> Attrition by Overtime Status</div>
                    </div>
                    <div class="chart-container">
                        <canvas id="overtimeChart"></canvas>
                    </div>
                </div>
                <div class="card">
                    <div class="card-header">
                        <div class="card-title"><i class="fa-solid fa-building"></i> Attrition Rate by Department</div>
                    </div>
                    <div class="chart-container">
                        <canvas id="deptChart"></canvas>
                    </div>
                </div>
                <div class="card">
                    <div class="card-header">
                        <div class="card-title"><i class="fa-solid fa-heart-pulse"></i> Job Satisfaction Impact</div>
                    </div>
                    <div class="chart-container">
                        <canvas id="satChart"></canvas>
                    </div>
                </div>
                <div class="card">
                    <div class="card-header">
                        <div class="card-title"><i class="fa-solid fa-layer-group"></i> Attrition by Job Level</div>
                    </div>
                    <div class="chart-container">
                        <canvas id="levelChart"></canvas>
                    </div>
                </div>
            </div>
        </div>

        <!-- TAB 2: AI RISK PREDICTOR -->
        <div id="predictor" class="tab-content">
            <div class="card">
                <div class="card-header">
                    <div class="card-title"><i class="fa-solid fa-wand-magic-sparkles"></i> Predict Employee Attrition Probability</div>
                </div>
                <form id="predictForm" onsubmit="handlePredict(event)">
                    <div class="form-grid">
                        <div class="form-group">
                            <label>Age</label>
                            <input type="number" class="form-control" name="Age" value="32" required min="18" max="70">
                        </div>
                        <div class="form-group">
                            <label>Department</label>
                            <select class="form-control" name="Department">
                                <option value="Research & Development">Research & Development</option>
                                <option value="Sales">Sales</option>
                                <option value="Human Resources">Human Resources</option>
                            </select>
                        </div>
                        <div class="form-group">
                            <label>Job Level</label>
                            <select class="form-control" name="JobLevel">
                                <option value="Entry">Entry Level</option>
                                <option value="Mid" selected>Mid Level</option>
                                <option value="Senior">Senior Level</option>
                            </select>
                        </div>
                        <div class="form-group">
                            <label>Monthly Income ($)</label>
                            <input type="number" class="form-control" name="MonthlyIncome" value="3800" step="100">
                        </div>
                        <div class="form-group">
                            <label>Years at Company</label>
                            <input type="number" class="form-control" name="YearsAtCompany" value="3">
                        </div>
                        <div class="form-group">
                            <label>Job Satisfaction (1 - Low to 4 - High)</label>
                            <select class="form-control" name="JobSatisfaction">
                                <option value="1">1 - Low</option>
                                <option value="2">2 - Medium</option>
                                <option value="3" selected>3 - High</option>
                                <option value="4">4 - Very High</option>
                            </select>
                        </div>
                        <div class="form-group">
                            <label>Work-Life Balance (1 - Poor to 4 - Excellent)</label>
                            <select class="form-control" name="WorkLifeBalance">
                                <option value="1">1 - Poor</option>
                                <option value="2">2 - Fair</option>
                                <option value="3" selected>3 - Good</option>
                                <option value="4">4 - Excellent</option>
                            </select>
                        </div>
                        <div class="form-group">
                            <label>Overtime Required?</label>
                            <select class="form-control" name="OverTime">
                                <option value="No">No</option>
                                <option value="Yes" selected>Yes</option>
                            </select>
                        </div>
                        <div class="form-group">
                            <label>Commute Distance (km)</label>
                            <input type="number" class="form-control" name="DistanceFromHomeKm" value="15">
                        </div>
                        <div class="form-group">
                            <label>Training Hours Completed</label>
                            <input type="number" class="form-control" name="TrainingHours" value="35">
                        </div>
                        <div class="form-group">
                            <label>Previous Companies Worked</label>
                            <input type="number" class="form-control" name="NumCompaniesWorked" value="2">
                        </div>
                        <button type="submit" class="btn-submit">
                            <i class="fa-solid fa-microchip"></i> Generate Attrition Risk Assessment
                        </button>
                    </div>
                </form>

                <div id="resultBox" class="result-box">
                    <h3 style="margin-bottom: 0.5rem;">AI Prediction Assessment</h3>
                    <div style="display: flex; align-items: center; justify-content: space-between;">
                        <span>Predicted Attrition Risk Score: <strong id="riskScore">0%</strong></span>
                        <span id="riskBadge" class="risk-badge badge-warning">Moderate Risk</span>
                    </div>

                    <div class="meter-bar">
                        <div id="meterFill" class="meter-fill" style="background: var(--accent-warning);"></div>
                    </div>

                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin-top: 1rem;">
                        <div>
                            <h4 style="color: var(--accent-secondary); margin-bottom: 0.5rem;"><i class="fa-solid fa-triangle-exclamation"></i> Key Risk Factors</h4>
                            <ul id="keyDrivers" style="padding-left: 1.2rem; color: var(--text-muted); font-size: 0.9rem;"></ul>
                        </div>
                        <div>
                            <h4 style="color: var(--accent-success); margin-bottom: 0.5rem;"><i class="fa-solid fa-lightbulb"></i> Recommended HR Interventions</h4>
                            <ul id="recommendations" style="padding-left: 1.2rem; color: var(--text-muted); font-size: 0.9rem;"></ul>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <!-- TAB 3: DATASET EXPLORER -->
        <div id="explorer" class="tab-content">
            <div class="card">
                <div class="card-header">
                    <div class="card-title"><i class="fa-solid fa-table"></i> Employee Records Database</div>
                </div>
                <div class="search-bar">
                    <input type="text" id="searchInput" class="form-control" placeholder="Search by department or role..." onkeyup="fetchRecords()">
                    <select id="deptFilter" class="form-control" style="width: 220px;" onchange="fetchRecords()">
                        <option value="">All Departments</option>
                        <option value="Research & Development">Research & Development</option>
                        <option value="Sales">Sales</option>
                        <option value="Human Resources">Human Resources</option>
                    </select>
                    <select id="attritionFilter" class="form-control" style="width: 180px;" onchange="fetchRecords()">
                        <option value="">All Statuses</option>
                        <option value="Yes">Attrition (Yes)</option>
                        <option value="No">Retained (No)</option>
                    </select>
                </div>
                <div class="table-responsive">
                    <table>
                        <thead>
                            <tr>
                                <th>Age</th>
                                <th>Department</th>
                                <th>Job Level</th>
                                <th>Monthly Income</th>
                                <th>Years</th>
                                <th>Job Sat.</th>
                                <th>Work-Life</th>
                                <th>Overtime</th>
                                <th>Commute</th>
                                <th>Attrition</th>
                            </tr>
                        </thead>
                        <tbody id="tableBody">
                            <!-- Populated dynamically -->
                        </tbody>
                    </table>
                </div>
            </div>
        </div>

        <!-- TAB 4: MODEL PERFORMANCE -->
        <div id="model-info" class="tab-content">
            <div class="chart-grid">
                <div class="card">
                    <div class="card-header">
                        <div class="card-title"><i class="fa-solid fa-sliders"></i> Model Architecture & Retraining</div>
                    </div>
                    <div class="form-group" style="margin-bottom: 1rem;">
                        <label>Select Algorithm</label>
                        <select id="algorithmSelect" class="form-control">
                            <option value="LogisticRegression">Logistic Regression (Balanced)</option>
                            <option value="RandomForest">Random Forest Classifier</option>
                        </select>
                    </div>
                    <button class="btn-submit" onclick="retrainModel()">
                        <i class="fa-solid fa-rotate"></i> Retrain Model Now
                    </button>
                    <div style="margin-top: 1.5rem;">
                        <h4>Evaluation Metrics</h4>
                        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin-top: 0.75rem;">
                            <div style="background: rgba(15,23,42,0.5); padding: 1rem; border-radius: 8px;">
                                <span style="color: var(--text-muted); font-size: 0.8rem;">Accuracy</span>
                                <p id="metric-acc" style="font-size: 1.4rem; font-weight: 700; color: var(--accent-secondary)">--</p>
                            </div>
                            <div style="background: rgba(15,23,42,0.5); padding: 1rem; border-radius: 8px;">
                                <span style="color: var(--text-muted); font-size: 0.8rem;">F1-Score</span>
                                <p id="metric-f1" style="font-size: 1.4rem; font-weight: 700; color: var(--accent-pink)">--</p>
                            </div>
                            <div style="background: rgba(15,23,42,0.5); padding: 1rem; border-radius: 8px;">
                                <span style="color: var(--text-muted); font-size: 0.8rem;">Precision</span>
                                <p id="metric-prec" style="font-size: 1.4rem; font-weight: 700; color: var(--accent-success)">--</p>
                            </div>
                            <div style="background: rgba(15,23,42,0.5); padding: 1rem; border-radius: 8px;">
                                <span style="color: var(--text-muted); font-size: 0.8rem;">Recall</span>
                                <p id="metric-rec" style="font-size: 1.4rem; font-weight: 700; color: var(--accent-warning)">--</p>
                            </div>
                        </div>
                    </div>
                </div>

                <div class="card">
                    <div class="card-header">
                        <div class="card-title"><i class="fa-solid fa-ranking-star"></i> Top Feature Importances</div>
                    </div>
                    <div class="chart-container">
                        <canvas id="importanceChart"></canvas>
                    </div>
                </div>
            </div>
        </div>

    </div>

    <script>
        let charts = {};

        function switchTab(tabId) {
            document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
            document.querySelectorAll('.nav-btn').forEach(el => el.classList.remove('active'));
            
            document.getElementById(tabId).classList.add('active');
            event.currentTarget.classList.add('active');

            if (tabId === 'explorer') fetchRecords();
        }

        async function loadStats() {
            const res = await fetch('/api/stats');
            const data = await res.json();

            document.getElementById('kpi-total').innerText = data.total_employees.toLocaleString();
            document.getElementById('kpi-rate').innerText = data.attrition_rate + '%';
            document.getElementById('kpi-salary').innerText = '$' + data.avg_income.toLocaleString();
            document.getElementById('kpi-satisfaction').innerText = data.avg_satisfaction + ' / 4';

            renderOvertimeChart(data.overtime_stats);
            renderDeptChart(data.dept_stats);
            renderSatChart(data.sat_stats);
            renderLevelChart(data.level_stats);
            updateModelMetrics(data.model_metrics);
        }

        function renderOvertimeChart(stats) {
            const ctx = document.getElementById('overtimeChart').getContext('2d');
            if (charts.overtime) charts.overtime.destroy();
            charts.overtime = new Chart(ctx, {
                type: 'bar',
                data: {
                    labels: stats.map(s => 'Overtime: ' + s.overtime),
                    datasets: [{
                        label: 'Attrition Rate (%)',
                        data: stats.map(s => s.rate),
                        backgroundColor: ['rgba(6, 182, 212, 0.7)', 'rgba(239, 68, 68, 0.7)'],
                        borderRadius: 8
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: { legend: { display: false } },
                    scales: { y: { beginAtZero: true, grid: { color: 'rgba(255,255,255,0.05)' } } }
                }
            });
        }

        function renderDeptChart(stats) {
            const ctx = document.getElementById('deptChart').getContext('2d');
            if (charts.dept) charts.dept.destroy();
            charts.dept = new Chart(ctx, {
                type: 'bar',
                data: {
                    labels: stats.map(s => s.department),
                    datasets: [{
                        label: 'Attrition Rate (%)',
                        data: stats.map(s => s.rate),
                        backgroundColor: 'rgba(99, 102, 241, 0.7)',
                        borderRadius: 8
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: { legend: { display: false } },
                    scales: { y: { beginAtZero: true, grid: { color: 'rgba(255,255,255,0.05)' } } }
                }
            });
        }

        function renderSatChart(stats) {
            const ctx = document.getElementById('satChart').getContext('2d');
            if (charts.sat) charts.sat.destroy();
            charts.sat = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: stats.map(s => 'Score ' + s.satisfaction),
                    datasets: [{
                        label: 'Attrition Rate (%)',
                        data: stats.map(s => s.rate),
                        borderColor: '#ec4899',
                        backgroundColor: 'rgba(236, 72, 153, 0.15)',
                        fill: true,
                        tension: 0.4
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: { y: { beginAtZero: true, grid: { color: 'rgba(255,255,255,0.05)' } } }
                }
            });
        }

        function renderLevelChart(stats) {
            const ctx = document.getElementById('levelChart').getContext('2d');
            if (charts.level) charts.level.destroy();
            charts.level = new Chart(ctx, {
                type: 'doughnut',
                data: {
                    labels: stats.map(s => s.job_level),
                    datasets: [{
                        data: stats.map(s => s.attrition),
                        backgroundColor: ['#6366f1', '#06b6d4', '#10b981']
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: { legend: { position: 'bottom' } }
                }
            });
        }

        function updateModelMetrics(metrics) {
            if (!metrics) return;
            document.getElementById('metric-acc').innerText = metrics.accuracy + '%';
            document.getElementById('metric-f1').innerText = metrics.f1 + '%';
            document.getElementById('metric-prec').innerText = metrics.precision + '%';
            document.getElementById('metric-rec').innerText = metrics.recall + '%';

            renderImportanceChart(metrics.feature_importances);
        }

        function renderImportanceChart(importances) {
            if (!importances) return;
            const ctx = document.getElementById('importanceChart').getContext('2d');
            if (charts.importance) charts.importance.destroy();
            charts.importance = new Chart(ctx, {
                type: 'bar',
                data: {
                    labels: importances.map(i => i.feature),
                    datasets: [{
                        label: 'Impact Magnitude',
                        data: importances.map(i => i.importance),
                        backgroundColor: 'rgba(16, 185, 129, 0.7)',
                        borderRadius: 6
                    }]
                },
                options: {
                    indexAxis: 'y',
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: { legend: { display: false } }
                }
            });
        }

        async function handlePredict(e) {
            e.preventDefault();
            const formData = new FormData(e.target);
            const body = {};
            formData.forEach((val, key) => body[key] = val);

            const res = await fetch('/api/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(body)
            });
            const data = await res.json();

            const box = document.getElementById('resultBox');
            box.style.display = 'block';

            document.getElementById('riskScore').innerText = data.probability + '%';
            
            const badge = document.getElementById('riskBadge');
            badge.innerText = data.risk_level;
            badge.className = 'risk-badge ' + (
                data.badge_color === 'danger' ? 'badge-danger' : 
                data.badge_color === 'warning' ? 'badge-warning' : 'badge-success'
            );

            const meter = document.getElementById('meterFill');
            meter.style.width = data.probability + '%';
            meter.style.background = data.badge_color === 'danger' ? '#ef4444' : (data.badge_color === 'warning' ? '#f59e0b' : '#10b981');

            document.getElementById('keyDrivers').innerHTML = data.key_drivers.map(d => `<li>${d}</li>`).join('');
            document.getElementById('recommendations').innerHTML = data.recommendations.map(r => `<li>${r}</li>`).join('');
        }

        async function fetchRecords() {
            const search = document.getElementById('searchInput').value;
            const dept = document.getElementById('deptFilter').value;
            const attrition = document.getElementById('attritionFilter').value;

            const res = await fetch(`/api/dataset?search=${search}&dept=${dept}&attrition=${attrition}`);
            const data = await res.json();

            const tbody = document.getElementById('tableBody');
            tbody.innerHTML = data.records.map(r => `
                <tr>
                    <td>${r.Age}</td>
                    <td>${r.Department}</td>
                    <td>${r.JobLevel}</td>
                    <td>$${Number(r.MonthlyIncome).toLocaleString()}</td>
                    <td>${r.YearsAtCompany} yrs</td>
                    <td>${r.JobSatisfaction} / 4</td>
                    <td>${r.WorkLifeBalance} / 4</td>
                    <td>${r.OverTime}</td>
                    <td>${r.DistanceFromHomeKm} km</td>
                    <td><span class="pill ${r.Attrition === 'Yes' ? 'pill-yes' : 'pill-no'}">${r.Attrition}</span></td>
                </tr>
            `).join('');
        }

        async function retrainModel() {
            const modelType = document.getElementById('algorithmSelect').value;
            const res = await fetch('/api/retrain', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ model_type: modelType })
            });
            const data = await res.json();
            alert(`Model retrained successfully using ${data.model_type}`);
            updateModelMetrics(data.metrics);
        }

        // Initialize on page load
        window.addEventListener('DOMContentLoaded', () => {
            loadStats();
        });
    </script>
</body>
</html>
"""

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
