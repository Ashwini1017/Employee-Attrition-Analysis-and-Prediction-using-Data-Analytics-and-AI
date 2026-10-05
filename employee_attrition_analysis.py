"""
Employee Attrition Analysis and Prediction using Data Analytics and AI
Student-level project for an IBM SkillsBuild Data Analytics with AI submission.

Run:
    pip install -r requirements.txt
    python employee_attrition_analysis.py

The script:
1. Loads the employee dataset.
2. Cleans missing values and duplicates.
3. Performs basic EDA.
4. Creates charts in results/charts/.
5. Trains a Logistic Regression classifier.
6. Prints evaluation metrics and saves a summary.
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "dataset" / "employee_data.csv"
CHART_DIR = BASE_DIR / "results" / "charts"
CHART_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_PATH)

print("\n--- Original dataset ---")
print(f"Rows: {len(df)}")
print(f"Columns: {len(df.columns)}")
print("\nMissing values before cleaning:")
print(df.isnull().sum())

# Cleaning
df = df.drop_duplicates().copy()
numeric_cols = ["Age", "MonthlyIncome", "YearsAtCompany", "JobSatisfaction",
                "WorkLifeBalance", "DistanceFromHomeKm", "TrainingHours",
                "NumCompaniesWorked"]
for col in numeric_cols:
    df[col] = df[col].fillna(df[col].median())

df["Department"] = df["Department"].fillna(df["Department"].mode()[0])
df["JobLevel"] = df["JobLevel"].fillna(df["JobLevel"].mode()[0])
df["OverTime"] = df["OverTime"].fillna(df["OverTime"].mode()[0])
df["Attrition"] = df["Attrition"].fillna(df["Attrition"].mode()[0])

# -------------------------
# EDA charts
# -------------------------
attrition_counts = df["Attrition"].value_counts()
plt.figure(figsize=(7, 5))
attrition_counts.plot(kind="bar")
plt.title("Employee Attrition Count")
plt.xlabel("Attrition")
plt.ylabel("Number of Employees")
plt.tight_layout()
plt.savefig(CHART_DIR / "attrition_count.png", dpi=150)
plt.close()

overtime_rate = pd.crosstab(df["OverTime"], df["Attrition"], normalize="index") * 100
overtime_rate.plot(kind="bar", figsize=(8, 5))
plt.title("Attrition by Overtime")
plt.xlabel("Overtime")
plt.ylabel("Percentage")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(CHART_DIR / "attrition_by_overtime.png", dpi=150)
plt.close()

dept_rate = pd.crosstab(df["Department"], df["Attrition"], normalize="index") * 100
dept_rate.plot(kind="bar", figsize=(9, 5))
plt.title("Attrition by Department")
plt.xlabel("Department")
plt.ylabel("Percentage")
plt.xticks(rotation=20, ha="right")
plt.tight_layout()
plt.savefig(CHART_DIR / "attrition_by_department.png", dpi=150)
plt.close()

plt.figure(figsize=(8, 5))
df.boxplot(column="MonthlyIncome", by="Attrition")
plt.suptitle("")
plt.title("Monthly Income by Attrition")
plt.xlabel("Attrition")
plt.ylabel("Monthly Income")
plt.tight_layout()
plt.savefig(CHART_DIR / "income_by_attrition.png", dpi=150)
plt.close()

# -------------------------
# Machine learning
# -------------------------
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

model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced"))
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

model.fit(X_train, y_train)
predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("\n--- Model Evaluation ---")
print(f"Test accuracy: {accuracy:.2%}")
print("\nClassification report:")
print(classification_report(y_test, predictions, target_names=["No Attrition", "Attrition"], zero_division=0))

cm = confusion_matrix(y_test, predictions)
plt.figure(figsize=(6, 5))
plt.imshow(cm)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.xticks([0, 1], ["No", "Yes"])
plt.yticks([0, 1], ["No", "Yes"])
for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i, j], ha="center", va="center")
plt.tight_layout()
plt.savefig(CHART_DIR / "confusion_matrix.png", dpi=150)
plt.close()

# Save summary
summary = f"""Employee Attrition Analysis - Run Summary

Rows after cleaning: {len(df)}
Columns: {len(df.columns)}
Attrition rate: {(df["Attrition"].eq("Yes").mean()):.2%}
Test accuracy: {accuracy:.2%}

Main observations:
- Overtime is associated with a higher attrition rate in this dataset.
- Job satisfaction and work-life balance are useful indicators to monitor.
- Employee income, tenure and distance from home can also help explain differences in attrition.
- The Logistic Regression model provides a simple baseline for predicting attrition.

Note: This is a synthetic educational dataset, so the findings should not be treated as real company statistics.
"""
(BASE_DIR / "results" / "analysis_summary.txt").write_text(summary, encoding="utf-8")
print("\nAnalysis completed. Charts and summary are saved in the results folder.")
