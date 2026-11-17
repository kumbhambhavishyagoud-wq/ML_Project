import time
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# All 13 standard input parameters for the UCI Cleveland Heart Disease dataset
INPUT_COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", 
    "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"
]

# Separate features into numeric and categorical for proper preprocessing
NUMERIC_COLUMNS = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAL_COLUMNS = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]

# Display fields matching all 13 features for form rendering
DISPLAY_FIELDS = [
    {"name": "age", "label": "Age (years)", "kind": "number", "placeholder": "e.g. 52", "min": "1", "max": "120"},
    {"name": "sex", "label": "Sex", "kind": "select", "options": [("1", "Male"), ("0", "Female")]},
    {"name": "cp", "label": "Chest Pain Type", "kind": "select", "options": [("0", "Typical Angina"), ("1", "Atypical Angina"), ("2", "Non-anginal Pain"), ("3", "Asymptomatic")]},
    {"name": "trestbps", "label": "Resting BP (mm Hg)", "kind": "number", "placeholder": "e.g. 120", "min": "70", "max": "250"},
    {"name": "chol", "label": "Serum Cholesterol (mg/dl)", "kind": "number", "placeholder": "e.g. 220", "min": "80", "max": "700"},
    {"name": "fbs", "label": "Fasting Blood Sugar", "kind": "select", "options": [("0", "Fasting Blood Sugar ≤ 120 mg/dl"), ("1", "Fasting Blood Sugar > 120 mg/dl")]},
    {"name": "restecg", "label": "Resting Electrocardiographic Result", "kind": "select", "options": [("0", "Normal"), ("1", "ST-T wave abnormality"), ("2", "Left ventricular hypertrophy")]},
    {"name": "thalach", "label": "Maximum Heart Rate Achieved", "kind": "number", "placeholder": "e.g. 150", "min": "50", "max": "250"},
    {"name": "exang", "label": "Exercise Induced Angina", "kind": "select", "options": [("0", "No"), ("1", "Yes")]},
    {"name": "oldpeak", "label": "ST Depression (Oldpeak)", "kind": "number", "placeholder": "e.g. 1.0", "min": "0.0", "max": "10.0"},
    {"name": "slope", "label": "Slope of Peak Exercise ST Segment", "kind": "select", "options": [("0", "Upsloping"), ("1", "Flat"), ("2", "Downsloping")]},
    {"name": "ca", "label": "Number of Major Vessels (0-3)", "kind": "select", "options": [("0", "0"), ("1", "1"), ("2", "2"), ("3", "3")]},
    {"name": "thal", "label": "Thalassemia", "kind": "select", "options": [("1", "Normal"), ("2", "Fixed Defect"), ("3", "Reversible Defect")]}
]

TARGETS = ["target", "num", "output", "condition", "heart_disease"]

def normalise_columns(df):
    df = df.copy()
    df.columns = [str(c).strip().lower().replace(" ", "_").replace("-", "_") for c in df.columns]
    aliases = {
        "resting_blood_pressure": "trestbps", "serum_cholestoral": "chol",
        "serum_cholesterol": "chol", "fasting_blood_sugar": "fbs",
        "resting_electrocardiographic_results": "restecg",
        "maximum_heart_rate_achieved": "thalach", "exercise_induced_angina": "exang",
        "chest_pain_type": "cp", "heart_disease": "target"
    }
    df = df.rename(columns=aliases)
    return df

def clean_dataset(source):
    df = normalise_columns(source)
    target = next((x for x in TARGETS if x in df.columns), None)
    if target is None: 
        raise ValueError("Target column is missing. It must be named target, num, output, condition or heart_disease.")
    
    missing = [c for c in INPUT_COLUMNS if c not in df.columns]
    if missing: 
        raise ValueError("Required columns missing: " + ", ".join(missing))
    
    df = df[INPUT_COLUMNS + [target]].copy()
    for c in df.columns: 
        df[c] = pd.to_numeric(df[c], errors="coerce")
        
    df = df.dropna(subset=[target])
    df[target] = (df[target] > 0).astype(int)
    df = df.dropna(subset=INPUT_COLUMNS, how="all")
    
    if len(df) < 30: 
        raise ValueError("At least 30 usable patient records are required.")
    if df[target].nunique() != 2: 
        raise ValueError("Target must contain both classes: 0 (no disease) and positive values (disease).")
    return df, target

def train_models(csv_path):
    source = pd.read_csv(csv_path)
    df, _ = clean_dataset(source)
    X = df[INPUT_COLUMNS]
    y = df.iloc[:, -1]
    
    if y.value_counts().min() < 2: 
        raise ValueError("Need at least two examples of each target class.")
        
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # --- Preprocessing Pipeline with ColumnTransformer ---
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_COLUMNS),
            ("cat", categorical_transformer, CATEGORICAL_COLUMNS)
        ]
    )
    
    models = {
      "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced"),
      "Logistic Regression": LogisticRegression(max_iter=1500),
      "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
      "Support Vector Machine": SVC(probability=True,random_state=42),  # Removed probability=True here
      "K-Nearest Neighbours": KNeighborsClassifier(n_neighbors=5)
    }
    
    results = []
    trained_models = {}
    
    for name, clf in models.items():
        # Combine preprocessor and the classifier into a single robust pipeline
        pipe = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", clf)
        ])
        
        start = time.perf_counter()
        pipe.fit(Xtr, ytr)
        elapsed = time.perf_counter() - start
        
        yp = pipe.predict(Xte)
        score = accuracy_score(yte, yp)
        
        results.append({
            "algorithm": name,
            "accuracy": round(score * 100, 2),
            "precision": round(precision_score(yte, yp, zero_division=0) * 100, 2),
            "recall": round(recall_score(yte, yp, zero_division=0) * 100, 2),
            "f1": round(f1_score(yte, yp, zero_division=0) * 100, 2),
            "training_seconds": round(elapsed, 4)
        })
        
        trained_models[name] = pipe

    sorted_results = sorted(results, key=lambda r: r["accuracy"], reverse=True)
    best_algorithm = sorted_results[0]["algorithm"]

    report = {
        "records": len(df),
        "input_features": INPUT_COLUMNS,
        "target": "binary disease outcome",
        "test_percent": 20,
        "selected_model": best_algorithm,
        "selected_model_accuracy": sorted_results[0]["accuracy"],
        "results": sorted_results
    }
    
    return report, trained_models

def get_best_model(trained_models, report):
    """Returns the pipeline of the top-performing model."""
    best_name = report["selected_model"]
    return trained_models.get(best_name)

def get_selected_model(trained_models, model_name):
    """Returns the specific model pipeline requested by name."""
    if model_name not in trained_models:
        available = list(trained_models.keys())
        raise ValueError(f"Model '{model_name}' not found. Choose from: {available}")
    return trained_models[model_name]

# 1. Train all models and get the performance report and dictionary of trained pipelines
csv_file_path = "heart.csv"


# After calling train_models and getting your best modheart_disease_datael pipeline:
report, trained_models = train_models(csv_file_path)
best_pipeline = get_best_model(trained_models, report)

# Save the best model pipeline to a pkl file
model_filename = "best_heart_model.pkl"
joblib.dump(best_pipeline, model_filename)
print(f"Best model ({report['selected_model']}) saved successfully to {model_filename}")

# --- How to load and use it later in production/app ---
# loaded_model = joblib.load("best_heart_model.pkl")
# prediction = loaded_model.predict(new_patient_df)