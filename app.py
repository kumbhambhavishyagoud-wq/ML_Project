import os
import io
import re
import json
import uuid
import glob
import logging
from datetime import datetime, timedelta
from types import SimpleNamespace
import joblib
import pandas as pd
import ollama

from google import genai
from google.genai import types

from dotenv import load_dotenv
from PIL import Image

from train_model import get_best_model, get_selected_model, train_models

from flask import (
    Flask,
    render_template,
    request,
    jsonify,
    redirect,
    url_for,
    flash,
    send_file
)

from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy import or_

from flask_bcrypt import Bcrypt

from flask_login import (
    LoginManager,
    UserMixin,
    login_user,
    logout_user,
    login_required,
    current_user
)

from werkzeug.utils import secure_filename


# =========================================================
# CONFIGURATION
# =========================================================

load_dotenv()

app = Flask(__name__)

app.config["SECRET_KEY"] = os.getenv(
    "SECRET_KEY",
    "change-this-secret-key-in-your-env-file"
)

app.config["SQLALCHEMY_DATABASE_URI"] = os.getenv("MYSQL_URL")
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

app.config["REMEMBER_COOKIE_DURATION"] = timedelta(days=30)
app.config["REMEMBER_COOKIE_HTTPONLY"] = True
app.config["REMEMBER_COOKIE_SAMESITE"] = "Lax"

app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"

app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

logging.basicConfig(level=logging.INFO)

bcrypt = Bcrypt(app)


class Base(DeclarativeBase):
    pass


db = SQLAlchemy(model_class=Base)
db.init_app(app)

custom_model = {}
custom_report = None
current_csv_path=None

# =========================================================
# LOGIN MANAGER
# =========================================================

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# =========================================================
# UPLOAD FOLDERS CONFIGURATION
# =========================================================

CSV_UPLOAD_FOLDER = os.path.join(app.root_path, "upload_csv")
IMAGE_UPLOAD_FOLDER = os.path.join(app.root_path, "uploads")

os.makedirs(CSV_UPLOAD_FOLDER, exist_ok=True)
os.makedirs(IMAGE_UPLOAD_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = IMAGE_UPLOAD_FOLDER


# =========================================================
# DATABASE MODELS
# =========================================================

class User(UserMixin, db.Model):
    __tablename__ = "users"

    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(255), nullable=False)

    def set_password(self, password):
        self.password = bcrypt.generate_password_hash(password).decode("utf-8")

    def check_password(self, password):
        return bcrypt.check_password_hash(self.password, password)


class PredictionHistory(db.Model):
    __tablename__ = "prediction_history"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(
        db.Integer,
        db.ForeignKey("users.id"),
        nullable=False,
        index=True
    )
    created_at = db.Column(
        db.DateTime,
        default=datetime.utcnow,
        nullable=False
    )
    prediction = db.Column(db.String(150), nullable=False)
    disease_probability = db.Column(db.Float, nullable=False)
    healthy_probability = db.Column(db.Float, nullable=False)
    input_data = db.Column(db.Text, nullable=False)
    ai_analysis = db.Column(db.Text, nullable=True)
    health_tips = db.Column(db.Text, nullable=True)

    user = db.relationship(
        "User",
        backref=db.backref(
            "prediction_history",
            lazy=True,
            cascade="all, delete-orphan"
        )
    )


@login_manager.user_loader
def load_user(user_id):
    try:
        return db.session.get(User, int(user_id))
    except (ValueError, TypeError):
        return None


# =========================================================
# DATABASE INITIALIZATION
# =========================================================

with app.app_context():
    db.create_all()
    app.logger.info("Database tables initialized.")


ALLOWED_IMAGE_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}


def allowed_image(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_IMAGE_EXTENSIONS
    )


# =========================================================
# MACHINE LEARNING MODEL
# =========================================================

FEATURE_NAMES = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal"
]

NUMERICAL_COLS = [
    "age",
    "trestbps",
    "chol",
    "thalach",
    "oldpeak"
]

CATEGORICAL_COLS = [
    "sex",
    "cp",
    "fbs",
    "restecg",
    "exang",
    "slope",
    "ca",
    "thal"
]

MODEL_PATH = os.path.join(app.root_path, "best_heart_model.pkl")

try:
    if not os.path.exists(MODEL_PATH):
        app.logger.info("Training initial heart disease models...")
        report, trained_models = train_models("heart.csv")
        best_pipeline = get_best_model(trained_models, report)
        joblib.dump(best_pipeline, MODEL_PATH)

    model = joblib.load(MODEL_PATH)

    if not hasattr(model, "predict_proba"):
        raise RuntimeError("The loaded model does not support predict_proba().")

    model_classes = {str(value) for value in model.classes_}

    if not {"0", "1"}.issubset(model_classes):
        raise RuntimeError("The model must contain classes 0 and 1.")

except Exception as e:
    app.logger.exception("Could not load ML model: %s", e)
    model = None


# =========================================================
# GENERAL HELPERS
# =========================================================

def clean_text(value):
    if value is None:
        return ""
    return str(value).strip()


MODEL_DISPLAY_NAMES = {
    "LogisticRegression": "Logistic Regression",
    "SVC": "Support Vector Machine",
    "RandomForestClassifier": "Random Forest",
    "KNeighborsClassifier": "K-Nearest Neighbours",
    "DecisionTreeClassifier": "Decision Tree",
    "GradientBoostingClassifier": "Gradient Boosting",
}


def get_estimator_name(ml_model):
    if ml_model is None:
        return "None"
    if hasattr(ml_model, "steps") and ml_model.steps:
        return type(ml_model.steps[-1][1]).__name__
    return type(ml_model).__name__


@app.context_processor
def inject_active_model():
    raw_name = get_estimator_name(model)
    friendly_name = MODEL_DISPLAY_NAMES.get(raw_name, raw_name)
    return {"active_model_name": friendly_name}


def json_safe_values(values):
    cleaned = {}
    for key, value in values.items():
        if value is None:
            cleaned[key] = None
        elif pd.isna(value):
            cleaned[key] = None
        elif hasattr(value, "item"):
            cleaned[key] = value.item()
        else:
            cleaned[key] = value
    return cleaned


def get_model_probability_map(probabilities):
    if model is None:
        raise RuntimeError("ML model is not loaded.")

    classes = list(model.classes_)
    probability_map = {}

    for index, class_label in enumerate(classes):
        probability_map[str(class_label)] = float(probabilities[index]) * 100

    if "0" not in probability_map or "1" not in probability_map:
        raise ValueError("Expected model classes 0 and 1 were not found.")

    disease_probability = probability_map["0"]
    healthy_probability = probability_map["1"]

    return disease_probability, healthy_probability


def get_prediction_label(disease_probability):
    if disease_probability > 60:
        return "High Model-Estimated Risk"
    elif disease_probability >= 40:
        return "Moderate Model-Estimated Risk"
    return "Low Model-Estimated Risk"


# =========================================================
# OLLAMA RESPONSE PARSING
# =========================================================

def extract_json_object(response_text):
    if not response_text:
        raise ValueError("Ollama returned an empty response.")

    text = response_text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)

    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1 or end <= start:
        raise ValueError("No valid JSON object found in Ollama response.")

    parsed = json.loads(text[start:end + 1])

    if not isinstance(parsed, dict):
        raise ValueError("Ollama response is not a JSON object.")

    return parsed


# =========================================================
# DEFAULT AI RESPONSE
# =========================================================

def get_default_ai_assessment():
    return {
        "analysis": (
            "Automated AI analysis is currently unavailable. "
            "The machine-learning prediction was processed separately. "
            "Please consult a qualified healthcare professional "
            "for an appropriate medical evaluation."
        ),
        "health_tips": [
            "Follow a balanced diet with a variety of foods.",
            "Choose physical activities appropriate for your health and circumstances.",
            "Maintain a consistent sleep schedule.",
            "Attend appropriate routine health check-ups."
        ],
        "ai_status": "unavailable"
    }


# =========================================================
# AI CLINICAL ASSESSMENT
# =========================================================

def generate_ai_assessment(
    values,
    prediction_text,
    disease_probability,
    healthy_probability
):
    system_instruction = """
You are a supportive clinical communication assistant.
Analyze the supplied heart-health model inputs and screening output.
"""

    user_prompt = f"""
PATIENT INPUT PARAMETERS:

{json.dumps(values, indent=2)}

MODEL CLASSIFICATION:

{prediction_text}

DISEASE MODEL PROBABILITY:

{disease_probability:.2f}%

HEALTHY MODEL PROBABILITY:

{healthy_probability:.2f}%

Write the assessment and recommendations in the required JSON structure following these rules:
1. Never diagnose or claim the patient definitely has or does not have heart disease.
2. Do not invent symptoms, medical history, test results, or missing patient information.
3. Provide 3 to 5 practical, engaging, and varied general wellness suggestions using friendly emojis.
4. Keep tips distinct and fresh every time.
5. Do not prescribe medicines, supplements, or specific medical treatments.
6. Use simple, warm, professional, and encouraging language.
7. Keep the complete response concise and under 200 words.
8. Most importantly, keep the language as simple as possible.
"""

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.2,
                response_mime_type="application/json",
                response_schema={
                    "type": "OBJECT",
                    "properties": {
                        "analysis": {
                            "type": "STRING",
                            "description": "A concise, clear, and encouraging clinical-style summary."
                        },
                        "health_tips": {
                            "type": "ARRAY",
                            "items": {"type": "STRING"},
                            "description": "3 to 5 practical recommendations with emojis."
                        }
                    },
                    "required": ["analysis", "health_tips"]
                }
            )
        )

        raw_response = response.text.strip()
        ai_result = json.loads(raw_response)

        analysis = ai_result.get("analysis")
        if not isinstance(analysis, str) or not analysis.strip():
            analysis = get_default_ai_assessment()["analysis"]

        tips = ai_result.get("health_tips", [])
        if isinstance(tips, str):
            tips = [tips]
        if not isinstance(tips, list):
            tips = []

        tips = [
            str(tip).strip()
            for tip in tips
            if isinstance(tip, (str, int, float)) and str(tip).strip()
        ]

        if len(tips) < 3:
            tips = get_default_ai_assessment()["health_tips"]

        tips = tips[:5]

        return {
            "analysis": analysis.strip(),
            "health_tips": tips,
            "ai_status": "success"
        }

    except Exception as e:
        app.logger.exception("Gemini AI assessment failed: %s", e)
        return get_default_ai_assessment()


# =========================================================
# SAVE PREDICTION HISTORY
# =========================================================

def save_prediction_history(result):
    if not current_user.is_authenticated:
        return None

    try:
        record = PredictionHistory(
            user_id=current_user.id,
            prediction=result["prediction"],
            disease_probability=float(result["disease_probability"]),
            healthy_probability=float(result["healthy_probability"]),
            input_data=json.dumps(result["input_data"]),
            ai_analysis=result["ai_analysis"],
            health_tips=json.dumps(result["health_tips"])
        )

        db.session.add(record)
        db.session.commit()
        return record.id

    except Exception as e:
        db.session.rollback()
        app.logger.exception("Prediction history save failed: %s", e)
        return None


# =========================================================
# COMMON PREDICTION FUNCTION
# =========================================================

def run_prediction(values, save_history=True):
    if model is None:
        raise RuntimeError("ML model is not loaded on the server.")

    missing_features = [
        feature
        for feature in FEATURE_NAMES
        if feature not in values or values[feature] is None
    ]

    if missing_features:
        raise ValueError("Missing model features: " + ", ".join(missing_features))

    ordered_values = {feature: values[feature] for feature in FEATURE_NAMES}

    input_df = pd.DataFrame([ordered_values], columns=FEATURE_NAMES)

    for col in NUMERICAL_COLS:
        input_df[col] = pd.to_numeric(input_df[col], errors="raise")

    for col in CATEGORICAL_COLS:
        input_df[col] = input_df[col].astype(str)

    predicted_class = model.predict(input_df)[0]
    probabilities = model.predict_proba(input_df)[0]

    disease_probability, healthy_probability = get_model_probability_map(probabilities)
    prediction_text = get_prediction_label(disease_probability)
    clean_values = json_safe_values(ordered_values)

    ai_result = generate_ai_assessment(
        clean_values,
        prediction_text,
        disease_probability,
        healthy_probability
    )

    result = {
        "prediction": prediction_text,
        "predicted_class": str(predicted_class),
        "disease_probability": round(disease_probability, 2),
        "healthy_probability": round(healthy_probability, 2),
        "input_data": clean_values,
        "extracted_values": clean_values,
        "patient_name": current_user.username if current_user.is_authenticated else "Guest",
        "ai_analysis": ai_result["analysis"],
        "ai_message": ai_result["analysis"],
        "health_tips": ai_result["health_tips"],
        "ai_status": ai_result["ai_status"]
    }

    if save_history:
        history_id = save_prediction_history(result)
        if history_id is not None:
            result["history_id"] = history_id

    return result


# =========================================================
# ROUTES
# =========================================================

@app.route("/")
def home():
    return render_template("home.html")


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/detect", methods=["GET", "POST"])
def detect():
    if request.method == "GET":
        return render_template("detect.html", form_data={})

    if model is None:
        return render_template(
            "detect.html",
            form_data=request.form,
            result={"error": "ML model is not loaded."}
        )

    try:
        values = {}
        for feature in FEATURE_NAMES:
            raw_value = request.form.get(feature, "").strip()
            if raw_value == "":
                raise ValueError(f"Please provide a value for {feature}.")

            if feature in NUMERICAL_COLS:
                values[feature] = float(raw_value)
            else:
                values[feature] = str(raw_value)

        result = run_prediction(values, save_history=True)

        return render_template(
            "detect.html",
            result=result,
            form_data=values
        )

    except Exception as e:
        db.session.rollback()
        app.logger.exception("Manual prediction failed: %s", e)

        return render_template(
            "detect.html",
            form_data=request.form,
            result={"error": str(e)}
        )


@app.route("/detect_file", methods=["POST"])
@login_required
def detect_file():
    if "report_image" not in request.files:
        return render_template("detect.html", error=True, message="No report image was uploaded.", data={})

    file = request.files["report_image"]

    if not file or not file.filename:
        return render_template("detect.html", error=True, message="Please select an image.", data={})

    file_path = None

    try:
        if model is None:
            raise RuntimeError("ML model is not loaded.")

        original_filename = secure_filename(file.filename)

        if not allowed_image(original_filename):
            raise ValueError("Please upload a PNG, JPG, JPEG or WEBP image.")

        image_bytes = file.read()
        if not image_bytes:
            raise ValueError("The uploaded image is empty.")

        verify_image = Image.open(io.BytesIO(image_bytes))
        verify_image.verify()

        image = Image.open(io.BytesIO(image_bytes))
        image = image.convert("RGB")
        image.thumbnail((1200, 1200))

        filename = str(uuid.uuid4()) + ".jpg"
        file_path = os.path.join(IMAGE_UPLOAD_FOLDER, filename)
        image.save(file_path, "JPEG", quality=90)

        extraction_instruction = f"""
You are a precise clinical data extraction assistant. Extract the exact 13 features required for the heart disease prediction model:
{FEATURE_NAMES}
Return valid JSON only. Output null for any missing or illegible field.
"""
        response = ollama.chat(
            model="gemma4",
            messages=[
                {"role": "system", "content": extraction_instruction},
                {
                    "role": "user",
                    "content": "Extract the 13 required parameters from this uploaded report.",
                    "images": [file_path]
                }
            ],
            format="json",
            options={"temperature": 0}
        )

        extracted = extract_json_object(response["message"]["content"])
        if not isinstance(extracted, dict):
            extracted = {}

        values = {}
        for feature in FEATURE_NAMES:
            val = extracted.get(feature)
            if val is not None and str(val).strip() != "" and str(val).lower() != "none":
                values[feature] = val
            else:
                values[feature] = None

        missing_fields = [feature for feature, value in values.items() if value is None]

        if missing_fields:
            return render_template(
                "detect.html",
                error=True,
                warning=True,
                message="Some values could not be extracted: " + ", ".join(missing_fields) + ". Please complete manually.",
                data=values,
                missing_fields=missing_fields
            )

        converted_values = {}
        for feature in FEATURE_NAMES:
            if feature in NUMERICAL_COLS:
                converted_values[feature] = float(values[feature])
            else:
                converted_values[feature] = str(values[feature])

        result = run_prediction(converted_values, save_history=True)
        result["source"] = "uploaded_report"

        return render_template("detect.html", result=result, data=converted_values)

    except Exception as e:
        db.session.rollback()
        app.logger.exception("Image extraction or prediction failed: %s", e)
        return render_template("detect.html", error=True, message="Could not process the uploaded report. " + str(e), data={})

    finally:
        if file_path and os.path.exists(file_path):
            try:
                os.remove(file_path)
            except OSError:
                pass




# Add this alongside your other global variables
  # Or set to your default dataset path, e.g., "data/heart.csv"

@app.route("/train", methods=["GET", "POST"])
@login_required
def train():
    global custom_model, custom_report, model, current_csv_path
    if request.method == "POST":
        if "train_data" not in request.files:
            return render_template("train.html", error=True, message="No file selected.")

        file = request.files["train_data"]

        if not file.filename or not file.filename.lower().endswith(".csv"):
            return render_template("train.html", error=True, message="Only CSV files are accepted.")

        filename = secure_filename(file.filename)
        file_path = os.path.join(CSV_UPLOAD_FOLDER, filename)
        file.save(file_path)

        try:
            report, trained_models = train_models(file_path)
            custom_model = trained_models
            custom_report = report
            current_csv_path = file_path  # Save the active CSV path

            return render_template(
                "train.html",
                success=True,
                report=report,
                selected_model=get_estimator_name(model),
                csv_filename=filename
            )
        except Exception as e:
            app.logger.exception("Custom model training failed: %s", e)
            return render_template("train.html", error=True, message=str(e))

    if request.method == 'GET':
        if not custom_model and custom_report is None:
            return render_template("train.html")
        else:
            # Extract the filename so it persists if the user refreshes or navigates back
            filename = os.path.basename(current_csv_path) if current_csv_path else None
            return render_template(
                "train.html",
                success=True,
                report=custom_report,
                selected_model=get_estimator_name(model),
                csv_filename=filename
            )

@app.route("/download_dataset", methods=["GET"])
@login_required
def download_dataset():
    if not current_csv_path or not os.path.exists(current_csv_path):
        return render_template("train.html", error=True, message="No dataset available to download.")

    return send_file(
        current_csv_path,
        as_attachment=True,
        download_name=os.path.basename(current_csv_path),
        mimetype="text/csv"
    )

@app.route('/default', methods=['POST'])
@app.route('/new', methods=["POST"])
@login_required
def default():
    global model
    global custom_model, custom_report,current_csv_path
    if request.method == 'POST':
        try:
            report, trained_models = train_models("heart.csv")
            best_pipeline = get_best_model(trained_models, report)
            joblib.dump(best_pipeline, MODEL_PATH)
            model = joblib.load(MODEL_PATH)
            for old_file in glob.glob(os.path.join(CSV_UPLOAD_FOLDER, '*.csv')):
                        try:
                            os.remove(old_file)
                        except OSError:
                            pass
            custom_report = None
            custom_model = {}
            current_csv_path = None
            if request.path == '/new':
                return render_template(url_for('train'))
            elif request.path == '/default':
                return render_template('detect.html')

        except Exception as e:
            return render_template("train.html", error=True, message=str(e))







@app.route("/custom", methods=["POST", "GET"])
@login_required
def custom():
    global model
    if request.method == "POST":
        model_name = request.form.get("model_name")

        if not custom_model:
            return render_template("train.html", error=True, message="No custom models found. Please train first.")

        selected = get_selected_model(custom_model, model_name)
        if selected is None:
            return render_template("train.html", error=True, message=f"Model '{model_name}' could not be found.")

        selected_classes = {str(v) for v in getattr(selected, "classes_", [])}
        if not hasattr(selected, "predict_proba") or not {"0", "1"}.issubset(selected_classes):
            return render_template("train.html", error=True, message="Selected model must support probabilities and binary classes 0 and 1.")

        model = selected
        joblib.dump(model, MODEL_PATH)

        return render_template(
            "detect.html",
            form_data={},
            message=f"Active model switched to: {get_estimator_name(model)}"
        )

    return render_template("train.html")


@app.route("/hospital", methods=["GET", "POST"])
@login_required
def hospital():
    if request.method == "GET":
        return render_template("hospital.html")

    if model is None:
        return jsonify({"error": "ML model is not loaded."}), 500

    if "csv_file" not in request.files:
        return jsonify({"error": "No CSV file was uploaded."}), 400

    file = request.files["csv_file"]
    if not file.filename or not file.filename.lower().endswith(".csv"):
        return jsonify({"error": "Only CSV files are supported."}), 400

    try:
        df = pd.read_csv(io.BytesIO(file.read()))

        # Check for target or existing prediction columns (case-insensitive)
        forbidden_targets = {"target", "target_prediction", "num", "output", "label", "class", "diagnosis"}
        found_targets = [col for col in df.columns if col.strip().lower() in forbidden_targets]

        if found_targets:
            return jsonify({
                "error": f"Invalid dataset: Target/prediction column(s) detected ({', '.join(found_targets)}). This appears to be an already labeled or predicted dataset."
            }), 400

        missing_columns = [f for f in FEATURE_NAMES if f not in df.columns]
        if missing_columns:
            return jsonify({"error": "Missing required columns: " + ", ".join(missing_columns)}), 400

        if df.empty:
            return jsonify({"error": "The uploaded CSV contains no records."}), 400

        model_df = df[FEATURE_NAMES].copy()
        for col in NUMERICAL_COLS:
            model_df[col] = pd.to_numeric(model_df[col], errors="coerce")

        if model_df.isnull().any().any():
            return jsonify({"error": "CSV contains missing or invalid values."}), 400

        for col in CATEGORICAL_COLS:
            model_df[col] = model_df[col].astype(str)

        predictions = model.predict(model_df)
        probabilities = model.predict_proba(model_df)

        classes = list(model.classes_)
        class_indices = {str(label): index for index, label in enumerate(classes)}

        disease_index = class_indices["0"]
        healthy_index = class_indices["1"]

        df["Target_Prediction"] = ["positive" if str(p) == "0" else "negative" for p in predictions]
        df["Probability_Diseased"] = [f"{float(row[disease_index]) * 100:.2f}%" for row in probabilities]
        df["Probability_Healthy"] = [f"{float(row[healthy_index]) * 100:.2f}%" for row in probabilities]

        return jsonify({
            "status": "success",
            "healthy_count": sum(1 for p in predictions if str(p) == "1"),
            "diseased_count": sum(1 for p in predictions if str(p) == "0"),
            "csv_data": df.to_csv(index=False)
        }), 200

    except Exception as e:
        app.logger.exception("Hospital CSV prediction failed: %s", e)
        return jsonify({"error": str(e)}), 500

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email", "").strip()
        password = request.form.get("password", "")

        if not email or not password:
            return render_template("login.html", error=True, message="Please enter your email and password.")

        user = db.session.execute(db.select(User).filter_by(email=email)).scalar_one_or_none()

        if not user or not user.check_password(password):
            return render_template("login.html", error=True, message="Invalid email or password.")

        login_user(user, remember=True)
        next_page = request.args.get("next")
        if next_page and next_page.startswith("/") and not next_page.startswith("//"):
            return redirect(next_page)

        return redirect(url_for("detect"))

    return render_template("login.html", error=False)


@app.route("/signin", methods=["GET", "POST"])
def signin():
    if request.method == "POST":
        username = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip()
        password = request.form.get("password", "")

        if not username or not email or not password:
            return render_template("signin.html", error=True, message="Please complete all fields.")

        existing_user = db.session.execute(
            db.select(User).where(or_(User.username == username, User.email == email))
        ).scalar_one_or_none()

        if existing_user:
            return render_template("signin.html", error=True, message="Username or email already registered.")

        new_user = User(username=username, email=email)
        new_user.set_password(password)

        try:
            db.session.add(new_user)
            db.session.commit()
            login_user(new_user, remember=True)
            return redirect(url_for("detect"))
        except Exception as e:
            db.session.rollback()
            app.logger.exception("Signup failed: %s", e)
            return render_template("signin.html", error=True, message="Registration failed. Please try again.")

    return render_template("signin.html", error=False)


@app.route("/logout", methods=["GET", "POST"])
@login_required
def logout():
    logout_user()
    return redirect(url_for("home"))


def parse_history_value(value, default):
    if value is None or not isinstance(value, str):
        return value or default
    value = value.strip()
    if not value:
        return default
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError, ValueError):
        return value


def normalize_history_tips(value):
    parsed = parse_history_value(value, [])
    if isinstance(parsed, list):
        return [str(item).strip() for item in parsed if item is not None and str(item).strip()]
    if isinstance(parsed, str):
        return [line.strip(" -•\t") for line in parsed.splitlines() if line.strip(" -•\t")] or [parsed]
    return [str(parsed)]


@app.route("/history")
@login_required
def history():
    records = PredictionHistory.query.filter_by(user_id=current_user.id).order_by(PredictionHistory.created_at.desc()).all()
    history_records = []
    selected_record = None
    history_id = request.args.get("history_id", type=int)

    for record in records:
        parsed_inputs = parse_history_value(record.input_data, {})
        if not isinstance(parsed_inputs, dict):
            parsed_inputs = {}

        history_item = SimpleNamespace(
            id=record.id,
            user_id=record.user_id,
            created_at=record.created_at,
            prediction=record.prediction,
            disease_probability=record.disease_probability,
            healthy_probability=record.healthy_probability,
            input_data=parsed_inputs,
            ai_analysis=record.ai_analysis,
            health_tips=normalize_history_tips(record.health_tips)
        )

        history_records.append(history_item)
        if history_id is not None and record.id == history_id:
            selected_record = history_item

    if history_id is not None and selected_record is None:
        flash("Report not found.", "warning")
        return redirect(url_for("history"))

    return render_template("history.html", history_records=history_records, selected_record=selected_record)


@app.errorhandler(413)
def file_too_large(error):
    return jsonify({"error": "Uploaded file is too large. Maximum allowed size is 10 MB."}), 413


if __name__ == "__main__":
    app.run(debug=True)