"""
Flask Backend - Salary Prediction API
POST /predict  →  JSON { predicted_salary, formatted_salary, input_features }
Stores every prediction in MySQL (falls back to SQLite if MySQL unavailable)
"""

import os, json, logging
from datetime import datetime

from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd

from flask import Flask
import mysql.connector

app = Flask(__name__)

conn = mysql.connector.connect(
    host="localhost",
    user="root",
    password="",
    database="salary_db"
)

cursor = conn.cursor()

# ── logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

app = Flask(__name__)

# ── Load model & metadata ─────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "best_model.pkl")
META_PATH  = os.path.join(BASE_DIR, "model", "metadata.json")

model    = joblib.load(MODEL_PATH)
with open(META_PATH) as f:
    metadata = json.load(f)

NUMERICAL_COLS   = metadata["numerical_cols"]
CATEGORICAL_COLS = metadata["categorical_cols"]
ALL_COLS         = CATEGORICAL_COLS + NUMERICAL_COLS
logger.info("Model loaded: %s", metadata["best_model_name"])

# ── Database setup ────────────────────────────────────────────────────────────
# Try MySQL first; fall back to SQLite for local dev without MySQL
DB_BACKEND = os.getenv("DB_BACKEND", "sqlite")   # set "mysql" in production

def get_db_connection():
    """Return a DB connection based on DB_BACKEND env var."""
    if DB_BACKEND == "mysql":
        import mysql.connector
        return mysql.connector.connect(
            host     = os.getenv("MYSQL_HOST", "localhost"),
            port     = int(os.getenv("MYSQL_PORT", 3306)),
            user     = os.getenv("MYSQL_USER", "root"),
            password = os.getenv("MYSQL_PASSWORD", ""),
            database = os.getenv("MYSQL_DB", "salary_db"),
        )
    else:
        import sqlite3
        conn = sqlite3.connect(os.path.join(BASE_DIR, "predictions.db"))
        conn.row_factory = sqlite3.Row
        return conn

def init_db():
    """Create predictions table if it doesn't exist."""
    conn = get_db_connection()
    cur  = conn.cursor()
    if DB_BACKEND == "mysql":
        cur.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id               INT AUTO_INCREMENT PRIMARY KEY,
                input_features   JSON         NOT NULL,
                predicted_salary FLOAT        NOT NULL,
                timestamp        DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
        """)
    else:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id               INTEGER PRIMARY KEY AUTOINCREMENT,
                input_features   TEXT    NOT NULL,
                predicted_salary REAL    NOT NULL,
                timestamp        TEXT    NOT NULL
            )
        """)
    conn.commit()
    conn.close()
    logger.info("DB initialized (backend=%s)", DB_BACKEND)

init_db()

def save_prediction(input_features: dict, predicted_salary: float):
    """Insert a prediction record into the DB."""
    try:
        conn = get_db_connection()
        cur  = conn.cursor()
        now  = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        feat_json = json.dumps(input_features)
        if DB_BACKEND == "mysql":
            cur.execute(
                "INSERT INTO predictions (input_features, predicted_salary, timestamp) VALUES (%s, %s, %s)",
                (feat_json, predicted_salary, now)
            )
        else:
            cur.execute(
                "INSERT INTO predictions (input_features, predicted_salary, timestamp) VALUES (?, ?, ?)",
                (feat_json, predicted_salary, now)
            )
        conn.commit()
        conn.close()
        logger.info("Prediction saved: ₹%,.0f", predicted_salary)
    except Exception as e:
        logger.error("DB save error: %s", e)

# ── Routes ─────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html", metadata=metadata)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)
        if not data:
            return jsonify({"error": "No JSON body received"}), 400

        # Validate required fields
        missing = [c for c in ALL_COLS if c not in data]
        if missing:
            return jsonify({"error": f"Missing fields: {missing}"}), 400

        # Build DataFrame in correct column order
        row = {c: [data[c]] for c in ALL_COLS}
        df_input = pd.DataFrame(row)

        # Type coercions
        for col in NUMERICAL_COLS:
            df_input[col] = pd.to_numeric(df_input[col], errors="raise")

        # Predict
        salary = float(model.predict(df_input)[0])
        salary = max(0, round(salary, 2))

        # Persist
        save_prediction(data, salary)

        return jsonify({
            "predicted_salary":  salary,
            "formatted_salary":  f"₹{salary:,.0f}",
            "model_used":        metadata["best_model_name"],
            "input_features":    data,
            "timestamp":         datetime.now().isoformat(),
        })

    except ValueError as ve:
        logger.warning("Validation error: %s", ve)
        return jsonify({"error": str(ve)}), 422
    except Exception as e:
        logger.error("Prediction error: %s", e)
        return jsonify({"error": "Internal server error", "detail": str(e)}), 500

@app.route("/health")
def health():
    return jsonify({"status": "ok", "model": metadata["best_model_name"],
                    "r2": metadata["metrics"]["best_model"]["r2"]})

@app.route("/metadata")
def get_metadata():
    return jsonify(metadata)

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
