# SalaryIQ — IT Salary Prediction System

> **Model**: Linear Regression | **R² = 0.9737** | **MAE ≈ ₹64,852**  
> **Stack**: Python · scikit-learn · Flask · MySQL/SQLite · Bootstrap 5

---

## Project Structure

```
salary_prediction/
├── salaries.csv            ← raw dataset (100,000 rows)
├── train.py                ← ML pipeline (preprocessing → training → saving)
├── app.py                  ← Flask REST API
├── setup_mysql.sql         ← MySQL schema creation script
├── render.yaml             ← Render.com deployment config
├── requirements.txt
├── model/
│   ├── best_model.pkl      ← saved best model (auto-generated)
│   └── metadata.json       ← feature options, metrics (auto-generated)
└── templates/
    └── index.html          ← Bootstrap frontend
```

---

## ML Pipeline Summary

### Data Filters Applied
- Industry: **IT only** (25,045 rows → 9,511 after job-title filter)
- Job titles containing: Software, Developer, Engineer, Data, AI, Machine Learning, DevOps, Cloud, Backend, Frontend, Full Stack, Cyber

### Preprocessing
| Step | Details |
|---|---|
| Drop | `Unnamed: 0`, `employee_id`, `industry` (constant after filter) |
| OneHotEncoding | gender, education, job_title, company_size, location, remote_work |
| StandardScaler | age, experience_years, skills_score, certifications, performance_rating |
| VIF check | All features VIF < 5 → none dropped |
| Train/Test split | 80% / 20% (7,608 / 1,903) |

### Model Comparison
| Model | R² | MAE | RMSE |
|---|---|---|---|
| Linear Regression | **0.9737** | ₹64,852 | ₹81,278 |
| Polynomial (deg=2) | 0.9736 | ₹64,994 | ₹81,388 |

**Winner: Linear Regression** (marginally better, far less complexity)

### 5-Fold Cross Validation
- Fold scores: [0.9734, 0.9742, 0.9752, 0.9744, 0.9752]
- **Mean R²: 0.9745 ± 0.0007** ← extremely stable

---

## Local Setup & Run

### 1. Prerequisites
```bash
python >= 3.10
pip
MySQL (optional — SQLite fallback works out of the box)
```

### 2. Install dependencies
```bash
cd salary_prediction
pip install -r requirements.txt
```

### 3. Train the model
```bash
python train.py
# Creates model/best_model.pkl and model/metadata.json
```

### 4a. Run with SQLite (no MySQL needed)
```bash
# DB_BACKEND defaults to sqlite
python app.py
```
Open: http://localhost:5000

### 4b. Run with MySQL
```bash
# First, set up the DB schema:
mysql -u root -p < setup_mysql.sql

# Then start the app with MySQL env vars:
DB_BACKEND=mysql \
MYSQL_HOST=localhost \
MYSQL_PORT=3306 \
MYSQL_USER=salary_user \
MYSQL_PASSWORD=yourpassword \
MYSQL_DB=salary_db \
python app.py
```

---

## API Reference

### POST `/predict`
**Request body (JSON):**
```json
{
  "gender": "Male",
  "education": "Master",
  "job_title": "Software Engineer",
  "company_size": "Large",
  "location": "Bangalore",
  "remote_work": "Yes",
  "age": 30,
  "experience_years": 7,
  "skills_score": 85,
  "certifications": 3,
  "performance_rating": 4
}
```

**Response:**
```json
{
  "predicted_salary": 2874500.0,
  "formatted_salary": "₹28,74,500",
  "model_used": "Linear Regression",
  "input_features": { "..." },
  "timestamp": "2025-01-15T14:32:10.123456"
}
```

### GET `/health`
```json
{ "status": "ok", "model": "Linear Regression", "r2": 0.9737 }
```

### GET `/metadata`
Returns all feature options and model metrics.

---

## Database Schema (MySQL)

```sql
CREATE TABLE predictions (
    id               INT AUTO_INCREMENT PRIMARY KEY,
    input_features   JSON     NOT NULL,
    predicted_salary FLOAT    NOT NULL,
    timestamp        DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
);
```

Every `/predict` call inserts one row automatically.

---

## Deploy on Render.com

### Step-by-Step

1. **Push to GitHub**
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git remote add origin https://github.com/youruser/salaryiq.git
   git push -u origin main
   ```

2. **Create Web Service on Render**
   - Go to https://render.com → New → Web Service
   - Connect your GitHub repo
   - Render auto-detects `render.yaml`

3. **Configure Environment Variables** (in Render dashboard)
   | Key | Value |
   |---|---|
   | `DB_BACKEND` | `sqlite` (or `mysql`) |
   | `MYSQL_HOST` | from Render MySQL add-on |
   | `MYSQL_USER` | from Render MySQL add-on |
   | `MYSQL_PASSWORD` | from Render MySQL add-on |
   | `MYSQL_DB` | `salary_db` |

4. **Add MySQL Add-on** (optional)
   - Render Dashboard → your service → Add-ons → MySQL
   - Copy connection details to env vars above

5. **Deploy**
   - Click Deploy — Render runs `train.py` then starts `gunicorn`
   - Access your app at `https://salaryiq.onrender.com`

### Important: Add `salaries.csv` to repo
Render needs the CSV to run `train.py` during build. Either commit the CSV or upload it as a Render Disk.

---

## Tech Stack

| Layer | Technology |
|---|---|
| ML | scikit-learn (LinearRegression, PolynomialFeatures, StandardScaler, OneHotEncoder) |
| Backend | Flask 3.x |
| Database | MySQL (production) / SQLite (dev fallback) |
| Frontend | HTML5 + Bootstrap 5 + Vanilla JS |
| Model persistence | joblib |
| Deployment | Render.com + gunicorn |

---

## Key Design Decisions

- **VIF analysis** implemented from scratch using NumPy `lstsq` (no statsmodels needed)
- **SQLite fallback** allows the app to run locally without any DB setup
- **Pipeline** encapsulates preprocessing + model so prediction is a single `.predict()` call
- **Metadata JSON** drives the frontend dynamically — no hardcoded dropdown values
