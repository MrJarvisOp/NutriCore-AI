#  Diet Health AI

An end-to-end, production-ready health plan generator that takes user biometrics
and an optional medical lab-report PDF, then returns a personalised **diet plan**,
**exercise plan**, and **medical-aware recommendations** — all via a clean REST API.

---

## Architecture

```
Client Input (FastAPI)
        ↓
PDF Processor  (PyMuPDF)
        ↓
OCR Layer      (text-layer first → Tesseract fallback)
        ↓
Medical Parser (Regex → name normalisation → LLM fallback)
        ↓
Health Engine  (100 % deterministic: BMI / BMR / TDEE / macros / flags)
        ↓
LangGraph Orchestrator
   ↙          ↓          ↘
Diet Node  Exercise Node  Validation Node
                ↓
      LLM Response Generator (recommendations only)
                ↓
        Final JSON Response
```

Validation loops back to the Diet Node (up to 3 retries) if calorie/protein
targets or medical constraints are not satisfied.

---

## Tech Stack

| Layer          | Technology                            |
|----------------|---------------------------------------|
| API            | FastAPI + Uvicorn                     |
| Orchestration  | LangGraph                             |
| OCR (text PDF) | PyMuPDF (`fitz`)                      |
| OCR (scanned)  | Tesseract via `pytesseract`           |
| LLM            | LLaMA 3 via Groq API (free tier)      |
| Database       | SQLite (POC) / PostgreSQL (prod)      |
| ORM            | SQLAlchemy 2 async                    |

---

## Project Structure

```
diet_health_project/
├── app/
│   ├── main.py                  # FastAPI app + lifespan
│   ├── config.py                # All settings & constants
│   ├── database.py              # SQLAlchemy ORM + async session
│   ├── api/
│   │   └── routes.py            # REST endpoints
│   ├── services/
│   │   ├── pdf_processor.py     # PyMuPDF page extraction
│   │   ├── ocr_service.py       # Smart OCR (text-layer / Tesseract)
│   │   ├── parser.py            # Hybrid medical-report parser
│   │   ├── health_engine.py     # Deterministic BMI/BMR/TDEE/macro engine
│   │   ├── diet_planner.py      # Hybrid diet builder (DB + rules)
│   │   ├── exercise_planner.py  # Rule-based exercise generator
│   │   ├── validation.py        # Plan validator (loops on failure)
│   │   └── llm_service.py       # Groq / LLaMA 3 wrapper
│   ├── graph/
│   │   └── flow.py              # LangGraph pipeline definition
│   ├── models/
│   │   └── schemas.py           # Pydantic schemas + GraphState
│   └── utils/
│       └── logger.py            # Structured logger
├── data/
│   └── food_db.json             # Food database (calories, macros)
├── tests/
│   └── test_suite.py            # 30+ unit + integration tests
├── .env.example                 # Environment variable template
└── requirements.txt
```

---

## Quick Start

### 1. Clone & configure

```bash
git clone <repo>
cd diet_health_project
cp .env.example .env
# Edit .env and add your GROQ_API_KEY
```

Get a **free** Groq API key at https://console.groq.com

### 2. Install dependencies

```bash
pip install -r requirements.txt

# macOS
brew install tesseract

# Ubuntu / Debian
sudo apt install tesseract-ocr tesseract-ocr-eng
```

### 3. Run the server

```bash
uvicorn app.main:app --reload
# API available at http://localhost:8000
# Docs at        http://localhost:8000/docs
```


---

## API Usage

### POST `/api/v1/health-plan`

**Form fields:**

| Field            | Type    | Required | Description                              |
|------------------|---------|----------|------------------------------------------|
| `age`            | int     | ✅       | Years (10-100)                           |
| `weight`         | float   | ✅       | Kilograms                                |
| `height`         | float   | ✅       | Centimetres                              |
| `goal`           | string  | ✅       | `fat_loss` / `muscle_gain` / `maintenance` |
| `gender`         | string  | ❌       | `male` / `female` / `other` (default: male) |
| `activity_level` | string  | ❌       | `sedentary` / `light` / `moderate` / `active` / `very_active` |
| `dietary_pref`   | string  | ❌       | Comma-separated: `vegetarian,vegan,gluten-free` |
| `allergies`      | string  | ❌       | Comma-separated allergens                |
| `medical_pdf`    | file    | ❌       | PDF lab report                           |

**curl example (no PDF):**

```bash
curl -X POST http://localhost:8000/api/v1/health-plan \
  -F "age=28" \
  -F "weight=75" \
  -F "height=178" \
  -F "goal=fat_loss" \
  -F "gender=male" \
  -F "activity_level=moderate"
```

**curl example (with PDF):**

```bash
curl -X POST http://localhost:8000/api/v1/health-plan \
  -F "age=45" \
  -F "weight=88" \
  -F "height=172" \
  -F "goal=maintenance" \
  -F "gender=female" \
  -F "activity_level=light" \
  -F "medical_pdf=@/path/to/lab_report.pdf"
```

**Sample response (abbreviated):**

```json
{
  "user_summary": { "age": 28, "weight_kg": 75, "goal": "fat_loss" },
  "health_metrics": {
    "bmi": 23.7,
    "bmi_category": "Normal",
    "target_calories": 1952,
    "protein_g": 135.0,
    "carbs_g": 195.2,
    "fat_g": 54.2
  },
  "diet_plan": {
    "daily_calories": 1960,
    "meals": [
      {
        "name": "Breakfast",
        "time": "7:00 AM",
        "foods": [
          { "name": "Oatmeal (1 cup cooked)", "calories": 166, "protein": 6 },
          { "name": "Greek Yogurt (1 cup)",   "calories": 130, "protein": 22 }
        ],
        "total_cal": 490
      }
    ]
  },
  "exercise_plan": {
    "goal": "Fat Loss",
    "weekly_days": 6,
    "workout_days": [ { "day": "Monday", "focus": "Full-Body Strength + Cardio" } ]
  },
  "recommendations": [
    "Eat protein at every meal to preserve muscle while in a calorie deficit.",
    "Aim for 7-9 hours of sleep — poor sleep elevates cortisol and increases cravings."
  ],
  "warnings": [],
  "disclaimer": "This plan is AI-generated for informational purposes only..."
}
```

---

## Design Principles

| Rule | Details |
|------|---------|
| ❌ LLM never does math | All calorie/macro calculations are pure Python |
| ❌ LLM never makes medical decisions | Flags and restrictions are deterministic rule engines |
| ✅ Hybrid parsing | Regex first → LLM only as fallback for medical reports |
| ✅ Validation loops | LangGraph retries plan generation up to 3× if constraints fail |
| ✅ Safety floors | Calories clamped to ≥ 1200 kcal; underweight users redirected |
| ✅ Medical-aware | 10 + medical conditions detected and factored into diet/exercise |

---

## Running Tests

```bash
# All 30+ tests
python -m pytest tests/test_suite.py -v

# Or standalone
python tests/test_suite.py
```

---

## Extending the Project

- **Add foods**: edit `data/food_db.json` — no code changes needed.
- **Add medical flags**: extend `parser.py` PATTERNS + `health_engine.py` flag rules.
- **Switch to PostgreSQL**: change `DATABASE_URL` in `.env`.
- **Add embeddings**: install `sentence-transformers` and add a semantic food-search node.
- **Frontend**: the FastAPI `/docs` Swagger UI works out of the box for demos.

---

## Disclaimer

This software is for **informational and educational purposes only**.
It is not a substitute for professional medical advice, diagnosis, or treatment.
Always consult a qualified healthcare provider before changing your diet or exercise routine.
