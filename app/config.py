import os
from dotenv import load_dotenv

load_dotenv()

APP_NAME = "Diet Health AI"
APP_VERSION = "1.0.0"
DEBUG = os.getenv("DEBUG", "false").lower() == "true"

GROQ_API_KEY    = os.getenv("GROQ_API_KEY", "")
LLM_MODEL       = os.getenv("LLM_MODEL", "llama3-8b-8192")
LLM_MAX_TOKENS  = int(os.getenv("LLM_MAX_TOKENS", "2048"))
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.3"))

# Tavily web search
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")

DATABASE_URL  = os.getenv("DATABASE_URL", "sqlite:///./diet_health.db")
TESSERACT_CMD = os.getenv("TESSERACT_CMD", "/usr/bin/tesseract")

BASE_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FOOD_DB_PATH = os.path.join(BASE_DIR, "data", "food_db.json")
UPLOAD_DIR   = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

BMI_UNDERWEIGHT    = 18.5
BMI_NORMAL_MAX     = 24.9
BMI_OVERWEIGHT_MAX = 29.9

MACRO_RATIOS = {
    "fat_loss":    {"protein": 0.35, "carbs": 0.40, "fat": 0.25},
    "muscle_gain": {"protein": 0.30, "carbs": 0.50, "fat": 0.20},
    "maintenance": {"protein": 0.25, "carbs": 0.50, "fat": 0.25},
}

CALORIE_ADJUSTMENTS = {
    "fat_loss":    -500,
    "muscle_gain": +300,
    "maintenance":    0,
}

PROTEIN_TARGETS = {
    "fat_loss":    1.8,
    "muscle_gain": 2.2,
    "maintenance": 1.4,
}

MIN_CALORIES       = 1200
MAX_CALORIES       = 4500
CALORIE_TOLERANCE  = 0.05
PROTEIN_TOLERANCE  = 0.10

CHOLESTEROL_HIGH             = 200
BLOOD_SUGAR_HIGH             = 100
BLOOD_PRESSURE_SYSTOLIC_HIGH = 130
HEMOGLOBIN_LOW_MALE          = 13.5
HEMOGLOBIN_LOW_FEMALE        = 12.0
TRIGLYCERIDES_HIGH           = 150
