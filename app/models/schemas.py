from __future__ import annotations
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, validator
from enum import Enum


# ─── Enums ───────────────────────────────────────────────────────────────────

class Goal(str, Enum):
    fat_loss    = "fat_loss"
    muscle_gain = "muscle_gain"
    maintenance = "maintenance"

class Gender(str, Enum):
    male   = "male"
    female = "female"
    other  = "other"

class ActivityLevel(str, Enum):
    sedentary    = "sedentary"
    light        = "light"
    moderate     = "moderate"
    active       = "active"
    very_active  = "very_active"


# ─── Input ───────────────────────────────────────────────────────────────────

class UserInput(BaseModel):
    age:            int            = Field(..., ge=10, le=100)
    weight:         float          = Field(..., gt=0, le=300, description="kg")
    height:         float          = Field(..., gt=0, le=250, description="cm")
    goal:           Goal
    gender:         Gender         = Gender.male
    activity_level: ActivityLevel  = ActivityLevel.moderate
    dietary_pref:   List[str]      = Field(default_factory=list,
                                          description="vegetarian, vegan, gluten-free…")
    allergies:      List[str]      = Field(default_factory=list)

    @validator("height")
    def height_reasonable(cls, v):
        if v < 50:
            raise ValueError("Height seems too low — did you mean cm?")
        return v


# ─── Medical ─────────────────────────────────────────────────────────────────

class MedicalData(BaseModel):
    cholesterol:         Optional[float] = None
    ldl:                 Optional[float] = None
    hdl:                 Optional[float] = None
    triglycerides:       Optional[float] = None
    blood_sugar:         Optional[float] = None
    hba1c:               Optional[float] = None
    hemoglobin:          Optional[float] = None
    blood_pressure:      Optional[str]   = None
    thyroid_tsh:         Optional[float] = None
    vitamin_d:           Optional[float] = None
    vitamin_b12:         Optional[float] = None
    creatinine:          Optional[float] = None
    # Extended fields (Task 3)
    iron:                Optional[float] = None   # µg/dL serum iron
    ferritin:            Optional[float] = None   # ng/mL
    alt:                 Optional[float] = None   # liver ALT U/L
    ast:                 Optional[float] = None   # liver AST U/L
    urea:                Optional[float] = None   # kidney BUN mg/dL
    uric_acid:           Optional[float] = None   # mg/dL
    sodium:              Optional[float] = None   # mEq/L
    potassium:           Optional[float] = None   # mEq/L
    flags:               List[str]       = Field(default_factory=list)
    raw_text:            Optional[str]   = None


# ─── Health Engine output ─────────────────────────────────────────────────────

class HealthMetrics(BaseModel):
    bmi:              float
    bmi_category:     str
    bmr:              float
    tdee:             float
    target_calories:  int
    protein_g:        float
    carbs_g:          float
    fat_g:            float
    medical_flags:    List[str] = Field(default_factory=list)
    warnings:         List[str] = Field(default_factory=list)


# ─── Meal / Diet ─────────────────────────────────────────────────────────────

class FoodItem(BaseModel):
    name:     str
    calories: int
    protein:  float
    carbs:    float
    fat:      float
    fiber:    float = 0.0
    quantity: str   = "1 serving"

class Meal(BaseModel):
    name:       str
    time:       str
    foods:      List[FoodItem]
    total_cal:  int
    total_prot: float
    total_carb: float
    total_fat:  float

class DietPlan(BaseModel):
    daily_calories:  int
    daily_protein:   float
    daily_carbs:     float
    daily_fat:       float
    meals:           List[Meal]
    hydration_liters: float
    notes:           List[str] = Field(default_factory=list)


# ─── Exercise ────────────────────────────────────────────────────────────────

class ExerciseItem(BaseModel):
    name:     str
    sets:     Optional[int]    = None
    reps:     Optional[str]    = None
    duration: Optional[str]    = None
    rest:     Optional[str]    = None
    notes:    Optional[str]    = None

class WorkoutDay(BaseModel):
    day:       str
    focus:     str
    exercises: List[ExerciseItem]
    duration:  str
    intensity: str

class ExercisePlan(BaseModel):
    goal:         str
    weekly_days:  int
    workout_days: List[WorkoutDay]
    cardio_note:  Optional[str] = None
    warnings:     List[str]     = Field(default_factory=list)


# ─── Validation result ───────────────────────────────────────────────────────

class ValidationFeedback(BaseModel):
    """Structured correction instructions passed to diet_node on retry.
    All adjustments are signed integers/floats (positive = add, negative = subtract).
    """
    calorie_adjustment:  int   = 0      # kcal to add or remove
    protein_adjustment:  float = 0.0    # grams to add or remove
    carb_adjustment:     float = 0.0
    fat_adjustment:      float = 0.0
    issues_summary:      List[str] = Field(default_factory=list)

    @classmethod
    def from_validation_result(
        cls,
        result: "ValidationResult",
        metrics: HealthMetrics,
        diet: DietPlan,
    ) -> "ValidationFeedback":
        """Compute signed correction deltas from what validation found wrong."""
        cal_adj  = metrics.target_calories - diet.daily_calories
        prot_adj = metrics.protein_g       - diet.daily_protein
        carb_adj = metrics.carbs_g         - diet.daily_carbs
        fat_adj  = metrics.fat_g           - diet.daily_fat
        return cls(
            calorie_adjustment=int(cal_adj),
            protein_adjustment=round(prot_adj, 1),
            carb_adjustment=round(carb_adj, 1),
            fat_adjustment=round(fat_adj, 1),
            issues_summary=result.issues,
        )


class ValidationResult(BaseModel):
    passed:   bool
    issues:   List[str]              = Field(default_factory=list)
    details:  Dict[str, Any]         = Field(default_factory=dict)
    feedback: Optional[ValidationFeedback] = None


# ─── Final response ──────────────────────────────────────────────────────────

class HealthPlanResponse(BaseModel):
    user_summary:      Dict[str, Any]
    health_metrics:    HealthMetrics
    medical_data:      Optional[MedicalData] = None
    diet_plan:         DietPlan
    exercise_plan:     ExercisePlan
    recommendations:   List[str]
    warnings:          List[str]
    validation_result: Optional[Any] = None          # expose to templates
    web_search_context: Optional[Dict[str, Any]] = None  # Tavily results
    disclaimer:        str = (
        "This plan is AI-generated for informational purposes only. "
        "Always consult a qualified healthcare professional before making "
        "changes to your diet or exercise routine, especially if you have "
        "a medical condition."
    )


# ─── LangGraph state ─────────────────────────────────────────────────────────

class GraphState(BaseModel):
    user_input:         UserInput
    pdf_path:           Optional[str]              = None
    pdf_text:           Optional[str]              = None
    medical_data:       Optional[MedicalData]      = None
    health_metrics:     Optional[HealthMetrics]    = None
    diet_plan:          Optional[DietPlan]         = None
    exercise_plan:      Optional[ExercisePlan]     = None
    validation_result:  Optional[ValidationResult] = None
    # Task 1: retry_count is only mutated by retry_node — never by the router
    retry_count:        int                        = 0
    # Task 2: structured feedback from validator → consumed by diet_node
    validation_feedback: Optional[ValidationFeedback] = None
    error:              Optional[str]              = None
    llm_context:        Optional[str]              = None
    # Tavily web search results enriching the plan
    web_search_results: Optional[Dict[str, Any]]   = None

    class Config:
        arbitrary_types_allowed = True
