"""
health_engine.py
----------------
100 % deterministic - NO LLM involvement.
Calculates BMI, BMR, TDEE, calorie targets, macros, and medical flags.
"""
from __future__ import annotations

from typing import List, Optional

from app.config import (
    BMI_UNDERWEIGHT, BMI_NORMAL_MAX, BMI_OVERWEIGHT_MAX,
    CALORIE_ADJUSTMENTS, MACRO_RATIOS, MIN_CALORIES,
    PROTEIN_TARGETS,
)
from app.models.schemas import (
    ActivityLevel, Gender, Goal,
    HealthMetrics, MedicalData, UserInput,
)
from app.utils.logger import get_logger

log = get_logger(__name__)

# Activity multipliers for TDEE (Mifflin-St Jeor)
ACTIVITY_MULTIPLIERS = {
    ActivityLevel.sedentary:   1.2,
    ActivityLevel.light:       1.375,
    ActivityLevel.moderate:    1.55,
    ActivityLevel.active:      1.725,
    ActivityLevel.very_active: 1.9,
}


class HealthEngine:
    """Deterministic health calculation layer."""

    def __init__(self, user: UserInput, medical: Optional[MedicalData] = None):
        self.user    = user
        self.medical = medical

    # ─── Public ──────────────────────────────────────────────────────────────

    def calculate(self) -> HealthMetrics:
        bmi, bmi_cat = self._bmi()
        bmr          = self._bmr()
        tdee         = self._tdee(bmr)
        target_cal   = self._target_calories(tdee, bmi)
        protein, carbs, fat = self._macros(target_cal)
        flags, warnings = self._medical_flags()

        metrics = HealthMetrics(
            bmi=round(bmi, 1),
            bmi_category=bmi_cat,
            bmr=round(bmr, 0),
            tdee=round(tdee, 0),
            target_calories=target_cal,
            protein_g=round(protein, 1),
            carbs_g=round(carbs, 1),
            fat_g=round(fat, 1),
            medical_flags=flags,
            warnings=warnings,
        )
        log.info(
            f"Health metrics - BMI:{metrics.bmi} | Target:{metrics.target_calories} kcal | "
            f"P:{metrics.protein_g}g C:{metrics.carbs_g}g F:{metrics.fat_g}g"
        )
        return metrics

    # ─── BMI ─────────────────────────────────────────────────────────────────

    def _bmi(self):
        h_m = self.user.height / 100
        bmi = self.user.weight / (h_m ** 2)
        if bmi < BMI_UNDERWEIGHT:
            cat = "Underweight"
        elif bmi <= BMI_NORMAL_MAX:
            cat = "Normal"
        elif bmi <= BMI_OVERWEIGHT_MAX:
            cat = "Overweight"
        else:
            cat = "Obese"
        return bmi, cat

    # ─── BMR (Mifflin-St Jeor) ────────────────────────────────────────────────

    def _bmr(self) -> float:
        w, h, a = self.user.weight, self.user.height, self.user.age
        if self.user.gender == Gender.female:
            return 10 * w + 6.25 * h - 5 * a - 161
        else:                                    # male / other
            return 10 * w + 6.25 * h - 5 * a + 5

    def _tdee(self, bmr: float) -> float:
        mult = ACTIVITY_MULTIPLIERS.get(self.user.activity_level, 1.55)
        return bmr * mult

    # ─── Calorie target ───────────────────────────────────────────────────────

    def _target_calories(self, tdee: float, bmi: float) -> int:
        adj = CALORIE_ADJUSTMENTS.get(self.user.goal, 0)

        # Safety: don't allow extreme deficit for underweight users
        if bmi < BMI_UNDERWEIGHT and self.user.goal == Goal.fat_loss:
            log.warning("User is underweight - overriding fat_loss to maintenance")
            adj = 0

        # Medical adjustment: hypothyroidism → reduce calories a bit
        if self.medical and "hypothyroidism_risk" in (self.medical.flags or []):
            adj -= 100

        raw = tdee + adj
        # Clamp to safe range
        return int(max(MIN_CALORIES, min(raw, 4500)))

    # ─── Macros ───────────────────────────────────────────────────────────────

    def _macros(self, calories: int):
        ratios = dict(MACRO_RATIOS[self.user.goal])   # copy

        # Adjust for medical flags
        if self.medical:
            flags = self.medical.flags or []
            if "high_cholesterol" in flags or "high_triglycerides" in flags:
                # Reduce fat, boost carbs slightly
                ratios["fat"]   = max(0.15, ratios["fat"] - 0.05)
                ratios["carbs"] += 0.05
            if "pre_diabetic_or_diabetic" in flags or "diabetes_risk" in flags:
                # Reduce carbs, boost protein/fat
                ratios["carbs"]   = max(0.30, ratios["carbs"] - 0.10)
                ratios["protein"] = min(0.40, ratios["protein"] + 0.05)
                ratios["fat"]    += 0.05
            # Re-normalise
            total = sum(ratios.values())
            ratios = {k: v / total for k, v in ratios.items()}

        protein_g = (calories * ratios["protein"]) / 4
        carbs_g   = (calories * ratios["carbs"])   / 4
        fat_g     = (calories * ratios["fat"])      / 9

        # Override if protein target by body weight is higher
        min_prot = self.user.weight * PROTEIN_TARGETS[self.user.goal]
        if protein_g < min_prot:
            deficit   = (min_prot - protein_g) * 4    # kcal to shift
            protein_g = min_prot
            carbs_g   = max(50, carbs_g - deficit / 4)

        return protein_g, carbs_g, fat_g

    # ─── Medical flags ────────────────────────────────────────────────────────

    def _medical_flags(self):
        flags:    List[str] = []
        warnings: List[str] = []

        if self.medical and self.medical.flags:
            flags = list(self.medical.flags)

        # BMI warnings
        bmi, cat = self._bmi()
        if cat == "Obese":
            warnings.append(
                "BMI indicates obesity - consult a physician before starting this plan.")
        elif cat == "Underweight":
            warnings.append(
                "BMI indicates underweight - calorie target adjusted to maintenance.")

        # Age
        if self.user.age >= 60:
            warnings.append(
                "Senior user: protein targets raised; impact exercises should be low-intensity.")
        elif self.user.age < 18:
            warnings.append(
                "Minor user: calorie targets are conservative; consult a paediatrician.")

        # Medical-specific warnings
        if "high_cholesterol" in flags:
            warnings.append("High cholesterol detected - low-saturated-fat diet recommended.")
        if "pre_diabetic_or_diabetic" in flags or "diabetes_risk" in flags:
            warnings.append(
                "Elevated blood sugar - low-glycaemic diet and regular monitoring advised.")
        if "hypertension" in flags:
            warnings.append(
                "High blood pressure - sodium restricted to <1500 mg/day.")
        if "anemia_risk" in flags:
            warnings.append("Possible anaemia - iron-rich foods and vitamin C pairing recommended.")
        if "hypothyroidism_risk" in flags:
            warnings.append(
                "Possible hypothyroidism - calorie target slightly reduced; consult an endocrinologist.")
        if "vitamin_d_deficiency" in flags:
            warnings.append("Vitamin D deficiency - supplement and sun exposure recommended.")

        return flags, warnings
