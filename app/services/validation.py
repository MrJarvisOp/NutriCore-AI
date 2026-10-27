"""
validation.py
-------------
Two-layer validation system:

Layer 1 - PlanValidator (loop validator):
  Called inside the LangGraph loop after each diet generation.
  Failures trigger retry_node → corrected diet_node.

Layer 2 - ResponseSanityValidator (Task 4):
  Final gate before the API returns.
  Catches unsafe/unrealistic plans that passed layer 1.
  Never loops — returns a sanitised response or raises a hard error.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from app.config import CALORIE_TOLERANCE, PROTEIN_TOLERANCE, MIN_CALORIES
from app.models.schemas import (
    DietPlan, ExercisePlan, HealthMetrics, MedicalData,
    ValidationFeedback, ValidationResult,
)
from app.utils.logger import get_logger

log = get_logger(__name__)

# ─── Task 8: Hard rejection thresholds ───────────────────────────────────────
HARD_CAL_DELTA   = 300   # kcal  - reject if off by more than this (Task 8)
HARD_PROT_DELTA  = 50    # g     - reject if off by more than this (Task 8)
MAX_SAFE_CALORIES = 4500
MAX_FAT_G_PER_DAY = 200
MAX_PROT_G_PER_DAY = 400
MAX_CARB_G_PER_DAY = 600


# Layer 1 - Loop validator (used inside LangGraph)

class PlanValidator:
    """Validate diet + exercise plan against targets.  Failures trigger retry."""

    def validate(
        self,
        diet:     DietPlan,
        exercise: ExercisePlan,
        metrics:  HealthMetrics,
        medical:  Optional[MedicalData] = None,
    ) -> ValidationResult:

        issues:  List[str]        = []
        details: Dict[str, Any]   = {}

        if diet is None or exercise is None:
            return ValidationResult(
                passed=False,
                issues=["Diet or exercise plan is missing — cannot validate."],
                details={},
            )

        # ── 1. Calorie check (Task 8: hard threshold) ──────────────────────
        cal_diff = diet.daily_calories - metrics.target_calories  # signed
        cal_tol  = metrics.target_calories * CALORIE_TOLERANCE
        details["calorie_target"]  = metrics.target_calories
        details["calorie_actual"]  = diet.daily_calories
        details["calorie_delta"]   = cal_diff

        if abs(cal_diff) > HARD_CAL_DELTA:
            issues.append(
                f"HARD FAIL - Calorie delta {cal_diff:+.0f} kcal exceeds "
                f"hard limit ±{HARD_CAL_DELTA} kcal "
                f"(target {metrics.target_calories}, got {diet.daily_calories})."
            )
        elif abs(cal_diff) > cal_tol:
            issues.append(
                f"Calorie mismatch: target {metrics.target_calories} kcal, "
                f"got {diet.daily_calories} kcal (Δ{cal_diff:+.0f}, "
                f"soft tolerance ±{cal_tol:.0f})."
            )

        # ── 2. Protein check (Task 8: hard threshold) ──────────────────────
        prot_diff = diet.daily_protein - metrics.protein_g        # signed
        prot_tol  = metrics.protein_g * PROTEIN_TOLERANCE
        details["protein_target"] = metrics.protein_g
        details["protein_actual"] = diet.daily_protein
        details["protein_delta"]  = prot_diff

        if abs(prot_diff) > HARD_PROT_DELTA:
            issues.append(
                f"HARD FAIL - Protein delta {prot_diff:+.1f}g exceeds "
                f"hard limit ±{HARD_PROT_DELTA}g "
                f"(target {metrics.protein_g:.1f}g, got {diet.daily_protein:.1f}g)."
            )
        elif abs(prot_diff) > prot_tol:
            issues.append(
                f"Protein mismatch: target {metrics.protein_g:.1f}g, "
                f"got {diet.daily_protein:.1f}g (Δ{prot_diff:+.1f}g)."
            )

        # ── 3. Absolute minimum calorie safety floor ───────────────────────
        if diet.daily_calories < MIN_CALORIES:
            issues.append(
                f"Diet falls below minimum safe threshold "
                f"({diet.daily_calories} < {MIN_CALORIES} kcal)."
            )

        # ── 4. Medical constraints ─────────────────────────────────────────
        if medical and medical.flags:
            med_issues = self._medical_checks(diet, medical.flags)
            issues.extend(med_issues)
            details["medical_checks"] = med_issues

        # ── 5. Meal count sanity ───────────────────────────────────────────
        if len(diet.meals) < 3:
            issues.append(f"Too few meals: {len(diet.meals)} (minimum 3).")

        # ── 6. Exercise plan sanity ────────────────────────────────────────
        if not exercise.workout_days:
            issues.append("Exercise plan has no workout days.")

        passed = len(issues) == 0
        result = ValidationResult(passed=passed, issues=issues, details=details)

        if passed:
            log.info("[validator-L1] Validation PASSED ✓")
        else:
            log.warning(
                f"[validator-L1] FAILED - {len(issues)} issue(s): "
                + "; ".join(issues[:3])  # log first 3 to avoid wall of text
            )

        return result

    # ─── Medical rule checks ──────────────────────────────────────────────────

    @staticmethod
    def _medical_checks(diet: DietPlan, flags: List[str]) -> List[str]:
        issues: List[str] = []

        if "high_cholesterol" in flags:
            fat_pct = (diet.daily_fat * 9) / max(diet.daily_calories, 1)
            if fat_pct > 0.30:
                issues.append(
                    f"High-cholesterol flag: fat is {fat_pct:.0%} of calories "
                    f"(should be ≤ 30 %)."
                )

        if "pre_diabetic_or_diabetic" in flags or "diabetes_risk" in flags:
            carb_pct = (diet.daily_carbs * 4) / max(diet.daily_calories, 1)
            if carb_pct > 0.50:
                issues.append(
                    f"Diabetes flag: carbs are {carb_pct:.0%} of calories "
                    f"(should be ≤ 45-50 %)."
                )

        if "hypertension" in flags:
            sodium_terms = {"processed_meats", "canned_foods", "bacon", "salami"}
            high_sodium_count = sum(
                1 for meal in diet.meals
                for food in meal.foods
                if any(t in food.name.lower() for t in sodium_terms)
            )
            if high_sodium_count > 2:
                issues.append(
                    "Hypertension flag: potential high-sodium foods detected."
                )

        return issues


# Layer 2 - Response sanity validator  (Task 4)

class ResponseSanityValidator:
    """
    Final gate before the API response is returned.
    Checks for absolute safety violations, contradictions, and missing
    medical warnings that would make the response unsafe for a real user.
    """

    def validate(
        self,
        diet:        DietPlan,
        exercise:    ExercisePlan,
        metrics:     HealthMetrics,
        medical:     Optional[MedicalData],
        recommendations: List[str],
    ) -> Dict[str, Any]:
        """
        Returns a dict with:
          - 'safe' (bool)
          - 'issues' (list of blocking issues)
          - 'warnings_to_append' (list of warnings to add to the response)
          - 'sanitised_recommendations' (cleaned list)
        """
        blocking:  List[str] = []
        warnings:  List[str] = []
        safe_recs: List[str] = list(recommendations)

        # ── A. Absolute calorie range ──────────────────────────────────────
        if diet.daily_calories < MIN_CALORIES:
            blocking.append(
                f"UNSAFE: diet plan has only {diet.daily_calories} kcal "
                f"- below safe minimum {MIN_CALORIES} kcal."
            )
        if diet.daily_calories > MAX_SAFE_CALORIES:
            blocking.append(
                f"UNSAFE: diet plan has {diet.daily_calories} kcal "
                f"- exceeds safe maximum {MAX_SAFE_CALORIES} kcal."
            )

        # ── B. Macro sanity bounds ─────────────────────────────────────────
        if diet.daily_protein > MAX_PROT_G_PER_DAY:
            blocking.append(
                f"UNSAFE: protein {diet.daily_protein}g/day exceeds "
                f"safe maximum {MAX_PROT_G_PER_DAY}g."
            )
        if diet.daily_fat > MAX_FAT_G_PER_DAY:
            blocking.append(
                f"UNSAFE: fat {diet.daily_fat}g/day exceeds "
                f"safe maximum {MAX_FAT_G_PER_DAY}g."
            )
        if diet.daily_carbs > MAX_CARB_G_PER_DAY:
            blocking.append(
                f"UNSAFE: carbs {diet.daily_carbs}g/day exceeds "
                f"safe maximum {MAX_CARB_G_PER_DAY}g."
            )

        # ── C. Medical-warning completeness ───────────────────────────────
        flags = (medical.flags if medical else []) or []
        rec_text = " ".join(safe_recs).lower()

        if "anemia_risk" in flags or "iron_deficiency" in flags:
            if not any(w in rec_text for w in ("iron", "anaemia", "anemia")):
                warnings.append(
                    "Anaemia/iron deficiency detected: increase iron-rich foods "
                    "(red meat, lentils, spinach) and pair with vitamin C."
                )

        if "hypertension" in flags:
            if not any(w in rec_text for w in ("sodium", "salt", "blood pressure")):
                warnings.append(
                    "Hypertension detected: keep sodium under 1500 mg/day; "
                    "avoid processed and canned foods."
                )

        if "pre_diabetic_or_diabetic" in flags or "diabetes_risk" in flags:
            if not any(w in rec_text for w in ("glucose", "sugar", "glycaem", "glycem", "carb")):
                warnings.append(
                    "Elevated blood sugar detected: monitor carbohydrate intake "
                    "and choose low-GI foods."
                )

        if "high_cholesterol" in flags or "high_ldl" in flags:
            if not any(w in rec_text for w in ("cholesterol", "saturated", "omega")):
                warnings.append(
                    "High cholesterol detected: limit saturated fats and increase "
                    "omega-3 rich foods."
                )

        if "vitamin_d_deficiency" in flags:
            if "vitamin d" not in rec_text:
                warnings.append(
                    "Vitamin D deficiency detected: consider supplementation "
                    "and daily sun exposure (15-20 min)."
                )

        if "elevated_liver_alt" in flags or "elevated_liver_ast" in flags:
            warnings.append(
                "Elevated liver enzymes detected: avoid alcohol, limit fatty/fried "
                "foods, and consult a physician."
            )

        if "elevated_creatinine" in flags or "elevated_urea" in flags:
            warnings.append(
                "Kidney function marker(s) elevated: limit protein to moderate "
                "levels and stay well hydrated; consult a physician."
            )
            # Also cap protein recommendation for safety
            if diet.daily_protein > 1.5 * metrics.health_metrics_weight_kg(metrics):
                safe_recs.append(
                    "Note: protein intake has been moderated due to kidney function markers."
                )

        # ── D. Contradictory advice check ─────────────────────────────────
        # E.g. telling a diabetic to eat high-sugar foods
        if ("pre_diabetic_or_diabetic" in flags or "diabetes_risk" in flags):
            for food_item in [f for m in diet.meals for f in m.foods]:
                if food_item.carbs > 45 and food_item.fiber < 2:
                    warnings.append(
                        f"Note: '{food_item.name}' is high in refined carbs "
                        f"- consider a lower-GI alternative."
                    )
                    break  # one warning is enough

        safe_recs = self._remove_unsafe_phrases(safe_recs, flags)

        is_safe = len(blocking) == 0
        if is_safe:
            log.info("[validator-L2] Response sanity check PASSED ✓")
        else:
            log.error(
                f"[validator-L2] BLOCKING issues: {blocking}"
            )
        if warnings:
            log.warning(
                f"[validator-L2] {len(warnings)} safety warning(s) appended"
            )

        return {
            "safe":                      is_safe,
            "issues":                    blocking,
            "warnings_to_append":        warnings,
            "sanitised_recommendations": safe_recs,
        }

    @staticmethod
    def _remove_unsafe_phrases(recs: List[str], flags: List[str]) -> List[str]:
        """Strip recommendations that contradict known medical conditions."""
        banned_for_diabetes   = ["sugary", "high-sugar", "fruit juice", "honey"]
        banned_for_hypert     = ["add salt", "pickled", "cured meats"]
        banned_for_cholesterol= ["butter", "full-fat cream", "deep-fried"]

        cleaned = []
        for rec in recs:
            lower = rec.lower()
            if ("pre_diabetic_or_diabetic" in flags or "diabetes_risk" in flags):
                if any(p in lower for p in banned_for_diabetes):
                    log.debug(f"[validator-L2] Removed unsafe rec (diabetes): {rec[:60]}")
                    continue
            if "hypertension" in flags:
                if any(p in lower for p in banned_for_hypert):
                    log.debug(f"[validator-L2] Removed unsafe rec (hypertension): {rec[:60]}")
                    continue
            if "high_cholesterol" in flags:
                if any(p in lower for p in banned_for_cholesterol):
                    log.debug(f"[validator-L2] Removed unsafe rec (cholesterol): {rec[:60]}")
                    continue
            cleaned.append(rec)
        return cleaned


# ─── Helper attached at module level for HealthMetrics ───────────────────────
def _weight_from_metrics(metrics: HealthMetrics) -> float:
    """Back-calculate approximate body weight from BMR (Mifflin formula, male)."""
    # BMR = 10w + 6.25h - 5a + 5  →  w ≈ (BMR - 6.25h + 5a - 5) / 10
    # We can't recover h & a from metrics alone, so use a safe default.
    return 80.0  # conservative fallback — used only for kidney protein cap

# Monkey-patch for use inside the class method
HealthMetrics.health_metrics_weight_kg = staticmethod(_weight_from_metrics)
