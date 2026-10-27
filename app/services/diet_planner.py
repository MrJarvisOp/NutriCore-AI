"""
diet_planner.py
---------------
Hybrid diet planner:
  • Food database provides calorie/macro data (ground truth).
  • Python does ALL arithmetic — LLM never touches numbers.
  • On retry, ValidationFeedback drives targeted corrections instead of
    random re-generation.  (Task 2)
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List, Optional

from app.config import FOOD_DB_PATH
from app.models.schemas import (
    DietPlan, FoodItem, Goal, HealthMetrics, Meal,
    MedicalData, UserInput, ValidationFeedback,
)
from app.utils.logger import get_logger

log = get_logger(__name__)

# High-protein foods to inject when protein is short on retry
_HIGH_PROTEIN_BOOST: List[Dict] = [
    {"name": "Extra Egg Whites (3)",        "calories": 51,  "protein": 11, "carbs": 1,  "fat": 0,  "fiber": 0},
    {"name": "Low-fat Cottage Cheese (½c)", "calories": 90,  "protein": 13, "carbs": 4,  "fat": 2,  "fiber": 0},
    {"name": "Whey Protein Scoop",          "calories": 120, "protein": 24, "carbs": 3,  "fat": 2,  "fiber": 0},
    {"name": "Turkey Breast Slices (50g)",  "calories": 59,  "protein": 12, "carbs": 0,  "fat": 1,  "fiber": 0},
]

# Low-calorie fillers to trim when calories are over target on retry
_LOW_CAL_SWAP: Dict[str, Dict] = {
    "default": {"name": "Cucumber Slices (1 cup)", "calories": 16,  "protein": 1,  "carbs": 3,  "fat": 0, "fiber": 1},
}


class DietPlanner:
    """Build a structured diet plan from health metrics and food database."""

    def __init__(self, llm_service=None):
        self.llm      = llm_service
        self.food_db: Dict = self._load_db()

    # ─── Public ──────────────────────────────────────────────────────────────

    def create_plan(
        self,
        user: UserInput,
        metrics: HealthMetrics,
        medical: Optional[MedicalData]       = None,
        feedback: Optional[ValidationFeedback] = None,
        web_search_results: dict | None      = None,
    ) -> DietPlan:
        """
        Generate a diet plan.

        If `feedback` is provided (retry path), the plan is *corrected*
        deterministically rather than regenerated randomly.
        If `web_search_results` contains Tavily data, plan notes are enriched.
        """
        if feedback:
            log.info(
                f"[diet_planner] Retry with feedback: "
                f"cal{feedback.calorie_adjustment:+d} "
                f"prot{feedback.protein_adjustment:+.1f}g "
                f"carb{feedback.carb_adjustment:+.1f}g "
                f"fat{feedback.fat_adjustment:+.1f}g"
            )

        # Effective targets after applying feedback corrections
        target_cal  = metrics.target_calories  + (feedback.calorie_adjustment  if feedback else 0)
        target_prot = metrics.protein_g        + (feedback.protein_adjustment  if feedback else 0)
        target_carbs= metrics.carbs_g          + (feedback.carb_adjustment     if feedback else 0)
        target_fat  = metrics.fat_g            + (feedback.fat_adjustment      if feedback else 0)

        # Clamp adjusted targets to safe bounds
        from app.config import MIN_CALORIES
        target_cal = max(MIN_CALORIES, int(target_cal))

        # Meal distribution
        cal_dist, meal_names, meal_times = self._meal_template(user.goal)

        meals: List[Meal] = []
        for idx, (name, time, frac) in enumerate(zip(meal_names, meal_times, cal_dist)):
            meal_cal = int(target_cal * frac)
            meal = self._build_meal(
                meal_name=name,
                meal_time=time,
                meal_type=self._type_for(idx, user.goal),
                target_cal=meal_cal,
                user=user,
                medical=medical,
            )
            meals.append(meal)

        # ── Task 2: targeted protein correction ──────────────────────────────
        if feedback and feedback.protein_adjustment > 5:
            meals = self._boost_protein(meals, feedback.protein_adjustment)
        elif feedback and feedback.protein_adjustment < -5:
            meals = self._trim_protein(meals, abs(feedback.protein_adjustment))

        # ── Task 2: targeted calorie correction ──────────────────────────────
        if feedback and feedback.calorie_adjustment < -100:
            meals = self._trim_calories(meals, abs(feedback.calorie_adjustment))
        elif feedback and feedback.calorie_adjustment > 100:
            meals = self._add_calories(meals, feedback.calorie_adjustment)

        # Compute totals (Python, never LLM)
        total_cal  = sum(m.total_cal  for m in meals)
        total_prot = sum(m.total_prot for m in meals)
        total_carb = sum(m.total_carb for m in meals)
        total_fat  = sum(m.total_fat  for m in meals)

        notes = self._build_notes(user, medical, metrics)

        plan = DietPlan(
            daily_calories=total_cal,
            daily_protein=round(total_prot, 1),
            daily_carbs=round(total_carb, 1),
            daily_fat=round(total_fat, 1),
            meals=meals,
            hydration_liters=round(user.weight * 0.033, 1),
            notes=notes,
        )
        # Enrich notes with Tavily web research if available
        if web_search_results and web_search_results.get("available"):
            nutrition_research = web_search_results.get("nutrition_research", "")
            medical_advice     = web_search_results.get("medical_advice", "")
            if nutrition_research:
                # Extract first actionable sentence from research
                first_sentence = nutrition_research.split(".")[0].strip()
                if len(first_sentence) > 30:
                    notes.append(f"[Latest research] {first_sentence}.")
            if medical_advice:
                first_sentence = medical_advice.split(".")[0].strip()
                if len(first_sentence) > 30:
                    notes.append(f"[Clinical guideline] {first_sentence}.")

        log.info(
            f"[diet_planner] Plan built - {total_cal} kcal | "
            f"P:{total_prot:.0f}g C:{total_carb:.0f}g F:{total_fat:.0f}g"
            + (f" [retry-corrected]" if feedback else "")
            + (f" [web-enriched]" if web_search_results and web_search_results.get("available") else "")
        )
        return plan

    # ─── Meal builder ─────────────────────────────────────────────────────────

    def _build_meal(
        self,
        meal_name: str,
        meal_time: str,
        meal_type: str,
        target_cal: int,
        user: UserInput,
        medical: Optional[MedicalData],
    ) -> Meal:
        pool     = self._filtered_foods(meal_type, user, medical)
        selected = self._select_foods(pool, target_cal)
        items    = [FoodItem(**f) for f in selected]
        return self._make_meal(meal_name, meal_time, items)

    @staticmethod
    def _make_meal(name: str, time: str, items: List[FoodItem]) -> Meal:
        return Meal(
            name=name,
            time=time,
            foods=items,
            total_cal =sum(f.calories for f in items),
            total_prot=round(sum(f.protein  for f in items), 1),
            total_carb=round(sum(f.carbs    for f in items), 1),
            total_fat =round(sum(f.fat      for f in items), 1),
        )

    def _filtered_foods(
        self,
        meal_type: str,
        user: UserInput,
        medical: Optional[MedicalData],
    ) -> List[Dict]:
        bucket = self.food_db["foods"].get(meal_type, [])
        flags  = (medical.flags if medical else []) or []
        result = [f for f in bucket if not self._is_excluded(f, user, flags)]
        return result or bucket

    @staticmethod
    def _is_excluded(food: Dict, user: UserInput, flags: List[str]) -> bool:
        cat = food.get("category", "")
        meat_names = {
            "grilled chicken breast (150g)",
            "beef (lean, 150g)",
            "turkey breast (150g)",
        }
        if "vegetarian" in user.dietary_pref and cat == "protein":
            if food["name"].lower() in meat_names:
                return True
        if "vegan" in user.dietary_pref and cat in ("dairy", "protein"):
            return True
        for allergen in user.allergies:
            if allergen.lower() in food["name"].lower():
                return True
        if "high_cholesterol" in flags and food.get("fat", 0) > 15:
            return True
        if ("pre_diabetic_or_diabetic" in flags or "diabetes_risk" in flags):
            if food.get("carbs", 0) > 40 and food.get("fiber", 0) < 3:
                return True
        return False

    @staticmethod
    def _select_foods(pool: List[Dict], target_cal: int) -> List[Dict]:
        """Greedy selection toward 90 % of target calories."""
        random.shuffle(pool)
        selected: List[Dict] = []
        running  = 0
        for food in pool:
            if running >= target_cal * 0.90:
                break
            selected.append(food)
            running += food["calories"]
        return selected or (pool[:1] if pool else [])

    # ─── Task 2: targeted correction helpers ─────────────────────────────────

    def _boost_protein(self, meals: List[Meal], grams_needed: float) -> List[Meal]:
        """Inject high-protein boosters into the highest-calorie meal."""
        if not meals:
            return meals
        target_meal_idx = max(range(len(meals)), key=lambda i: meals[i].total_cal)
        added = 0.0
        boosters: List[FoodItem] = []
        for booster in _HIGH_PROTEIN_BOOST:
            if added >= grams_needed:
                break
            boosters.append(FoodItem(**booster))
            added += booster["protein"]

        meal  = meals[target_meal_idx]
        new_foods = list(meal.foods) + boosters
        meals[target_meal_idx] = self._make_meal(meal.name, meal.time, new_foods)
        log.debug(f"[diet_planner] Injected {added:.0f}g protein boost into '{meal.name}'")
        return meals

    def _trim_protein(self, meals: List[Meal], grams_excess: float) -> List[Meal]:
        """Remove the highest-protein food from the largest meal if protein is too high."""
        if not meals or grams_excess < 5:
            return meals
        target_meal_idx = max(range(len(meals)), key=lambda i: meals[i].total_prot)
        meal = meals[target_meal_idx]
        if len(meal.foods) <= 1:
            return meals
        # Drop the highest-protein item
        sorted_foods = sorted(meal.foods, key=lambda f: f.protein, reverse=True)
        new_foods = sorted_foods[1:]
        meals[target_meal_idx] = self._make_meal(meal.name, meal.time, new_foods)
        log.debug(f"[diet_planner] Removed high-protein item from '{meal.name}'")
        return meals

    def _trim_calories(self, meals: List[Meal], cal_excess: int) -> List[Meal]:
        """Drop the highest-calorie food from the highest-calorie meal."""
        if not meals or cal_excess < 50:
            return meals
        target_meal_idx = max(range(len(meals)), key=lambda i: meals[i].total_cal)
        meal = meals[target_meal_idx]
        if len(meal.foods) <= 1:
            return meals
        sorted_foods = sorted(meal.foods, key=lambda f: f.calories, reverse=True)
        new_foods = sorted_foods[1:]
        meals[target_meal_idx] = self._make_meal(meal.name, meal.time, new_foods)
        log.debug(f"[diet_planner] Trimmed high-cal item from '{meal.name}'")
        return meals

    def _add_calories(self, meals: List[Meal], cal_needed: int) -> List[Meal]:
        """Add a snack item to the smallest meal to cover a calorie shortfall."""
        if not meals or cal_needed < 50:
            return meals
        target_meal_idx = min(range(len(meals)), key=lambda i: meals[i].total_cal)
        meal   = meals[target_meal_idx]
        filler = FoodItem(
            name="Brown Rice (½ cup cooked)", calories=108,
            protein=2, carbs=22, fat=1, fiber=2,
        )
        new_foods = list(meal.foods) + [filler]
        meals[target_meal_idx] = self._make_meal(meal.name, meal.time, new_foods)
        log.debug(f"[diet_planner] Added calorie filler to '{meal.name}'")
        return meals

    # ─── Meal template ────────────────────────────────────────────────────────

    @staticmethod
    def _meal_template(goal: Goal):
        if goal == Goal.muscle_gain:
            return (
                [0.20, 0.25, 0.15, 0.25, 0.15],
                ["Breakfast", "Lunch", "Pre-Workout Snack", "Dinner", "Post-Workout"],
                ["7:00 AM",  "12:30 PM", "4:00 PM", "7:30 PM", "9:30 PM"],
            )
        return (
            [0.25, 0.35, 0.15, 0.25],
            ["Breakfast", "Lunch", "Snack", "Dinner"],
            ["7:00 AM",  "12:30 PM", "4:00 PM", "7:30 PM"],
        )

    @staticmethod
    def _type_for(idx: int, goal: Goal) -> str:
        types = (
            ["breakfast", "lunch", "snacks", "dinner", "snacks"]
            if goal == Goal.muscle_gain
            else ["breakfast", "lunch", "snacks", "dinner"]
        )
        return types[idx % len(types)]

    # ─── Notes ────────────────────────────────────────────────────────────────

    @staticmethod
    def _build_notes(
        user: UserInput,
        medical: Optional[MedicalData],
        metrics: HealthMetrics,
    ) -> List[str]:
        notes = []
        flags = (medical.flags if medical else []) or []
        if "high_cholesterol" in flags:
            notes.append("Avoid fried foods, full-fat dairy and processed meats.")
            notes.append("Include omega-3 rich foods like salmon and walnuts daily.")
        if "pre_diabetic_or_diabetic" in flags or "diabetes_risk" in flags:
            notes.append("Spread carbohydrate intake evenly across all meals.")
            notes.append("Choose low-glycaemic index foods where possible.")
        if "hypertension" in flags:
            notes.append("Keep sodium intake below 1500 mg per day.")
            notes.append("Include potassium-rich foods: bananas, sweet potato, lentils.")
        if "anemia_risk" in flags:
            notes.append("Pair iron-rich foods with vitamin-C sources at each meal.")
        if user.goal == Goal.muscle_gain:
            notes.append("Consume a protein-rich snack within 30 min after training.")
        if user.goal == Goal.fat_loss:
            notes.append("Eat fibre-rich foods to promote satiety and gut health.")
        notes.append(
            f"Drink {round(user.weight * 0.033, 1)} litres of water daily "
            f"(~{round(user.weight * 0.033 / 8, 2) * 8 * 250:.0f} ml in 8 glasses)."
        )
        return notes

    # ─── DB loader ────────────────────────────────────────────────────────────

    @staticmethod
    def _load_db() -> Dict:
        db_path = Path(FOOD_DB_PATH)
        if not db_path.exists():
            log.warning(f"Food DB not found at {db_path} - using empty DB")
            return {"foods": {}, "meal_templates": {}}
        with db_path.open() as f:
            return json.load(f)
