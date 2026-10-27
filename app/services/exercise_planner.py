"""
exercise_planner.py
-------------------
Fully rule-based exercise plan generator.
No LLM involvement - goal + medical flags → deterministic workout structure.
"""
from __future__ import annotations

from typing import List, Optional

from app.models.schemas import (
    ActivityLevel, ExerciseItem, ExercisePlan, Goal,
    HealthMetrics, MedicalData, UserInput, WorkoutDay,
)
from app.utils.logger import get_logger

log = get_logger(__name__)


# ─── Exercise libraries ──────────────────────────────────────────────────────

CARDIO_EXERCISES = [
    ExerciseItem(name="Brisk Walking",       duration="30-40 min", intensity="moderate",
                 notes="Keep heart rate at 55-65 % max HR"),
    ExerciseItem(name="Cycling (stationary)", duration="30 min",   intensity="moderate"),
    ExerciseItem(name="Swimming",             duration="30 min",   intensity="moderate"),
    ExerciseItem(name="Jump Rope",            duration="15-20 min", intensity="high",
                 notes="Intervals: 1 min on / 1 min rest"),
    ExerciseItem(name="Rowing Machine",       duration="20-25 min", intensity="moderate-high"),
]

STRENGTH_UPPER = [
    ExerciseItem(name="Push-ups",           sets=3, reps="10-15",  rest="60 s"),
    ExerciseItem(name="Dumbbell Bench Press", sets=4, reps="8-12", rest="90 s"),
    ExerciseItem(name="Bent-over Row",      sets=4, reps="8-12",   rest="90 s"),
    ExerciseItem(name="Overhead Press",     sets=3, reps="8-10",   rest="90 s"),
    ExerciseItem(name="Bicep Curls",        sets=3, reps="10-12",  rest="60 s"),
    ExerciseItem(name="Tricep Dips",        sets=3, reps="10-15",  rest="60 s"),
]

STRENGTH_LOWER = [
    ExerciseItem(name="Squats",             sets=4, reps="8-12",   rest="90 s"),
    ExerciseItem(name="Romanian Deadlift",  sets=4, reps="8-10",   rest="90 s"),
    ExerciseItem(name="Lunges",             sets=3, reps="10 each leg", rest="60 s"),
    ExerciseItem(name="Leg Press",          sets=3, reps="12-15",  rest="90 s"),
    ExerciseItem(name="Calf Raises",        sets=3, reps="15-20",  rest="45 s"),
]

STRENGTH_FULL = [
    ExerciseItem(name="Burpees",            sets=3, reps="10",     rest="60 s"),
    ExerciseItem(name="Dumbbell Squat-Press", sets=3, reps="10",   rest="60 s"),
    ExerciseItem(name="Mountain Climbers",  sets=3, duration="30 s", rest="30 s"),
    ExerciseItem(name="Plank",              sets=3, duration="30-60 s", rest="30 s"),
    ExerciseItem(name="Kettlebell Swings",  sets=4, reps="15",     rest="60 s"),
]

CORE_EXERCISES = [
    ExerciseItem(name="Plank",              sets=3, duration="30-60 s", rest="30 s"),
    ExerciseItem(name="Bicycle Crunches",   sets=3, reps="20",     rest="30 s"),
    ExerciseItem(name="Russian Twists",     sets=3, reps="15 each side", rest="30 s"),
    ExerciseItem(name="Dead Bug",           sets=3, reps="10 each side", rest="30 s"),
]

LOW_IMPACT = [
    ExerciseItem(name="Yoga / Stretching",  duration="30-40 min",
                 notes="Focus on mobility and flexibility"),
    ExerciseItem(name="Resistance Bands Row", sets=3, reps="15",   rest="60 s"),
    ExerciseItem(name="Wall Push-ups",       sets=3, reps="12-15", rest="60 s"),
    ExerciseItem(name="Chair Squats",        sets=3, reps="10-12", rest="60 s"),
    ExerciseItem(name="Seated Leg Raises",   sets=3, reps="15",    rest="45 s"),
]


class ExercisePlanner:
    """Rule-based exercise plan generator."""

    def create_plan(
        self,
        user: UserInput,
        metrics: HealthMetrics,
        medical: Optional[MedicalData] = None,
        web_search_results: dict | None = None,
    ) -> ExercisePlan:
        flags    = (medical.flags if medical else []) or []
        warnings: List[str] = []
        high_impact_ok = self._high_impact_ok(flags, user, warnings)

        if user.goal == Goal.fat_loss:
            plan = self._fat_loss_plan(user, high_impact_ok)
        elif user.goal == Goal.muscle_gain:
            plan = self._muscle_gain_plan(user, high_impact_ok)
        else:
            plan = self._maintenance_plan(user, high_impact_ok)

        # Add medical-specific warnings
        if "hypertension" in flags:
            warnings.append(
                "High blood pressure: avoid maximal-effort lifts and Valsalva manoeuvre. "
                "Always warm up and cool down."
            )
        if "pre_diabetic_or_diabetic" in flags or "diabetes_risk" in flags:
            warnings.append(
                "Blood sugar management: exercise after meals; monitor glucose levels "
                "and carry fast-acting carbs."
            )
        if "high_cholesterol" in flags:
            warnings.append(
                "Elevated cholesterol: aerobic exercise is especially beneficial - "
                "aim for 150 min moderate cardio per week."
            )
        if "anemia_risk" in flags:
            warnings.append(
                "Possible anaemia: start at lower intensity and increase gradually. "
                "Watch for unusual breathlessness."
            )

        # Enrich cardio_note with live exercise research
        if web_search_results and web_search_results.get("available"):
            ex_research = web_search_results.get("exercise_research", "")
            if ex_research and plan.cardio_note:
                first = ex_research.split(".")[0].strip()
                if len(first) > 30:
                    plan.cardio_note += f" [Research: {first}.]"
        plan.warnings = warnings
        log.info(f"Exercise plan - {len(plan.workout_days)} days, goal: {user.goal}")
        return plan

    # ─── Goal-based plans ────────────────────────────────────────────────────

    def _fat_loss_plan(self, user: UserInput, high_impact_ok: bool) -> ExercisePlan:
        cardio = CARDIO_EXERCISES[:2] if high_impact_ok else [LOW_IMPACT[0]]
        days = [
            WorkoutDay(day="Monday",    focus="Full-Body Strength + Cardio",
                       exercises=STRENGTH_FULL[:4] + cardio[:1],
                       duration="50-60 min", intensity="Moderate-High"),
            WorkoutDay(day="Tuesday",   focus="Cardio (LISS)",
                       exercises=CARDIO_EXERCISES[:2],
                       duration="40-45 min", intensity="Moderate"),
            WorkoutDay(day="Wednesday", focus="Active Recovery / Core",
                       exercises=CORE_EXERCISES + [LOW_IMPACT[0]],
                       duration="30 min", intensity="Low"),
            WorkoutDay(day="Thursday",  focus="Full-Body Strength",
                       exercises=STRENGTH_UPPER[:3] + STRENGTH_LOWER[:3],
                       duration="50 min", intensity="Moderate-High"),
            WorkoutDay(day="Friday",    focus="Cardio (HIIT)",
                       exercises=[CARDIO_EXERCISES[3]],
                       duration="30 min", intensity="High"),
            WorkoutDay(day="Saturday",  focus="Upper Body Strength",
                       exercises=STRENGTH_UPPER,
                       duration="45 min", intensity="Moderate"),
            WorkoutDay(day="Sunday",    focus="Rest / Light Walk",
                       exercises=[ExerciseItem(name="Walking", duration="20-30 min",
                                               notes="Easy pace, recovery")],
                       duration="30 min", intensity="Very Low"),
        ]
        return ExercisePlan(
            goal="Fat Loss",
            weekly_days=6,
            workout_days=days,
            cardio_note="Aim for 150-200 min moderate cardio weekly. "
                        "HIIT sessions 2× / week for accelerated fat burning.",
        )

    def _muscle_gain_plan(self, user: UserInput, high_impact_ok: bool) -> ExercisePlan:
        days = [
            WorkoutDay(day="Monday",    focus="Chest & Triceps",
                       exercises=STRENGTH_UPPER[:3] + [STRENGTH_UPPER[5]],
                       duration="55-65 min", intensity="High"),
            WorkoutDay(day="Tuesday",   focus="Back & Biceps",
                       exercises=[STRENGTH_UPPER[2], STRENGTH_UPPER[3], STRENGTH_UPPER[4]],
                       duration="55 min", intensity="High"),
            WorkoutDay(day="Wednesday", focus="Rest / Light Cardio",
                       exercises=[CARDIO_EXERCISES[0]],
                       duration="30 min", intensity="Low"),
            WorkoutDay(day="Thursday",  focus="Legs",
                       exercises=STRENGTH_LOWER,
                       duration="60-70 min", intensity="High"),
            WorkoutDay(day="Friday",    focus="Shoulders & Core",
                       exercises=[STRENGTH_UPPER[3]] + CORE_EXERCISES,
                       duration="50 min", intensity="Moderate-High"),
            WorkoutDay(day="Saturday",  focus="Full Body Compound",
                       exercises=STRENGTH_FULL[:4],
                       duration="45-55 min", intensity="Moderate-High"),
            WorkoutDay(day="Sunday",    focus="Rest",
                       exercises=[ExerciseItem(name="Rest Day", notes="Focus on sleep and nutrition")],
                       duration="—", intensity="None"),
        ]
        return ExercisePlan(
            goal="Muscle Gain",
            weekly_days=5,
            workout_days=days,
            cardio_note="Keep cardio to 2× 20-30 min light sessions to preserve muscle mass.",
        )

    def _maintenance_plan(self, user: UserInput, high_impact_ok: bool) -> ExercisePlan:
        exercises = STRENGTH_FULL[:3] + CARDIO_EXERCISES[:2]
        if not high_impact_ok:
            exercises = LOW_IMPACT
        days = [
            WorkoutDay(day="Monday",    focus="Full Body Strength",
                       exercises=STRENGTH_UPPER[:3] + STRENGTH_LOWER[:2],
                       duration="45-50 min", intensity="Moderate"),
            WorkoutDay(day="Wednesday", focus="Cardio + Core",
                       exercises=CARDIO_EXERCISES[:2] + CORE_EXERCISES[:2],
                       duration="45 min", intensity="Moderate"),
            WorkoutDay(day="Friday",    focus="Full Body Strength",
                       exercises=STRENGTH_LOWER[:3] + STRENGTH_UPPER[3:],
                       duration="45 min", intensity="Moderate"),
            WorkoutDay(day="Sunday",    focus="Flexibility / Yoga",
                       exercises=[LOW_IMPACT[0]],
                       duration="30 min", intensity="Low"),
        ]
        return ExercisePlan(
            goal="Maintenance",
            weekly_days=4,
            workout_days=days,
            cardio_note="150 min moderate aerobic activity per week as per WHO guidelines.",
        )

    # ─── Safety check ─────────────────────────────────────────────────────────

    @staticmethod
    def _high_impact_ok(flags: List[str], user: UserInput, warnings: List[str]) -> bool:
        if "hypertension" in flags:
            warnings.append("High-impact exercises replaced with low-impact alternatives.")
            return False
        if user.age >= 65:
            warnings.append("Age 65+: low-impact exercises recommended.")
            return False
        return True
