"""
routes.py
---------
FastAPI routes for the Diet Health AI service.
"""
from __future__ import annotations

import os
import uuid
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from app.config import UPLOAD_DIR
from app.graph.flow import compiled_graph
from app.models.schemas import (
    ActivityLevel, Gender, Goal,
    GraphState, HealthPlanResponse, UserInput,
)
from app.services.validation import ResponseSanityValidator
from app.utils.logger import get_logger

log    = get_logger(__name__)
router = APIRouter(prefix="/api/v1", tags=["Health Plan"])

_sanity = ResponseSanityValidator()


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _save_upload(upload: UploadFile) -> str:
    """Persist the uploaded file and return its path."""
    ext      = os.path.splitext(upload.filename or "")[-1] or ".pdf"
    filename = f"{uuid.uuid4().hex}{ext}"
    path     = os.path.join(UPLOAD_DIR, filename)
    with open(path, "wb") as f:
        f.write(upload.file.read())
    log.info(f"[routes] Saved upload → {path}")
    return path


def _cleanup_pdf(pdf_path: Optional[str]) -> None:
    """Task 6: Delete the uploaded PDF after processing."""
    if not pdf_path:
        return
    try:
        if os.path.exists(pdf_path):
            os.remove(pdf_path)
            log.info(f"[routes] Deleted processed PDF: {pdf_path}")
    except OSError as exc:
        log.warning(f"[routes] Could not delete PDF {pdf_path}: {exc}")


def _build_response(final_state: GraphState) -> HealthPlanResponse:
    """Assemble and sanity-check the final response object from graph state."""
    recommendations: list = []
    if final_state.llm_context:
        recommendations = [
            line.strip()
            for line in final_state.llm_context.splitlines()
            if line.strip()
        ]

    if not recommendations:
        recommendations = [
            "Stay hydrated — drink water before each meal.",
            "Prioritise 7-9 hours of quality sleep.",
            "Track your meals for the first 2 weeks to build awareness.",
            "Consult your doctor before starting any new supplement.",
        ]

    all_warnings = list(final_state.health_metrics.warnings or [])
    if (
        final_state.validation_result
        and not final_state.validation_result.passed
    ):
        all_warnings.append(
            "Plan validation flagged minor issues - "
            "consider consulting a registered dietitian."
        )

    # ── Task 4: Layer 2 response sanity validation ────────────────────────
    sanity = _sanity.validate(
        diet=final_state.diet_plan,
        exercise=final_state.exercise_plan,
        metrics=final_state.health_metrics,
        medical=final_state.medical_data,
        recommendations=recommendations,
    )

    if not sanity["safe"]:
        log.error(
            f"[routes] ResponseSanityValidator BLOCKING issues: "
            f"{sanity['issues']}"
        )
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Generated plan failed final safety validation.",
                "issues": sanity["issues"],
            },
        )

    # Append any extra safety warnings discovered in layer 2
    all_warnings.extend(sanity["warnings_to_append"])
    recommendations = sanity["sanitised_recommendations"]

    return HealthPlanResponse(
        user_summary={
            "age":            final_state.user_input.age,
            "weight_kg":      final_state.user_input.weight,
            "height_cm":      final_state.user_input.height,
            "goal":           final_state.user_input.goal,
            "activity_level": final_state.user_input.activity_level,
        },
        health_metrics=final_state.health_metrics,
        medical_data=final_state.medical_data,
        diet_plan=final_state.diet_plan,
        exercise_plan=final_state.exercise_plan,
        recommendations=recommendations,
        warnings=all_warnings,
    )


# ─── Endpoints ───────────────────────────────────────────────────────────────

@router.post("/health-plan", response_model=HealthPlanResponse)
async def generate_health_plan(
    age:            int           = Form(...),
    weight:         float         = Form(..., description="kg"),
    height:         float         = Form(..., description="cm"),
    goal:           Goal          = Form(...),
    gender:         Gender        = Form(Gender.male),
    activity_level: ActivityLevel = Form(ActivityLevel.moderate),
    dietary_pref:   str           = Form("", description="Comma-separated: vegetarian,vegan"),
    allergies:      str           = Form("", description="Comma-separated allergens"),
    medical_pdf:    Optional[UploadFile] = File(None),
):
    """Generate a personalised diet + exercise plan."""
    log.info(
        f"[routes] Request received - age={age} weight={weight} "
        f"height={height} goal={goal}"
    )

    try:
        user = UserInput(
            age=age,
            weight=weight,
            height=height,
            goal=goal,
            gender=gender,
            activity_level=activity_level,
            dietary_pref=[p.strip() for p in dietary_pref.split(",") if p.strip()],
            allergies=[a.strip() for a in allergies.split(",") if a.strip()],
        )
    except Exception as exc:
        log.warning(f"[routes] Input validation error: {exc}")
        raise HTTPException(status_code=422, detail=str(exc))

    # Task 6: track pdf_path so we can clean it up in finally block
    pdf_path: Optional[str] = None
    try:
        if medical_pdf and medical_pdf.filename:
            pdf_path = _save_upload(medical_pdf)

        initial: dict = {"user_input": user, "retry_count": 0}
        if pdf_path:
            initial["pdf_path"] = pdf_path

        log.info("[routes] Starting LangGraph pipeline …")
        raw_state = compiled_graph.invoke(initial)
        
        if not isinstance(raw_state, dict):
            raise ValueError(f"Unexpected graph output type: {type(raw_state)}")
        
        final_state = GraphState(**raw_state)

        if final_state.error and not final_state.diet_plan:
            log.error(f"[routes] Pipeline returned error: {final_state.error}")
            raise HTTPException(status_code=500, detail=final_state.error)

        return _build_response(final_state)

    except HTTPException:
        raise
    except Exception as exc:
        log.error(f"[routes] Unhandled pipeline error: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Pipeline error: {exc}")
    finally:
        # Task 6: always clean up the uploaded PDF, even on error
        _cleanup_pdf(pdf_path)


@router.get("/health", summary="Health check")
async def health_check():
    return {"status": "ok", "service": "Diet Health AI"}


@router.get("/goals", summary="List available goals")
async def list_goals():
    return {"goals": [g.value for g in Goal]}
