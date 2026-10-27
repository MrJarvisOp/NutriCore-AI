"""
main.py
-------
FastAPI app with:
- Jinja2 templates for all HTML pages
- Real SSE endpoint that streams pipeline progress while it runs
- POST /plan → shows processing.html which polls /plan/stream → redirects to result
- POST /api/v1/health-plan → JSON API (unchanged)
- Full Tavily + LLM integration
"""
from __future__ import annotations

import asyncio
import json
import os
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.api.routes import router as api_router
from app.config import APP_NAME, APP_VERSION, DEBUG, UPLOAD_DIR
from app.graph.flow import compiled_graph
from app.models.schemas import (
    ActivityLevel, Gender, Goal, GraphState,
    HealthPlanResponse, UserInput,
)
from app.services.validation import ResponseSanityValidator
from app.utils.logger import get_logger

log     = get_logger(__name__)
_sanity = ResponseSanityValidator()

BASE_DIR   = Path(__file__).parent.parent
TMPL_DIR   = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"
STATIC_DIR.mkdir(exist_ok=True)

templates = Jinja2Templates(directory=str(TMPL_DIR))

# ── In-memory job store (keyed by job_id) ────────────────────────────────────
# Each entry:  {"status": "running"|"done"|"error",
#               "logs": [...],  "plan_html": "...",  "error": "..."}
_jobs: dict[str, dict] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info(f"Starting {APP_NAME} v{APP_VERSION}")
    yield
    log.info(f"Shutting down {APP_NAME}")


app = FastAPI(
    title=APP_NAME, version=APP_VERSION,
    description="AI-powered personalised health plan generator",
    lifespan=lifespan, debug=DEBUG,
)

app.add_middleware(
    CORSMiddleware, allow_origins=["*"],
    allow_methods=["*"], allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
app.include_router(api_router)


# Helpers

def _save_upload(upload: UploadFile, data: bytes) -> str:
    ext      = os.path.splitext(upload.filename or "")[-1] or ".pdf"
    filename = f"{uuid.uuid4().hex}{ext}"
    path     = os.path.join(UPLOAD_DIR, filename)
    with open(path, "wb") as f:
        f.write(data)
    log.info(f"[web] Saved PDF → {path}")
    return path


def _cleanup(pdf_path: str | None) -> None:
    if pdf_path and os.path.exists(pdf_path):
        try:
            os.remove(pdf_path)
        except OSError:
            pass


def _build_plan_object(final_state: GraphState, user: UserInput,
                        recommendations: list, warnings: list) -> HealthPlanResponse:
    ws = final_state.web_search_results or {}
    web_ctx = None
    if ws.get("available"):
        web_ctx = {k: v for k, v in ws.items()
                   if k != "available" and v}

    return HealthPlanResponse(
        user_summary=dict(
            age=user.age, weight_kg=user.weight, height_cm=user.height,
            goal=user.goal, activity_level=user.activity_level,
        ),
        health_metrics=final_state.health_metrics,
        medical_data=final_state.medical_data,
        diet_plan=final_state.diet_plan,
        exercise_plan=final_state.exercise_plan,
        recommendations=recommendations,
        warnings=warnings,
        validation_result=final_state.validation_result,
        web_search_context=web_ctx,
    )


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


# HTML routes

@app.get("/", response_class=HTMLResponse)
@app.get("/plan", response_class=HTMLResponse)
async def plan_form(request: Request):
    return templates.TemplateResponse(
        request=request, name="form.html",
        context={"form": {}, "errors": []},
    )


@app.post("/plan", response_class=HTMLResponse)
async def start_plan(
    request:        Request,
    age:            int           = Form(...),
    weight:         float         = Form(...),
    height:         float         = Form(...),
    goal:           Goal          = Form(...),
    gender:         Gender        = Form(Gender.male),
    activity_level: ActivityLevel = Form(ActivityLevel.moderate),
    dietary_pref:   str           = Form(""),
    allergies:      str           = Form(""),
    medical_pdf:    UploadFile    = File(None),
):
    """
    Validate inputs, save PDF, register a job_id,
    then render the processing page. The processing page
    polls /plan/stream/<job_id> via SSE and redirects when done.
    """
    form_vals = dict(age=age, weight=weight, height=height, goal=goal,
                     gender=gender, activity_level=activity_level,
                     dietary_pref=dietary_pref, allergies=allergies)

    try:
        user = UserInput(
            age=age, weight=weight, height=height, goal=goal,
            gender=gender, activity_level=activity_level,
            dietary_pref=[p.strip() for p in dietary_pref.split(",") if p.strip()],
            allergies=[a.strip() for a in allergies.split(",") if a.strip()],
        )
    except Exception as exc:
        return templates.TemplateResponse(
            request=request, name="form.html",
            context={"errors": [str(exc)], "form": form_vals},
            status_code=422,
        )

    # Read PDF bytes synchronously before the background task
    pdf_bytes: bytes | None = None
    pdf_filename: str | None = None
    if medical_pdf and medical_pdf.filename:
        pdf_bytes    = await medical_pdf.read()
        pdf_filename = medical_pdf.filename

    job_id = uuid.uuid4().hex
    _jobs[job_id] = {"status": "running", "logs": [], "plan_html": None, "error": None}

    # Fire-and-forget pipeline in background
    asyncio.create_task(_run_pipeline(job_id, user, pdf_bytes, pdf_filename))

    has_pdf = bool(pdf_bytes)
    return templates.TemplateResponse(
        request=request, name="processing.html",
        context={
            "job_id":  job_id,
            "goal":    str(goal),
            "has_pdf": has_pdf,
            "age":     age,
            "weight":  weight,
            "height":  height,
        },
    )


@app.get("/plan/result/{job_id}", response_class=HTMLResponse)
async def plan_result(request: Request, job_id: str):
    """Serve the pre-rendered plan HTML stored by the background task."""
    job = _jobs.get(job_id)
    if not job:
        return templates.TemplateResponse(
            request=request, name="error.html",
            context={"code": 404, "title": "Plan Not Found",
                     "detail": f"Job {job_id} not found or expired."},
            status_code=404,
        )
    if job["status"] == "error":
        return templates.TemplateResponse(
            request=request, name="error.html",
            context={"code": 500, "title": "Pipeline Error",
                     "detail": job["error"]},
            status_code=500,
        )
    if job["status"] != "done" or not job["plan_html"]:
        return templates.TemplateResponse(
            request=request, name="processing.html",
            context={"job_id": job_id, "goal": "fat_loss",
                     "has_pdf": False, "age": "-", "weight": "-", "height": "-"},
        )
    return HTMLResponse(content=job["plan_html"])


# SSE streaming endpoint - real pipeline progress

@app.get("/plan/stream/{job_id}")
async def plan_stream(request: Request, job_id: str):
    """
    SSE endpoint polled by processing.html.
    Streams log lines as they arrive, then sends a 'done' or 'error' event.
    """
    async def event_gen() -> AsyncGenerator[str, None]:
        job = _jobs.get(job_id)
        if not job:
            yield _sse("error", {"message": "Job not found"})
            return

        sent = 0
        while True:
            if await request.is_disconnected():
                break

            job = _jobs.get(job_id, {})
            logs = job.get("logs", [])

            # Send any new log lines
            while sent < len(logs):
                yield _sse("log", logs[sent])
                sent += 1
                await asyncio.sleep(0)

            if job.get("status") == "done":
                yield _sse("done", {"redirect": f"/plan/result/{job_id}"})
                break
            elif job.get("status") == "error":
                yield _sse("error", {"message": job.get("error", "Unknown error")})
                break

            await asyncio.sleep(0.3)

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


# Background pipeline task

async def _run_pipeline(
    job_id: str,
    user: UserInput,
    pdf_bytes: bytes | None,
    pdf_filename: str | None,
) -> None:
    """
    Runs the LangGraph pipeline in a thread (non-blocking).
    Emits structured log events into _jobs[job_id]["logs"].
    On completion stores pre-rendered HTML into _jobs[job_id]["plan_html"].
    """
    def emit(node: str, level: str, message: str):
        _jobs[job_id]["logs"].append({
            "node": node, "level": level, "message": message
        })

    pdf_path: str | None = None

    try:
        emit("init", "info", f"Profile loaded — {user.age}y, {user.weight}kg, goal: {user.goal}")

        if pdf_bytes and pdf_filename:
            pdf_path = _save_upload(
                type("F", (), {"filename": pdf_filename})(),  # lightweight stub
                pdf_bytes,
            )
            emit("pdf", "info", f"PDF saved ({len(pdf_bytes):,} bytes) — running OCR")
        else:
            emit("pdf", "info", "No medical PDF provided — skipping OCR")

        emit("health", "info", "Computing BMI / BMR / TDEE …")

        initial = {"user_input": user, "retry_count": 0}
        if pdf_path:
            initial["pdf_path"] = pdf_path

        emit("pipeline", "info", "LangGraph pipeline starting …")

        # Run in executor so we don't block the event loop
        loop = asyncio.get_event_loop()
        raw_state = await loop.run_in_executor(None, compiled_graph.invoke, initial)

        if not isinstance(raw_state, dict):
            raise ValueError(f"Unexpected graph output type: {type(raw_state)}")

        final_state = GraphState(**raw_state)

        # Emit node-level summaries from the completed state
        if final_state.health_metrics:
            m = final_state.health_metrics
            emit("health", "ok",
                 f"BMI {m.bmi} ({m.bmi_category}) | Target {m.target_calories} kcal | "
                 f"P:{m.protein_g:.0f}g C:{m.carbs_g:.0f}g F:{m.fat_g:.0f}g")

        ws = final_state.web_search_results or {}
        if ws.get("available"):
            found = sum(1 for k, v in ws.items() if k != "available" and v)
            emit("search", "ok", f"Tavily web search complete — {found} research areas enriched")
        else:
            emit("search", "info", "Web search skipped (TAVILY_API_KEY not set)")

        if final_state.medical_data and final_state.medical_data.flags:
            flags = final_state.medical_data.flags
            emit("parser", "warn",
                 f"Medical flags detected: {', '.join(flags[:5])}")
        else:
            emit("parser", "info", "No medical flags detected")

        if final_state.diet_plan:
            d = final_state.diet_plan
            emit("diet", "ok",
                 f"Diet plan: {len(d.meals)} meals, {d.daily_calories} kcal/day")

        if final_state.exercise_plan:
            e = final_state.exercise_plan
            emit("exercise", "ok",
                 f"Exercise plan: {e.weekly_days}×/week, goal: {e.goal}")

        if final_state.validation_result:
            v = final_state.validation_result
            if v.passed:
                emit("validation", "ok", "Plan validation PASSED ✓")
            else:
                retries = final_state.retry_count
                emit("validation", "warn",
                     f"Validation issues resolved after {retries} retry cycle(s)")

        if final_state.error and not final_state.diet_plan:
            raise ValueError(final_state.error)

        # Build recommendations
        recs = []
        if final_state.llm_context:
            recs = [l.strip() for l in final_state.llm_context.splitlines() if l.strip()]
        if not recs:
            recs = [
                "Stay hydrated — drink water before each meal.",
                "Prioritise 7-9 hours of quality sleep.",
                "Track your meals for the first 2 weeks to build awareness.",
                "Consult your doctor before starting any new supplement.",
            ]

        emit("llm", "ok", f"Generated {len(recs)} personalised recommendations")

        all_warnings = list(final_state.health_metrics.warnings or [])
        sanity = _sanity.validate(
            diet=final_state.diet_plan, exercise=final_state.exercise_plan,
            metrics=final_state.health_metrics, medical=final_state.medical_data,
            recommendations=recs,
        )

        if not sanity["safe"]:
            issues_str = "; ".join(sanity["issues"][:3])
            raise ValueError(f"Safety validation failed: {issues_str}")

        all_warnings.extend(sanity["warnings_to_append"])
        emit("safety", "ok", "Response safety validation PASSED ✓")

        plan = _build_plan_object(
            final_state, user,
            sanity["sanitised_recommendations"],
            all_warnings,
        )

        # Pre-render plan.html and store it
        tmpl = templates.get_template("plan.html")
        plan_html = tmpl.render(plan=plan)

        _jobs[job_id]["plan_html"] = plan_html
        _jobs[job_id]["status"]    = "done"
        emit("done", "ok", "Pipeline complete — loading your plan …")

    except Exception as exc:
        log.error(f"[pipeline:{job_id}] Error: {exc}", exc_info=True)
        _jobs[job_id]["status"] = "error"
        _jobs[job_id]["error"]  = str(exc)
        emit("error", "error", f"Pipeline failed: {exc}")

    finally:
        _cleanup(pdf_path)


# Error handlers

@app.exception_handler(404)
async def not_found(request: Request, exc):
    if "application/json" in request.headers.get("accept", ""):
        return JSONResponse({"detail": "Not found"}, status_code=404)
    return templates.TemplateResponse(
        request=request, name="error.html",
        context={"code": 404, "title": "Page Not Found",
                 "detail": f"No route: {request.url.path}"},
        status_code=404,
    )


@app.exception_handler(Exception)
async def server_error(request: Request, exc):
    log.error(f"Unhandled: {exc}", exc_info=True)
    if "application/json" in request.headers.get("accept", ""):
        return JSONResponse({"detail": str(exc)}, status_code=500)
    return templates.TemplateResponse(
        request=request, name="error.html",
        context={"code": 500, "title": "Internal Server Error", "detail": str(exc)},
        status_code=500,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
