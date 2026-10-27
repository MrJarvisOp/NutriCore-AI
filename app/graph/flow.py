"""
flow.py
-------
LangGraph orchestration pipeline with Tavily web search enrichment.

START → input_node → pdf_node → parser_node → health_node
      → web_search_node (Tavily)
      → diet_node → exercise_node → validation_node
         ↙ (fail, retry < MAX)     ↘ (pass OR max retries)
     retry_node                      llm_node → END
         ↓
     diet_node  (correction-driven)
"""
from __future__ import annotations
from typing import Literal

from langgraph.graph import END, START, StateGraph

from app.models.schemas import (
    GraphState, MedicalData, ValidationFeedback, ValidationResult,
)
from app.services.diet_planner      import DietPlanner
from app.services.exercise_planner  import ExercisePlanner
from app.services.health_engine     import HealthEngine
from app.services.llm_service       import LLMService
from app.services.parser            import MedicalReportParser
from app.services.tavily_service    import TavilyService
from app.services.validation        import PlanValidator
from app.utils.logger               import get_logger

log = get_logger(__name__)

MAX_RETRIES = 3

# ── Singleton services ────────────────────────────────────────────────────────
_llm       = LLMService()
_tavily    = TavilyService()
_parser    = MedicalReportParser(llm_service=_llm)
_diet      = DietPlanner(llm_service=_llm)
_exercise  = ExercisePlanner()
_validator = PlanValidator()


# Node functions
def input_node(state: GraphState) -> dict:
    log.info(
        f"[input_node] User: age={state.user_input.age}, "
        f"weight={state.user_input.weight}kg, goal={state.user_input.goal}"
    )
    return {}


def pdf_node(state: GraphState) -> dict:
    pdf_path = state.pdf_path
    if not pdf_path:
        log.info("[pdf_node] No PDF - skipping")
        return {}
    try:
        from app.services.ocr_service import OCRService
        text = OCRService(pdf_path).extract_text()
        log.info(f"[pdf_node] Extracted {len(text)} chars")
        return {"pdf_text": text}
    except Exception as exc:
        log.warning(f"[pdf_node] Failed: {exc}")
        return {"error": f"PDF extraction failed: {exc}"}


def parser_node(state: GraphState) -> dict:
    if not state.pdf_text:
        log.info("[parser_node] No PDF text - empty MedicalData")
        return {"medical_data": MedicalData()}
    try:
        medical = _parser.parse(state.pdf_text)
        log.info(f"[parser_node] Flags: {medical.flags}")
        return {"medical_data": medical}
    except Exception as exc:
        log.error(f"[parser_node] Error: {exc}")
        return {"medical_data": MedicalData(), "error": str(exc)}


def health_node(state: GraphState) -> dict:
    try:
        metrics = HealthEngine(
            user=state.user_input, medical=state.medical_data
        ).calculate()
        log.info(f"[health_node] BMI={metrics.bmi} | {metrics.target_calories} kcal")
        return {"health_metrics": metrics}
    except Exception as exc:
        log.error(f"[health_node] Error: {exc}")
        return {"error": str(exc)}


def web_search_node(state: GraphState) -> dict:
    """
    Tavily web search enrichment node.
    Fetches live, evidence-based nutrition & exercise research
    that gets woven into the LLM recommendations and plan notes.
    Gracefully skips if Tavily is not configured.
    """
    if not _tavily.is_available():
        log.info("[web_search_node] Tavily not configured - skipping")
        return {"web_search_results": {"available": False}}

    metrics = state.health_metrics
    flags   = (state.medical_data.flags if state.medical_data else []) or []
    goal    = str(state.user_input.goal)

    results: dict = {"available": True}

    try:
        log.info("[web_search_node] Fetching nutrition research …")
        results["nutrition_research"] = _tavily.search_nutrition_research(goal, flags)
    except Exception as exc:
        log.warning(f"[web_search_node] nutrition search failed: {exc}")
        results["nutrition_research"] = ""

    try:
        log.info("[web_search_node] Fetching exercise research …")
        results["exercise_research"] = _tavily.search_exercise_research(goal, flags)
    except Exception as exc:
        log.warning(f"[web_search_node] exercise search failed: {exc}")
        results["exercise_research"] = ""

    try:
        if flags:
            log.info("[web_search_node] Fetching medical diet advice …")
            results["medical_advice"] = _tavily.search_medical_diet_advice(flags)
        else:
            results["medical_advice"] = ""
    except Exception as exc:
        log.warning(f"[web_search_node] medical search failed: {exc}")
        results["medical_advice"] = ""

    try:
        if metrics:
            log.info("[web_search_node] Fetching calorie target research …")
            results["calorie_research"] = _tavily.search_calorie_targets(
                goal, state.user_input.age, metrics.bmi
            )
        else:
            results["calorie_research"] = ""
    except Exception as exc:
        log.warning(f"[web_search_node] calorie search failed: {exc}")
        results["calorie_research"] = ""

    try:
        supplements = _tavily.get_supplement_recommendations(flags, goal)
        results["supplement_suggestions"] = supplements
    except Exception as exc:
        log.warning(f"[web_search_node] supplement search failed: {exc}")
        results["supplement_suggestions"] = []

    log.info(
        f"[web_search_node] Done. "
        f"nutrition={bool(results.get('nutrition_research'))}, "
        f"exercise={bool(results.get('exercise_research'))}, "
        f"medical={bool(results.get('medical_advice'))}"
    )
    return {"web_search_results": results}


def diet_node(state: GraphState) -> dict:
    try:
        plan = _diet.create_plan(
            user=state.user_input,
            metrics=state.health_metrics,
            medical=state.medical_data,
            feedback=state.validation_feedback,
            web_search_results=state.web_search_results,
        )
        log.info(
            f"[diet_node] {plan.daily_calories} kcal | retry={state.retry_count}"
        )
        return {"diet_plan": plan}
    except Exception as exc:
        log.error(f"[diet_node] Error: {exc}")
        return {"error": str(exc)}


def exercise_node(state: GraphState) -> dict:
    try:
        plan = _exercise.create_plan(
            user=state.user_input,
            metrics=state.health_metrics,
            medical=state.medical_data,
            web_search_results=state.web_search_results,
        )
        log.info(f"[exercise_node] {len(plan.workout_days)} days")
        return {"exercise_plan": plan}
    except Exception as exc:
        log.error(f"[exercise_node] Error: {exc}")
        return {"error": str(exc)}


def validation_node(state: GraphState) -> dict:
    if state.diet_plan is None or state.exercise_plan is None:
        return {"validation_result": ValidationResult(
            passed=False, issues=["Diet or exercise plan missing."]
        )}
    result = _validator.validate(
        diet=state.diet_plan, exercise=state.exercise_plan,
        metrics=state.health_metrics, medical=state.medical_data,
    )
    if not result.passed and state.diet_plan and state.health_metrics:
        result.feedback = ValidationFeedback.from_validation_result(
            result, state.health_metrics, state.diet_plan
        )
    return {"validation_result": result}


def retry_node(state: GraphState) -> dict:
    new_count = state.retry_count + 1
    feedback  = (state.validation_result.feedback if state.validation_result else None)
    log.info(
        f"[retry_node] retry_count {state.retry_count} → {new_count}"
        + (f" | cal{feedback.calorie_adjustment:+d} prot{feedback.protein_adjustment:+.1f}g" if feedback else "")
    )
    return {"retry_count": new_count, "validation_feedback": feedback}


def llm_node(state: GraphState) -> dict:
    """
    Final LLM node: generates personalised recommendations enriched
    by Tavily web search context.
    """
    try:
        flags   = (state.medical_data.flags if state.medical_data else []) or []
        ws      = state.web_search_results or {}
        goal    = str(state.user_input.goal)

        # Build enriched context from Tavily results
        web_context_parts = []
        if ws.get("nutrition_research"):
            web_context_parts.append(
                f"Current nutrition research:\n{ws['nutrition_research']}"
            )
        if ws.get("exercise_research"):
            web_context_parts.append(
                f"Current exercise research:\n{ws['exercise_research']}"
            )
        if ws.get("medical_advice"):
            web_context_parts.append(
                f"Medical dietary guidelines:\n{ws['medical_advice']}"
            )
        if ws.get("calorie_research"):
            web_context_parts.append(
                f"Calorie target research:\n{ws['calorie_research']}"
            )

        web_context = "\n\n".join(web_context_parts) if web_context_parts else None

        recs = _llm.generate_recommendations(
            user_summary={
                "age":    state.user_input.age,
                "weight": state.user_input.weight,
                "height": state.user_input.height,
            },
            health_metrics=state.health_metrics.dict(),
            medical_flags=flags,
            goal=goal,
            web_context=web_context,
        )

        # Append live supplement suggestions from Tavily
        supplements = ws.get("supplement_suggestions", [])
        for s in supplements[:2]:
            if s and s not in recs:
                recs.append(f"[Evidence-based] {s}")

        log.info(f"[llm_node] {len(recs)} recommendations generated")
        return {"llm_context": "\n".join(recs)}

    except Exception as exc:
        log.warning(f"[llm_node] Failed (non-fatal): {exc}")
        return {}


# ── Router ────────────────────────────────────────────────────────────────────

def route_after_validation(state: GraphState) -> Literal["retry_node", "llm_node"]:
    result = state.validation_result
    if result and result.passed:
        log.info("[router] Passed ✓ → llm_node")
        return "llm_node"
    if state.retry_count >= MAX_RETRIES:
        log.warning(f"[router] Max retries reached → llm_node")
        return "llm_node"
    log.info(f"[router] Failed (retry {state.retry_count + 1}/{MAX_RETRIES}) → retry_node")
    return "retry_node"


# ── Graph assembly ────────────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    graph = StateGraph(GraphState)

    graph.add_node("input_node",      input_node)
    graph.add_node("pdf_node",        pdf_node)
    graph.add_node("parser_node",     parser_node)
    graph.add_node("health_node",     health_node)
    graph.add_node("web_search_node", web_search_node)   # NEW: Tavily
    graph.add_node("diet_node",       diet_node)
    graph.add_node("exercise_node",   exercise_node)
    graph.add_node("validation_node", validation_node)
    graph.add_node("retry_node",      retry_node)
    graph.add_node("llm_node",        llm_node)

    graph.add_edge(START,              "input_node")
    graph.add_edge("input_node",       "pdf_node")
    graph.add_edge("pdf_node",         "parser_node")
    graph.add_edge("parser_node",      "health_node")
    graph.add_edge("health_node",      "web_search_node")   # web search after metrics
    graph.add_edge("web_search_node",  "diet_node")
    graph.add_edge("diet_node",        "exercise_node")
    graph.add_edge("exercise_node",    "validation_node")

    graph.add_conditional_edges(
        "validation_node",
        route_after_validation,
        {"retry_node": "retry_node", "llm_node": "llm_node"},
    )
    graph.add_edge("retry_node", "diet_node")
    graph.add_edge("llm_node",   END)

    return graph.compile()


compiled_graph = build_graph()
