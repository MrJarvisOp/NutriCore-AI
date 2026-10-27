"""
tavily_service.py
-----------------
Tavily web search integration for enriching health plans with live,
evidence-based research. Used by the LangGraph web_search_node to
fetch current nutrition science, exercise research, and food data.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from app.config import TAVILY_API_KEY
from app.utils.logger import get_logger

log = get_logger(__name__)

_SEARCH_TIMEOUT = 12  # seconds


class TavilyService:
    """Fetch live health/nutrition research via Tavily Search API."""

    def __init__(self):
        self._client = None
        if not TAVILY_API_KEY:
            log.warning("[tavily] TAVILY_API_KEY not set - web search disabled")

    def _get_client(self):
        if self._client is None:
            try:
                from tavily import TavilyClient
                self._client = TavilyClient(api_key=TAVILY_API_KEY)
            except ImportError:
                raise ImportError("tavily-python not installed. Run: pip install tavily-python")
        return self._client

    def is_available(self) -> bool:
        return bool(TAVILY_API_KEY)
    
    def _goal_to_str(self, goal: Any) -> str:
        raw = goal.value if hasattr(goal, "value") else str(goal)

        if "." in raw:
            raw = raw.split(".", 1)[1]

        return raw.replace("_", " ")
    
    # ── Core search ────────────────────────────────────────────────────────

    def search(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """Run a Tavily search. Returns list of {title, url, content, score}."""
        if not self.is_available():
            log.warning("[tavily] No API key - returning empty results")
            return []
        try:
            client = self._get_client()
            resp = client.search(
                query=query,
                search_depth="advanced",
                max_results=max_results,
                include_answer=True,
                include_raw_content=False,
            )
            results = resp.get("results", [])
            log.info(f"[tavily] '{query}' → {len(results)} results")
            return results
        except Exception as exc:
            log.warning(f"[tavily] Search failed ({exc}) - returning empty")
            return []

    def get_answer(self, query: str) -> str:
        """Return Tavily's direct answer string for a query, or ''."""
        if not self.is_available():
            return ""
        try:
            client = self._get_client()
            resp = client.search(
                query=query,
                search_depth="advanced",
                max_results=3,
                include_answer=True,
            )
            return resp.get("answer", "") or ""
        except Exception as exc:
            log.warning(f"[tavily] get_answer failed: {exc}")
            return ""

    # ── Domain-specific helpers ────────────────────────────────────────────

    def search_nutrition_research(self, goal: str, flags: List[str]) -> str:
        """
        Fetch live nutrition recommendations for the user's goal and
        medical flags. Returns a plain-text summary suitable for the LLM.
        """
        flag_str = ", ".join(flags[:4]) if flags else "none"

        goal_str = self._goal_to_str(goal)

        query = (
            f"evidence-based nutrition recommendations for Goal of {goal_str} "
            f"diet 2026 {'with ' + flag_str if flags else ''}"
        )

        results = self.search(query, max_results=4)
        if not results:
            return ""
        lines = []
        for r in results[:3]:
            snippet = r.get("content", "")[:400].strip()
            if snippet:
                lines.append(f"• [{r.get('title', 'Source')}]: {snippet}")
        return "\n".join(lines)

    def search_exercise_research(self, goal: str, flags: List[str]) -> str:
        """Fetch live exercise science for the user's goal and health flags."""
        flag_str = ", ".join(flags[:3]) if flags else ""
        query = (
            f"best exercise protocol for {self._goal_to_str(goal)} 2026 "
            f"{'considering ' + flag_str if flag_str else ''} scientific evidence"
        )
        results = self.search(query, max_results=3)
        if not results:
            return ""
        lines = []
        for r in results[:3]:
            snippet = r.get("content", "")[:350].strip()
            if snippet:
                lines.append(f"• [{r.get('title', 'Source')}]: {snippet}")
        return "\n".join(lines)

    def search_food_benefits(self, foods: List[str]) -> Dict[str, str]:
        """For up to 3 foods, return a live health benefit summary."""
        benefits: Dict[str, str] = {}
        for food in foods[:3]:
            answer = self.get_answer(f"health benefits of {food} nutrition science")
            if answer:
                benefits[food] = answer[:300]
        return benefits

    def search_medical_diet_advice(self, flags: List[str]) -> str:
        """Fetch current medical dietary guidelines for detected health flags."""
        if not flags:
            return ""
        priority = flags[0].replace("_", " ")
        query = f"dietary guidelines for {priority} latest medical evidence 2026"
        answer = self.get_answer(query)
        if answer:
            return answer[:500]
        results = self.search(query, max_results=3)
        if results:
            return results[0].get("content", "")[:500]
        return ""

    def search_calorie_targets(self, goal: str, age: int, bmi: float) -> str:
        """Fetch contextual research on calorie targets for this profile."""
        query = (
            f"optimal calorie deficit for {self._goal_to_str(goal)} "
            f"age {age} BMI {bmi:.1f} evidence 2026"
        )
        return self.get_answer(query)

    def get_supplement_recommendations(self, flags: List[str], goal: str) -> List[str]:
        """Return evidence-based supplement suggestions from live search."""
        if not flags and goal == "maintenance":
            return []
        flag_str = " ".join(flags[:2]).replace("_", " ") if flags else self._goal_to_str(goal)
        query = f"evidence-based supplements for {flag_str} {self._goal_to_str(goal)} 2026"
        answer = self.get_answer(query)
        if not answer:
            return []
        # Split answer into bullet points
        lines = [l.strip("•- ").strip() for l in answer.split(".") if len(l.strip()) > 20]
        return lines[:3]
