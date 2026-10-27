"""
llm_service.py
--------------
Thin wrapper around Groq LLaMA 3.
Used only for: recommendations (with Tavily web context) and fallback parsing.
Task 7: timeout + safe fallback on any failure.
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from app.config import GROQ_API_KEY, LLM_MAX_TOKENS, LLM_MODEL, LLM_TEMPERATURE
from app.utils.logger import get_logger

log = get_logger(__name__)

LLM_TIMEOUT_SECONDS = 15

_DEFAULT_RECOMMENDATIONS = [
    "Stay hydrated - aim for 2-3 litres of water daily.",
    "Prioritise 7-9 hours of quality sleep each night.",
    "Track your meals for the first two weeks to build nutritional awareness.",
    "Include a 10-minute walk after main meals to aid blood sugar control.",
    "Eat slowly and stop at 80% fullness to avoid overeating.",
    "Prepare meals in advance to reduce reliance on processed foods.",
    "Consult a registered dietitian before adding any supplements.",
]


class LLMService:
    """Groq-hosted LLaMA 3 with timeout and safe fallback."""

    def __init__(self):
        if not GROQ_API_KEY:
            log.warning("[llm_service] GROQ_API_KEY not set - LLM features use fallbacks.")
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                from groq import Groq
                self._client = Groq(api_key=GROQ_API_KEY)
            except ImportError:
                log.error("[llm_service] groq package not installed. Run: pip install groq")
                raise
        return self._client

    def generate(
        self,
        prompt: str,
        system: str = "You are a professional nutritionist and health coach. Be concise, evidence-based, and safety-first.",
        max_tokens: int = LLM_MAX_TOKENS,
        temperature: float = LLM_TEMPERATURE,
    ) -> str:
        if not GROQ_API_KEY:
            log.warning("[llm_service] No API key - returning empty string")
            return ""

        import signal

        def _timeout_handler(signum, frame):
            raise TimeoutError(f"LLM call exceeded {LLM_TIMEOUT_SECONDS}s timeout")

        try:
            signal.signal(signal.SIGALRM, _timeout_handler)
            signal.alarm(LLM_TIMEOUT_SECONDS)
        except AttributeError:
            pass  # Windows

        try:
            client   = self._get_client()
            response = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user",   "content": prompt},
                ],
                max_tokens=max_tokens,
                temperature=temperature,
                timeout=LLM_TIMEOUT_SECONDS,
            )
            text = response.choices[0].message.content.strip()
            log.debug(f"[llm_service] Response received ({len(text)} chars)")
            return text
        except TimeoutError as exc:
            log.warning(f"[llm_service] Timeout: {exc}")
            raise
        except Exception as exc:
            log.error(f"[llm_service] Generation failed: {exc}")
            raise
        finally:
            try:
                signal.alarm(0)
            except AttributeError:
                pass

    def generate_recommendations(
        self,
        user_summary:   Dict[str, Any],
        health_metrics: Dict[str, Any],
        medical_flags:  List[str],
        goal:           str,
        web_context:    Optional[str] = None,
    ) -> List[str]:
        """Return 7 personalised recommendations enriched by Tavily context."""
        flag_str    = ", ".join(medical_flags) if medical_flags else "none"
        bmi_val     = health_metrics.get("bmi", "")
        bmi_cat     = health_metrics.get("bmi_category", "")
        cals        = health_metrics.get("target_calories", "")
        web_section = ""
        if web_context:
            web_section = "\n\nRecent evidence from web search:\n" + web_context

        prompt = (
            "Given this user profile:\n"
            "- Goal: " + goal + "\n"
            "- BMI: " + str(bmi_val) + " (" + str(bmi_cat) + ")\n"
            "- Target calories: " + str(cals) + " kcal/day\n"
            "- Medical flags: " + flag_str
            + web_section + "\n\n"
            "Generate 7 concise, actionable, evidence-based, medically-safe "
            "personalised recommendations. Use the web search evidence above where "
            "relevant. Be specific, not generic.\n"
            "Return ONLY a JSON array of strings - no preamble, no markdown fences.\n"
            'Example: ["Drink water before meals", "Sleep 7-8 hours"]'
        )
        try:
            raw   = self.generate(prompt, max_tokens=600)
            clean = re.sub(r"```(?:json)?", "", raw).strip().strip("`")
            items = json.loads(clean)
            if isinstance(items, list) and items:
                log.info(f"[llm_service] Generated {len(items)} recommendations")
                return [str(i) for i in items]
        except (TimeoutError, Exception) as exc:
            log.warning(f"[llm_service] recommendations failed ({exc}) - using defaults")

        return list(_DEFAULT_RECOMMENDATIONS)

    def explain_plan(self, plan_summary: str, goal: str, warnings: List[str]) -> str:
        prompt = (
            "Briefly explain in 3-5 sentences why the following " + goal
            + " health plan is suitable. "
            "Mention key features and any important cautions.\n\n"
            "Plan summary: " + plan_summary + "\n"
            "Warnings: " + str(warnings)
        )
        try:
            return self.generate(prompt, max_tokens=300)
        except Exception as exc:
            log.warning(f"[llm_service] explain_plan failed ({exc}) - using fallback")
            return (
                "This " + goal + " plan has been generated based on your biometric data "
                "and any medical flags detected. Please follow the recommendations "
                "and consult your healthcare provider for personalised advice."
            )
