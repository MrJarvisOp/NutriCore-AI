"""
parser.py
---------
Hybrid medical-report parser:
  1. Regex extraction (expanded — Task 3)
  2. Name normalisation (alias table)
  3. LLM fallback ONLY for truly unrecognised edge cases

Task 3: LLM fallback threshold raised from <2 fields to <4 fields.
        Comprehensive regex patterns added for:
        Hemoglobin, Iron, Ferritin, Vitamin D, Vitamin B12, TSH,
        Blood Sugar, Lipid Profile, Liver Function (ALT/AST),
        Kidney Function (Creatinine/BUN/Urea), Uric Acid, Electrolytes.
"""
from __future__ import annotations

import re
from typing import Any, Dict, Optional

from app.models.schemas import MedicalData
from app.utils.logger import get_logger

log = get_logger(__name__)

# ─── Alias / normalisation table ─────────────────────────────────────────────
# Maps abbreviations and synonyms to canonical field names used in PATTERNS.
ALIASES: Dict[str, str] = {
    # Haemoglobin
    r"\bHb\b":              "hemoglobin",
    r"\bHGB\b":             "hemoglobin",
    r"\bHaemoglobin\b":     "hemoglobin",
    # Cholesterol
    r"\bTC\b":              "cholesterol",
    r"\bTotal\s+Chol\b":    "cholesterol",
    # Lipids
    r"\bLDL-C\b":           "ldl",
    r"\bHDL-C\b":           "hdl",
    r"\bTG\b":              "triglycerides",
    r"\bTriglycerids\b":    "triglycerides",   # common misspelling in OCR
    # Blood sugar
    r"\bFasting\s+Glucose\b": "blood_sugar",
    r"\bFBS\b":             "blood_sugar",
    r"\bRBS\b":             "blood_sugar",     # random blood sugar
    r"\bPPBS\b":            "blood_sugar",
    # HbA1c
    r"\bHbA1c\b":           "hba1c",
    r"\bA1C\b":             "hba1c",
    r"\bGlycated\s+Hb\b":   "hba1c",
    # Thyroid
    r"\bTSH\b":             "thyroid_tsh",
    r"\bT3\b":              "t3",              # free T3 (not stored — alias only)
    r"\bT4\b":              "t4",
    # Vitamins
    r"\bVit\.?\s*D\b":      "vitamin_d",
    r"\b25-OH\s*Vit\s*D\b": "vitamin_d",
    r"\bVit\.?\s*B-?12\b":  "vitamin_b12",
    r"\bCobalamin\b":        "vitamin_b12",
    # Iron panel
    r"\bSerum\s+Iron\b":    "iron",
    r"\bSr\.?\s*Iron\b":    "iron",
    r"\bFerritin\b":        "ferritin",
    r"\bTIBC\b":            "tibc",
    # Liver
    r"\bALT\b":             "alt",
    r"\bSGPT\b":            "alt",
    r"\bAST\b":             "ast",
    r"\bSGOT\b":            "ast",
    r"\bALP\b":             "alp",
    # Kidney
    r"\bCreatinine\b":      "creatinine",
    r"\bBlood\s+Urea\s+Nitrogen\b": "urea",
    r"\bBUN\b":             "urea",
    r"\bUrea\b":            "urea",
    r"\bUric\s+Acid\b":     "uric_acid",
    r"\beGFR\b":            "egfr",
    # Electrolytes
    r"\bSodium\b":          "sodium",
    r"\bPotassium\b":       "potassium",
    r"\bChloride\b":        "chloride",
    # BP
    r"\bBP\b":              "blood_pressure",
    r"\bBlood\s+Pressure\b": "blood_pressure",
}

# ─── Comprehensive regex patterns (Task 3) ────────────────────────────────────
# Each pattern captures the FIRST numeric value after the field label.
# Uses a shared value pattern: optional spaces/colons/equals, then the number.
_V  = r"[\s:=\--]+(\d{1,4}(?:\.\d+)?)"   # numeric value
_BP = r"[\s:=\--]+(\d{2,3}/\d{2,3})"      # blood pressure  "120/80"

PATTERNS: Dict[str, re.Pattern] = {
    # ── Lipid panel ──────────────────────────────────────────────────────────
    "cholesterol":    re.compile(
        r"(?:total\s+)?cholesterol" + _V, re.IGNORECASE),
    "ldl":            re.compile(
        r"(?:LDL(?:\s*[--]\s*C)?|low[\s-]density\s+lipoprotein)" + _V, re.IGNORECASE),
    "hdl":            re.compile(
        r"(?:HDL(?:\s*[--]\s*C)?|high[\s-]density\s+lipoprotein)" + _V, re.IGNORECASE),
    "triglycerides":  re.compile(
        r"triglycerides?" + _V, re.IGNORECASE),

    # ── Blood sugar ───────────────────────────────────────────────────────────
    "blood_sugar":    re.compile(
        r"(?:fasting\s+)?(?:blood\s+)?(?:glucose|sugar|FBS|RBS|PPBS)" + _V,
        re.IGNORECASE),
    "hba1c":          re.compile(
        r"(?:HbA1c|A1C|hba1c|glycated\s+h[ae]moglobin)" + _V + r"\s*%?",
        re.IGNORECASE),

    # ── Complete blood count ───────────────────────────────────────────────────
    "hemoglobin":     re.compile(
        r"(?:h[ae]moglobin|HGB?)" + _V, re.IGNORECASE),

    # ── Iron panel ────────────────────────────────────────────────────────────
    "iron":           re.compile(
        r"(?:serum\s+)?(?:Sr\.?\s*)?iron" + _V, re.IGNORECASE),
    "ferritin":       re.compile(
        r"ferritin" + _V, re.IGNORECASE),

    # ── Thyroid ───────────────────────────────────────────────────────────────
    "thyroid_tsh":    re.compile(
        r"TSH" + _V, re.IGNORECASE),

    # ── Vitamins ──────────────────────────────────────────────────────────────
    "vitamin_d":      re.compile(
        r"(?:vitamin\s+D|25-OH\s*vit\s*D)" + _V, re.IGNORECASE),
    "vitamin_b12":    re.compile(
        r"(?:vitamin\s+B-?12|cobalamin)" + _V, re.IGNORECASE),

    # ── Liver function ────────────────────────────────────────────────────────
    "alt":            re.compile(
        r"(?:ALT|SGPT)" + _V, re.IGNORECASE),
    "ast":            re.compile(
        r"(?:AST|SGOT)" + _V, re.IGNORECASE),

    # ── Kidney function ───────────────────────────────────────────────────────
    "creatinine":     re.compile(
        r"creatinine" + _V, re.IGNORECASE),
    "urea":           re.compile(
        r"(?:blood\s+urea\s+nitrogen|BUN|urea)" + _V, re.IGNORECASE),
    "uric_acid":      re.compile(
        r"uric[\s_]+acid" + _V, re.IGNORECASE),  # matches both space and underscore (post-alias)

    # ── Electrolytes ──────────────────────────────────────────────────────────
    "sodium":         re.compile(
        r"sodium" + _V, re.IGNORECASE),
    "potassium":      re.compile(
        r"potassium" + _V, re.IGNORECASE),

    # ── Blood pressure ────────────────────────────────────────────────────────
    "blood_pressure": re.compile(
        r"(?:blood[\s_]+pressure|BP)" + _BP, re.IGNORECASE),  # matches post-alias blood_pressure too
}

# Fields stored on MedicalData (all PATTERNS keys that map to model fields)
_MODEL_FIELDS = {
    "cholesterol", "ldl", "hdl", "triglycerides", "blood_sugar", "hba1c",
    "hemoglobin", "iron", "ferritin", "thyroid_tsh", "vitamin_d", "vitamin_b12",
    "alt", "ast", "creatinine", "urea", "uric_acid", "sodium", "potassium",
    "blood_pressure",
}


class MedicalReportParser:
    """Parse raw text from a medical report into structured MedicalData."""

    def __init__(self, llm_service=None):
        self.llm = llm_service

    # ─── Public ──────────────────────────────────────────────────────────────

    def parse(self, raw_text: str) -> MedicalData:
        normalised = self._normalise(raw_text)
        extracted  = self._regex_extract(normalised)

        log.debug(
            f"[parser] Regex found {len(extracted)} field(s): "
            f"{list(extracted.keys())}"
        )

        # Task 3: raise LLM fallback threshold — only call LLM if very few
        #         recognised fields were found (< 4 instead of < 2).
        if self.llm and self._extraction_is_thin(extracted):
            log.info(
                f"[parser] Only {len(extracted)} field(s) from regex - "
                "using LLM fallback for edge-case values"
            )
            llm_data = self._llm_fallback(raw_text)
            # Regex results take priority; LLM fills gaps only
            for k, v in llm_data.items():
                if k not in extracted or extracted[k] is None:
                    extracted[k] = v
        else:
            log.info(
                f"[parser] Regex extracted {len(extracted)} field(s) - "
                "LLM fallback not needed"
            )

        # Filter to only keys that exist on MedicalData
        model_kwargs = {k: v for k, v in extracted.items() if k in _MODEL_FIELDS}
        medical = MedicalData(raw_text=raw_text, **model_kwargs)
        medical.flags = self._compute_flags(medical)
        log.info(f"[parser] Parsed medical data - flags: {medical.flags}")
        return medical

    # ─── Private ─────────────────────────────────────────────────────────────

    @staticmethod
    def _normalise(text: str) -> str:
        """Apply alias substitutions so downstream patterns get canonical labels."""
        for alias_pattern, canonical in ALIASES.items():
            text = re.sub(alias_pattern, canonical, text, flags=re.IGNORECASE)
        return text

    @staticmethod
    def _regex_extract(text: str) -> Dict[str, Any]:
        results: Dict[str, Any] = {}
        for field, pattern in PATTERNS.items():
            match = pattern.search(text)
            if match:
                raw_val = match.group(1)
                if field == "blood_pressure":
                    results[field] = raw_val
                else:
                    try:
                        results[field] = float(raw_val)
                    except ValueError:
                        pass
        return results

    @staticmethod
    def _extraction_is_thin(data: Dict[str, Any]) -> bool:
        """Task 3: threshold raised to 4 to reduce unnecessary LLM calls."""
        return len(data) < 4

    def _llm_fallback(self, raw_text: str) -> Dict[str, Any]:
        """Ask the LLM to extract remaining key medical values as JSON."""
        prompt = f"""
Extract medical values from the lab report below.
Return ONLY a valid JSON object. Include only keys where values are present.
Keys: cholesterol, ldl, hdl, triglycerides, blood_sugar, hba1c, hemoglobin,
      iron, ferritin, thyroid_tsh, vitamin_d, vitamin_b12,
      alt, ast, creatinine, urea, uric_acid, sodium, potassium, blood_pressure.

All numeric values must be plain numbers (no units).
blood_pressure must be a string like "120/80".

Lab report:
\"\"\"
{raw_text[:3000]}
\"\"\"
"""
        try:
            import json
            response = self.llm.generate(prompt, max_tokens=512)
            clean = re.sub(r"```(?:json)?", "", response).strip().strip("`")
            parsed = json.loads(clean)
            log.debug(f"[parser] LLM extracted {len(parsed)} field(s)")
            return parsed
        except Exception as exc:
            log.warning(f"[parser] LLM fallback failed: {exc}")
            return {}

    # ─── Flag logic ───────────────────────────────────────────────────────────

    @staticmethod
    def _compute_flags(med: MedicalData) -> list:
        from app.config import (
            CHOLESTEROL_HIGH, BLOOD_SUGAR_HIGH,
            BLOOD_PRESSURE_SYSTOLIC_HIGH,
            HEMOGLOBIN_LOW_MALE, TRIGLYCERIDES_HIGH,
        )
        flags = []

        # Lipids
        if med.cholesterol and med.cholesterol > CHOLESTEROL_HIGH:
            flags.append("high_cholesterol")
        if med.ldl and med.ldl > 130:
            flags.append("high_ldl")
        if med.triglycerides and med.triglycerides > TRIGLYCERIDES_HIGH:
            flags.append("high_triglycerides")

        # Blood sugar
        if med.blood_sugar and med.blood_sugar > BLOOD_SUGAR_HIGH:
            flags.append("pre_diabetic_or_diabetic")
        if med.hba1c and med.hba1c > 5.7:
            flags.append("diabetes_risk")

        # Blood count / iron
        if med.hemoglobin and med.hemoglobin < HEMOGLOBIN_LOW_MALE:
            flags.append("anemia_risk")
        if med.ferritin and med.ferritin < 12:
            flags.append("iron_deficiency")
        if med.iron and med.iron < 60:
            flags.append("low_serum_iron")

        # Thyroid
        if med.thyroid_tsh:
            if med.thyroid_tsh > 4.5:
                flags.append("hypothyroidism_risk")
            elif med.thyroid_tsh < 0.4:
                flags.append("hyperthyroidism_risk")

        # Blood pressure
        if med.blood_pressure:
            try:
                systolic = int(str(med.blood_pressure).split("/")[0])
                if systolic >= BLOOD_PRESSURE_SYSTOLIC_HIGH:
                    flags.append("hypertension")
            except (ValueError, IndexError):
                pass

        # Vitamins
        if med.vitamin_d and med.vitamin_d < 20:
            flags.append("vitamin_d_deficiency")
        if med.vitamin_b12 and med.vitamin_b12 < 200:
            flags.append("vitamin_b12_deficiency")

        # Liver function
        if med.alt and med.alt > 56:
            flags.append("elevated_liver_alt")
        if med.ast and med.ast > 40:
            flags.append("elevated_liver_ast")

        # Kidney function
        if med.creatinine and med.creatinine > 1.3:
            flags.append("elevated_creatinine")
        if med.urea and med.urea > 20:
            flags.append("elevated_urea")
        if med.uric_acid and med.uric_acid > 7.0:
            flags.append("high_uric_acid")

        # Electrolytes
        if med.sodium and (med.sodium < 136 or med.sodium > 145):
            flags.append("sodium_imbalance")
        if med.potassium and (med.potassium < 3.5 or med.potassium > 5.0):
            flags.append("potassium_imbalance")

        return flags
