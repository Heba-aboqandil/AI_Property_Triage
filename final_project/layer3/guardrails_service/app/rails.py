"""
NeMo Guardrails wrapper.

Two separate LLMRails instances share the same config directory:
- input_rails  → used by POST /check/input
- output_rails → used by POST /check/output

The LLM is called only when the deterministic keyword checks do not produce
a definitive verdict, keeping latency low for obvious cases.
"""

import os
import re
from pathlib import Path

from nemoguardrails import LLMRails, RailsConfig

_RAILS_DIR = Path(__file__).parent.parent / "rails"

# ---------------------------------------------------------------------------
# Deterministic pre-checks (fast path — no LLM call needed)
# ---------------------------------------------------------------------------

_SPAM_PATTERNS = re.compile(
    r"(click here|free money|act now|limited time|you have been selected"
    r"|buy cheap|winner winner|make \$\d+)",
    re.IGNORECASE,
)

_OFFENSIVE_PATTERNS = re.compile(
    r"\b(hate|kill|terrorist|porn|explicit|nude)\b",
    re.IGNORECASE,
)

_LISTING_SIGNALS = re.compile(
    r"\b(bedroom|apartment|villa|house|property|listing|sqm|floor|kitchen"
    r"|bathroom|balcony|parking|garden|office|commercial|industrial|retail"
    r"|for sale|for rent|price|location|city|neighborhood)\b",
    re.IGNORECASE,
)

_FABRICATION_PATTERNS = re.compile(
    r"(guaranteed (price|value|return)|legally (guaranteed|certified)"
    r"|certified by law|ISO \d+|LEED (platinum|gold|certified)"
    r"|zoning (approved|certified)|ownership (guaranteed|transferred))",
    re.IGNORECASE,
)


def _deterministic_input_check(text: str) -> tuple[bool, str | None]:
    """Return (passed, reason). reason is None when passed=True."""
    if _SPAM_PATTERNS.search(text):
        return False, "Input identified as spam or promotional content."
    if _OFFENSIVE_PATTERNS.search(text):
        return False, "Input contains offensive or inappropriate content."
    if len(text.strip()) < 20:
        return False, "Input is too short to be a valid property listing."
    return True, None


def _deterministic_output_check(text: str) -> tuple[bool, str | None]:
    if _FABRICATION_PATTERNS.search(text):
        return False, "Report contains a fabricated certification or false legal claim."
    return True, None


# ---------------------------------------------------------------------------
# NeMo Guardrails LLM-backed check (slow path)
# ---------------------------------------------------------------------------

_config: RailsConfig | None = None
_rails: LLMRails | None = None


def _get_rails() -> LLMRails:
    global _config, _rails
    if _rails is None:
        _config = RailsConfig.from_path(str(_RAILS_DIR))
        _rails = LLMRails(_config)
    return _rails


async def check_input(text: str) -> dict:
    """
    Returns {"pass": bool, "reason": str|None, "safe_text": None}
    """
    passed, reason = _deterministic_input_check(text)
    if not passed:
        return {"pass": False, "reason": reason, "safe_text": None}

    # Check listing signal — if no property keywords at all, use LLM
    if not _LISTING_SIGNALS.search(text):
        try:
            rails = _get_rails()
            response = await rails.generate_async(
                messages=[{"role": "user", "content": text}]
            )
            content: str = response.get("content", "")
            if "REJECTED" in content:
                reason = content.replace("REJECTED:", "").strip()
                return {"pass": False, "reason": reason, "safe_text": None}
        except Exception:
            # Fail open on LLM error — let the listing through with a flag
            pass

    return {"pass": True, "reason": None, "safe_text": None}


async def check_output(text: str) -> dict:
    """
    Returns {"pass": bool, "reason": str|None, "safe_text": str|None}
    """
    passed, reason = _deterministic_output_check(text)
    if not passed:
        return {"pass": False, "reason": reason, "safe_text": None}

    try:
        rails = _get_rails()
        response = await rails.generate_async(
            messages=[{"role": "user", "content": text}]
        )
        content: str = response.get("content", "")
        if "FLAGGED" in content:
            reason = content.replace("FLAGGED:", "").strip()
            return {"pass": False, "reason": reason, "safe_text": None}
    except Exception:
        pass

    return {"pass": True, "reason": None, "safe_text": text}
