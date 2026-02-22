"""
pipeline/normalize.py
---------------------
Stage 3: Colloquial Kannada → Standard Written Kannada normalization.

Responsibilities:
- Fix common phonetic/colloquial variations in Kannada script.
- Perform purely deterministic text substitutions (no ML required).
- Preserve original meaning — only orthographic corrections.

This runs BEFORE translation as a pre-processing step to improve
IndicTrans2 translation quality on dialectal/informal Kannada.
"""

import logging
import re

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Replacement table
# ---------------------------------------------------------------------------
# Each entry: (colloquial_form, standard_form)
# Ordered from longest to shortest to avoid partial matches getting shadowed.

_REPLACEMENTS: list[tuple[str, str]] = [
    # Verbs / pronouns
    ("ಮೊಂಚಿನೆ", "ಮೊದಲು"),      # "earlier/first" colloquial → standard
    ("ತುಮ್ಬ",   "ತುಂಬ"),        # "very/a lot" — anusvara drop
    ("ಸೊಲ್ಪ",   "ಸ್ವಲ್ಪ"),       # "a little"
    ("ಯೂಸ್",    "ಬಳಸಿ"),        # "use" (English borrowing → Kannada)
    ("ಇಸ್ಟ್",   "ಇಷ್ಟ"),        # "like/love" — ṭa mispelling
    ("ನಿವು",    "ನೀವು"),        # "you" — vowel shortening
    ("ಹೇಂಗೆ",   "ಹೇಗೆ"),        # "how" — dialectal nasal
    ("ಎಂತ",     "ಏನು"),         # "what" — dialectal
    ("ಅದ್ಕೆ",   "ಅದಕ್ಕೆ"),       # "for that" — elision
    ("ಇದ್ಕೆ",   "ಇದಕ್ಕೆ"),       # "for this" — elision
    ("ಕೊಡ್ರಿ",  "ಕೊಡಿ"),        # "give (polite)" — dialectal
    ("ಮಾಡ್ಕೋ",  "ಮಾಡಿಕೊ"),      # "do for yourself" — contraction
    ("ಹೋಗ್ಬೇಕು","ಹೋಗಬೇಕು"),     # "need to go"
    ("ಬರ್ತಾ",   "ಬರುತ್ತಾ"),      # progressive suffix elision
    ("ಮಾಡ್ತೀನಿ","ಮಾಡುತ್ತೇನೆ"),   # "I will do" — strong elision
    ("ಇಟ್ಕೊ",   "ಇಟ್ಟುಕೊ"),      # "keep for yourself"
    ("ಗೊತ್ತಿಲ್ಲ","ಗೊತ್ತಿಲ್ಲ"),   # already correct; listed for completeness
    ("ನಮ್ಮ",    "ನಮ್ಮ"),         # already correct; no-op guard
    ("ಆಯ್ತು",   "ಆಯಿತು"),       # "it's done" — vowel drop
    ("ಮಾಡ್ಬೇಡ", "ಮಾಡಬೇಡ"),      # "don't do"
    ("ನೋಡ್ತಾ",  "ನೋಡುತ್ತಾ"),     # "while watching"
]


def normalize_kannada(text: str) -> str:
    """
    Normalize colloquial / dialectal Kannada text to standard written form.

    Applies a prioritized list of deterministic string replacements.
    Does NOT change meaning or word order.

    Args:
        text: Raw Kannada string (may be colloquial / phonetically spelled).

    Returns:
        Normalized Kannada string with standard orthography.

    Example:
        >>> normalize_kannada("ಸೊಲ್ಪ ನೋಡ್ತಾ ಇರಿ")
        'ಸ್ವಲ್ಪ ನೋಡುತ್ತಾ ಇರಿ'
    """
    normalized = text

    for colloquial, standard in _REPLACEMENTS:
        if colloquial == standard:
            continue  # no-op guard — skip identity mappings

        if colloquial in normalized:
            normalized = normalized.replace(colloquial, standard)
            logger.debug(
                "[normalize] %r → %r  (in: %r)",
                colloquial, standard, text[:60],
            )

    return normalized
