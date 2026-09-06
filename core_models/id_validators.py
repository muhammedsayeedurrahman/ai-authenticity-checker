"""
Format/checksum validators for common Indian government ID numbers
(Aadhaar, PAN, Voter ID / EPIC).

Not OCR - the user types (or leaves blank) the ID number printed on the
document; this validates whether that string is well-formed for the
selected ID type. Two different levels of rigor depending on what's
actually publicly documented, and the module is explicit about which is
which rather than implying more confidence than exists:

  - Aadhaar: UIDAI publishes that the 12th digit is a Verhoeff checksum
    over the first 11 digits. The Verhoeff algorithm itself is a fixed,
    public 1969 construction (not something that changes over time,
    unlike the legal citations elsewhere in this project) - implemented
    here in full, so an Aadhaar number that fails it is a genuine,
    verifiable defect, not a heuristic guess.
  - PAN / Voter ID (EPIC): the format (length, character classes) is
    published, but neither the Income Tax Department (PAN) nor the
    Election Commission (EPIC) publicly documents a check-digit
    algorithm. Only format is verified for these two - the validators
    say so explicitly in their `reason` text rather than implying a
    checksum that doesn't exist.

This is one signal among several in a document analysis, not a
standalone authenticity determination - a well-formed, checksum-valid
number does not by itself prove the document is genuine (a fabricated
number can be constructed to pass), and a malformed one does not by
itself prove it's fake (a typo in manual entry is far more likely than
forgery). Combine with the forensic/AI-generation signals in
core/pipeline.py::analyze_document.
"""

from __future__ import annotations

import re
from typing import Optional

# ---- Verhoeff algorithm tables (fixed, public; RFC/ISO-style constants) ----
_D = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    [1, 2, 3, 4, 0, 6, 7, 8, 9, 5],
    [2, 3, 4, 0, 1, 7, 8, 9, 5, 6],
    [3, 4, 0, 1, 2, 8, 9, 5, 6, 7],
    [4, 0, 1, 2, 3, 9, 5, 6, 7, 8],
    [5, 9, 8, 7, 6, 0, 4, 3, 2, 1],
    [6, 5, 9, 8, 7, 1, 0, 4, 3, 2],
    [7, 6, 5, 9, 8, 2, 1, 0, 4, 3],
    [8, 7, 6, 5, 9, 3, 2, 1, 0, 4],
    [9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
]
_P = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    [1, 5, 7, 6, 2, 8, 3, 0, 9, 4],
    [5, 8, 0, 3, 7, 9, 6, 1, 4, 2],
    [8, 9, 1, 6, 0, 4, 3, 5, 2, 7],
    [9, 4, 5, 3, 1, 2, 6, 8, 7, 0],
    [4, 2, 8, 6, 5, 7, 3, 9, 0, 1],
    [2, 7, 9, 3, 8, 0, 6, 4, 1, 5],
    [7, 0, 4, 6, 9, 1, 3, 2, 5, 8],
]
_INV = [0, 4, 3, 2, 1, 5, 6, 7, 8, 9]


def verhoeff_checksum_valid(number: str) -> bool:
    """Standard Verhoeff validation: True if the last digit is a valid
    checksum for the digits preceding it."""
    c = 0
    for i, digit in enumerate(reversed(number)):
        c = _D[c][_P[i % 8][int(digit)]]
    return c == 0


def generate_verhoeff_checksum(number: str) -> str:
    """Compute the Verhoeff check digit for `number`. Used only by this
    module's own tests to construct valid synthetic examples - never
    called from the analysis pipeline."""
    c = 0
    for i, digit in enumerate(reversed(number)):
        c = _D[c][_P[(i + 1) % 8][int(digit)]]
    return str(_INV[c])


def validate_aadhaar(number: str) -> dict:
    cleaned = re.sub(r"[\s-]", "", number or "")
    if not re.fullmatch(r"[2-9][0-9]{11}", cleaned):
        return {
            "valid": False,
            "reason": "Not a 12-digit Aadhaar number (must be 12 digits and not start with 0 or 1).",
        }
    if not verhoeff_checksum_valid(cleaned):
        return {
            "valid": False,
            "reason": "12 digits, but fails the Verhoeff checksum UIDAI uses for Aadhaar numbers.",
        }
    return {
        "valid": True,
        "reason": "Well-formed 12-digit number that passes the Verhoeff checksum.",
    }


def validate_pan(number: str) -> dict:
    cleaned = (number or "").strip().upper()
    if not re.fullmatch(r"[A-Z]{5}[0-9]{4}[A-Z]", cleaned):
        return {
            "valid": False,
            "reason": "Not in the standard PAN format (5 letters, 4 digits, 1 letter).",
        }
    return {
        "valid": True,
        "reason": "Matches the standard PAN format. The check letter's generation algorithm is "
                  "not publicly documented by the Income Tax Department, so only format is "
                  "verified here — not a checksum.",
    }


def validate_voter_id(number: str) -> dict:
    cleaned = (number or "").strip().upper()
    if not re.fullmatch(r"[A-Z]{3}[0-9]{7}", cleaned):
        return {
            "valid": False,
            "reason": "Not in the standard EPIC/Voter ID format (3 letters, 7 digits).",
        }
    return {
        "valid": True,
        "reason": "Matches the standard EPIC format. Exact formats vary by state and issuance "
                  "era, and there is no publicly documented checksum — only format is verified here.",
    }


VALIDATORS = {
    "aadhaar": validate_aadhaar,
    "pan": validate_pan,
    "voter_id": validate_voter_id,
}

ID_LABELS = {
    "aadhaar": "Aadhaar",
    "pan": "PAN",
    "voter_id": "Voter ID (EPIC)",
}


def validate_id_number(id_type: Optional[str], number: Optional[str]) -> Optional[dict]:
    """Returns None when there's nothing to validate (no type selected,
    or no number entered) - the caller should skip this signal entirely
    rather than render a false "valid": null case, since "not attempted"
    and "attempted and inconclusive" are different things."""
    if not id_type or id_type == "other" or not number or not number.strip():
        return None
    validator = VALIDATORS.get(id_type)
    if validator is None:
        return None
    result = validator(number)
    result["id_type"] = id_type
    result["id_label"] = ID_LABELS.get(id_type, id_type)
    return result
