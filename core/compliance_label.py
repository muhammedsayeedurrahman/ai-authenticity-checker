"""
Compliance-labeling advisory layer for ProofyX (India IT Rules 2026).

India's IT Rules 2026 amendment requires intermediaries operating in India
to label synthetically-generated content, maintain traceability records,
and take down flagged harmful deepfakes within a 3-hour SLA. This module
does not run any new ML model — it maps the pipeline's existing
risk_score/confidence output onto that labeling taxonomy so a platform can
act on it (show a label, start an SLA clock, log an audit entry).

This is a heuristic compliance-workflow aid, not legal advice — every
label carries COMPLIANCE_DISCLAIMER to make that explicit. The taxonomy
below is engineering's reading of the amendment and should be reviewed by
counsel before being relied on for an actual regulatory determination;
see docs/COMPLIANCE.md.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

COMPLIANCE_DISCLAIMER = (
    "Automated heuristic determination for compliance-workflow support — "
    "not legal advice and not a substitute for the platform's own "
    "regulatory determination under applicable law."
)

REGULATORY_BASIS = (
    "India Information Technology (Intermediary Guidelines and Digital "
    "Media Ethics Code) Rules, 2026 amendment — synthetically-generated-"
    "information labeling and traceability obligations"
)

# Versioned so a later correction to the taxonomy doesn't retroactively
# change what a previously-stored label meant.
LABEL_RULESET_VERSION = "in-it-rules-2026.v1"

# 3-hour takedown SLA per the 2026 amendment.
DEFAULT_SLA_SECONDS = 10800

# Thresholds intentionally mirror the pipeline's own Verdict/Confidence
# cutoffs (core/types.py) rather than inventing a second scale.
FLAGGED_THRESHOLD = 0.60
ADVISORY_THRESHOLD = 0.45


class LabelCode(str, Enum):
    SYNTHETICALLY_GENERATED = "synthetically_generated"
    POSSIBLY_SYNTHETIC = "possibly_synthetic"
    NO_SYNTHETIC_INDICATORS = "no_synthetic_indicators"
    INDETERMINATE = "indeterminate"


_LABEL_DISPLAY: dict[LabelCode, str] = {
    LabelCode.SYNTHETICALLY_GENERATED: "Synthetically generated (AI content)",
    LabelCode.POSSIBLY_SYNTHETIC: "Possibly synthetic — recommend manual review",
    LabelCode.NO_SYNTHETIC_INDICATORS: "No synthetic-content indicators",
    LabelCode.INDETERMINATE: "Indeterminate — analysis incomplete",
}

_RECOMMENDED_ACTION: dict[LabelCode, str] = {
    LabelCode.SYNTHETICALLY_GENERATED: "label_and_review",
    LabelCode.POSSIBLY_SYNTHETIC: "manual_review",
    LabelCode.NO_SYNTHETIC_INDICATORS: "none",
    LabelCode.INDETERMINATE: "manual_review",
}


def _classify(risk_score: float, confidence: str) -> LabelCode:
    if risk_score >= FLAGGED_THRESHOLD:
        if confidence == "HIGH":
            return LabelCode.SYNTHETICALLY_GENERATED
        return LabelCode.POSSIBLY_SYNTHETIC
    if risk_score >= ADVISORY_THRESHOLD:
        return LabelCode.POSSIBLY_SYNTHETIC
    return LabelCode.NO_SYNTHETIC_INDICATORS


def build_compliance_label(
    risk_score: float,
    confidence: str,
    detector_version: str = "",
    label_basis: Optional[list[str]] = None,
    cybercrime_category: str = "none",
    flagged_by_complaint: bool = False,
    error: bool = False,
) -> dict[str, Any]:
    """Build an India-IT-Rules-2026-shaped compliance label for one analysis.

    Args:
        risk_score: P(fake) from the pipeline, 0.0-1.0.
        confidence: "HIGH" | "MEDIUM" | "LOW", from Confidence.from_risk_score.
        detector_version: identifies which fusion/model path produced the
            score (e.g. "proofyx/learned", "proofyx/corefakenet"), stored
            alongside the label so a past determination stays reconstructable
            even after the pipeline changes.
        label_basis: human-readable evidence strings for why the label was
            assigned (e.g. model agreement, EXIF findings, temporal jumps).
        cybercrime_category: the category from core/cybercrime_risk.py's
            assess_*_cybercrime_risk (or "none"). A non-"none" category is
            treated as a proxy signal that this content is in a harm class
            the takedown SLA is meant for.
        flagged_by_complaint: set True when the caller knows a platform
            grievance/complaint was actually filed against this content —
            ProofyX cannot infer this on its own, so it defaults False.
        error: True when the underlying analysis failed — always yields
            INDETERMINATE regardless of risk_score, since a failed
            analysis must never be mistaken for a clean one.

    Returns:
        Plain JSON-serializable dict, independent of any dict passed in
        via label_basis (the input list is copied, never referenced).
    """
    label_code = LabelCode.INDETERMINATE if error else _classify(risk_score, confidence)

    sla_applies = (
        not error
        and label_code == LabelCode.SYNTHETICALLY_GENERATED
        and (cybercrime_category != "none" or flagged_by_complaint)
    )

    return {
        "label_code": label_code.value,
        "label_display": _LABEL_DISPLAY[label_code],
        "requires_visible_label": label_code in (
            LabelCode.SYNTHETICALLY_GENERATED, LabelCode.POSSIBLY_SYNTHETIC,
        ),
        "requires_embedded_metadata": label_code == LabelCode.SYNTHETICALLY_GENERATED,
        "label_basis": list(label_basis or []),
        "regulatory_basis": REGULATORY_BASIS,
        "ruleset_version": LABEL_RULESET_VERSION,
        "detector_version": detector_version,
        "risk_score": float(risk_score),
        "confidence": confidence,
        "recommended_action": _RECOMMENDED_ACTION[label_code],
        "sla_applies": sla_applies,
        "sla_deadline_seconds": DEFAULT_SLA_SECONDS if sla_applies else None,
        "assessed_at": datetime.now(timezone.utc).isoformat(),
        "disclaimer": COMPLIANCE_DISCLAIMER,
    }
