"""
Cybercrime-risk advisory layer for ProofyX.

Deepfake automation is increasingly used for scams and fraud: AI voice
cloning for "executive fraud" phone calls, face-swapped photos to bypass
identity/KYC checks, and fabricated video calls to impersonate real people.
This module does not run any new ML model — it synthesizes signals the
pipeline already produces (model risk scores, EXIF/AI-software forensics,
audio manipulation typing, temporal splice indicators) into a single,
plain-language advisory so end users understand *why* a result might be
risky in a real-world fraud scenario, not just that a score crossed a
threshold.

This is a heuristic advisory, not a determination of criminal activity —
every non-"none" result carries DISCLAIMER to make that explicit.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional

DISCLAIMER = (
    "Automated heuristic advisory based on forensic signal patterns — "
    "not a legal or definitive determination of criminal activity."
)

# Gating thresholds: intentionally high-confidence-only, to avoid flagging
# ordinary AI-generated content (art, avatars) as "cybercrime risk". Each
# threshold sits above the corresponding modality's own HIGH-risk cutoff.
AUDIO_VOICE_CLONE_THRESHOLD = 0.60
IMAGE_IDENTITY_FRAUD_THRESHOLD = 0.65
VIDEO_IMPERSONATION_THRESHOLD = 0.65


class CybercrimeCategory(str, Enum):
    NONE = "none"
    VOICE_CLONE_FRAUD = "voice_clone_fraud"
    SYNTHETIC_IDENTITY = "synthetic_identity"
    IMPERSONATION_VIDEO = "impersonation_video"


_NONE_RESULT: dict[str, Any] = {
    "category": CybercrimeCategory.NONE.value,
    "label": "",
    "description": "",
    "advisory": "",
    "signals": [],
    "disclaimer": DISCLAIMER,
}


def _result(
    category: CybercrimeCategory, label: str, description: str,
    advisory: str, signals: list[str],
) -> dict[str, Any]:
    return {
        "category": category.value,
        "label": label,
        "description": description,
        "advisory": advisory,
        "signals": signals,
        "disclaimer": DISCLAIMER,
    }


def assess_audio_cybercrime_risk(
    risk_score: float, manipulation_type: str = "", evidence: Optional[list[str]] = None,
) -> dict[str, Any]:
    """Flag audio consistent with voice-cloning phone/impersonation fraud."""
    if risk_score < AUDIO_VOICE_CLONE_THRESHOLD:
        return dict(_NONE_RESULT)

    signals = [f"AI voice synthesis detected with {risk_score * 100:.0f}% confidence"]
    signals.extend(e for e in (evidence or []) if e != "no significant spectral artifacts")

    return _result(
        CybercrimeCategory.VOICE_CLONE_FRAUD,
        label="Voice-cloning fraud pattern",
        description=(
            "This audio shows strong signs of AI voice cloning or speech "
            "synthesis — the same technique used in impersonation scams "
            "such as fake executive, family-emergency, or bank-official "
            "phone calls that pressure victims into transferring money or "
            "sharing credentials."
        ),
        advisory=(
            "Do not act on financial requests, credential requests, or "
            "urgent instructions from this audio without independently "
            "verifying the speaker's identity through a separate, "
            "trusted channel."
        ),
        signals=signals,
    )


def assess_image_cybercrime_risk(
    risk_score: float, face_detected: bool,
    exif_suspicious: bool = False, exif_findings: Optional[list[str]] = None,
) -> dict[str, Any]:
    """Flag images consistent with fake-identity / KYC-bypass fraud."""
    if risk_score < IMAGE_IDENTITY_FRAUD_THRESHOLD or not face_detected or not exif_suspicious:
        return dict(_NONE_RESULT)

    signals = [f"Face manipulation detected with {risk_score * 100:.0f}% confidence"]
    signals.extend(exif_findings or [])

    return _result(
        CybercrimeCategory.SYNTHETIC_IDENTITY,
        label="Synthetic identity / identity-fraud pattern",
        description=(
            "This image combines a high-confidence face manipulation with "
            "an absence of genuine camera provenance (no camera metadata, "
            "or signs of AI-generation software) — a pattern associated "
            "with fake IDs, fraudulent profile photos, and attempts to "
            "bypass identity or biometric verification (KYC)."
        ),
        advisory=(
            "Treat this image as unverified for identity-proofing "
            "purposes. Request a live capture or additional verification "
            "before trusting it for account creation or approval."
        ),
        signals=signals,
    )


def assess_video_cybercrime_risk(
    risk_score: float, faces_detected_in_frames: int = 0,
    significant_jumps: int = 0, score_variance: float = 0.0,
) -> dict[str, Any]:
    """Flag videos consistent with fabricated video-call impersonation."""
    has_splice_signal = significant_jumps > 0 or score_variance > 0.02
    if risk_score < VIDEO_IMPERSONATION_THRESHOLD or not faces_detected_in_frames or not has_splice_signal:
        return dict(_NONE_RESULT)

    signals = [
        f"Face manipulation detected across sampled frames "
        f"({risk_score * 100:.0f}% average risk)",
    ]
    if significant_jumps > 0:
        signals.append(
            f"{significant_jumps} abrupt score jump(s) between adjacent "
            "frames (possible splice point)"
        )
    if score_variance > 0.02:
        signals.append("Inconsistent risk across frames (temporal instability)")

    return _result(
        CybercrimeCategory.IMPERSONATION_VIDEO,
        label="Video impersonation pattern",
        description=(
            "This video shows a high-confidence face manipulation with "
            "signs of frame-level splicing — consistent with fabricated "
            "video calls or statements used to impersonate real people "
            "(e.g. fake video-call scams or manipulated evidence)."
        ),
        advisory=(
            "Independently verify any claims, instructions, or "
            "identity made in this video before relying on it."
        ),
        signals=signals,
    )
