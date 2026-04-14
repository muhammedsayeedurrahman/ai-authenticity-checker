"""
Explainability utilities for deepfake detection.

Provides structured risk explanations with evidence summaries
for image, video, audio, and multimodal analysis.
"""


def explain_risk(score, model_scores=None):
    """
    Generate structured risk explanation for image/video analysis.

    Args:
        score: Final risk score (0.0 - 1.0).
        model_scores: Optional dict of per-model scores for evidence.

    Returns:
        str with detailed risk explanation.
    """
    if score > 0.8:
        level = "CRITICAL"
        desc = "Very high probability of AI manipulation"
    elif score > 0.6:
        level = "HIGH"
        desc = "Strong indicators of AI generation or manipulation"
    elif score > 0.4:
        level = "MEDIUM"
        desc = "Some manipulation indicators detected"
    elif score > 0.2:
        level = "LOW"
        desc = "Minor anomalies, likely authentic"
    else:
        level = "MINIMAL"
        desc = "No significant manipulation indicators"

    explanation = f"{level} RISK — {desc}"

    if model_scores:
        evidence = []
        if model_scores.get("vit_prob", 0) > 0.6:
            evidence.append("ViT detected deepfake patterns")
        if model_scores.get("face_prob", 0) > 0.6:
            evidence.append("facial manipulation artifacts found")
        if model_scores.get("forensic_prob", 0) > 0.5:
            evidence.append("noise/ELA inconsistency detected")
        if model_scores.get("frequency_prob", 0) > 0.5:
            evidence.append("frequency-domain anomalies")
        if model_scores.get("eff_prob", 0) > 0.6:
            evidence.append("EfficientNet flagged AI generation")
        if model_scores.get("dino_prob", 0) > 0.6:
            evidence.append("DINOv2 detected synthetic features")

        if evidence:
            explanation += f". Evidence: {'; '.join(evidence)}"

    return explanation


def explain_audio_risk(fake_prob):
    """Explain audio deepfake risk level."""
    if fake_prob > 0.7:
        return "HIGH RISK — AI-generated speech detected (voice cloning / TTS)"
    elif fake_prob > 0.5:
        return "MEDIUM RISK — Possible AI-generated audio"
    elif fake_prob > 0.3:
        return "LOW RISK — Inconclusive, minor anomalies detected"
    else:
        return "MINIMAL RISK — Audio appears authentic"


def explain_multimodal(modality_scores, final_score):
    """
    Generate explanation for multimodal fusion result.

    Args:
        modality_scores: dict with image/video/audio scores.
        final_score: fused risk score.

    Returns:
        str explanation.
    """
    active = {k: v for k, v in modality_scores.items() if v is not None}

    if not active:
        return "No modalities analyzed"

    parts = []
    for mod, score in active.items():
        if score > 70:
            parts.append(f"{mod}: high risk ({score}%)")
        elif score > 40:
            parts.append(f"{mod}: medium risk ({score}%)")
        else:
            parts.append(f"{mod}: low risk ({score}%)")

    risk_pct = final_score * 100
    if final_score > 0.7:
        verdict = "Strong evidence of manipulation across modalities"
    elif final_score > 0.4:
        verdict = "Partial manipulation indicators detected"
    else:
        verdict = "Content appears authentic across analyzed modalities"

    return f"{verdict}. Per-modality: {'; '.join(parts)}"
