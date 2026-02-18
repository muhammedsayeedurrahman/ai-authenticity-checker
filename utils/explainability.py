def explain_risk(score):
    if score > 0.7:
        return "High AI Risk (Photorealistic Manipulation)"
    elif score > 0.4:
        return "Medium AI Risk"
    else:
        return "Low AI Risk"


def explain_audio_risk(fake_prob):
    """Explain audio deepfake risk level."""
    if fake_prob > 0.7:
        return "High Risk — AI-generated speech detected (voice cloning / TTS)"
    elif fake_prob > 0.5:
        return "Medium Risk — Possible AI-generated audio"
    elif fake_prob > 0.3:
        return "Low Risk — Inconclusive, minor anomalies detected"
    else:
        return "Minimal Risk — Audio appears authentic"
