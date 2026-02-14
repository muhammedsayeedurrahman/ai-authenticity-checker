def explain_risk(score):
    if score > 0.7:
        return "High AI Risk (Photorealistic Manipulation)"
    elif score > 0.4:
        return "Medium AI Risk"
    else:
        return "Low AI Risk"
