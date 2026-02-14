import sys
import os

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import torch
from torchvision import transforms

from utils.image_utils import load_rgb_image
from utils.explainability import explain_risk
from pipeline.face_gate import face_present
from core_models.face_deepfake_model import FaceDeepfakeModel
from core_models.dinov2_auth_model import DINOv2AuthModel
from core_models.efficientnet_auth_model import EfficientNetAuthModel

# Ensemble weights
WEIGHTS = {"dino": 0.4, "efficientnet": 0.35, "face": 0.25}
HIGH_CONFIDENCE_OVERRIDE = 0.9

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def load_models(models_dir="models"):
    """Load all three models and return them in eval mode."""
    face_model = FaceDeepfakeModel().to(device)
    face_model.load_state_dict(
        torch.load(os.path.join(models_dir, "image_face_model.pth"), map_location=device, weights_only=False)
    )
    face_model.eval()

    dino = DINOv2AuthModel().to(device)
    dino.load_state_dict(
        torch.load(os.path.join(models_dir, "dinov2_auth_model.pth"), map_location=device, weights_only=False)
    )
    dino.eval()

    eff = EfficientNetAuthModel().to(device)
    eff.load_state_dict(
        torch.load(os.path.join(models_dir, "efficientnet_auth_model.pth"), map_location=device, weights_only=False)
    )
    eff.eval()

    return face_model, dino, eff


def predict(image_path, face_model, dino, eff):
    """Run the full pipeline on a single image and return results dict."""
    image = load_rgb_image(image_path)
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        dino_prob = dino(tensor).item()
        eff_prob = eff(tensor).item()

        face_prob = 0.0
        has_face = face_present(image_path)
        if has_face:
            real_prob = face_model(tensor).item()
            face_prob = 1.0 - real_prob

    # Weighted average ensemble
    final_risk = (
        WEIGHTS["dino"] * dino_prob
        + WEIGHTS["efficientnet"] * eff_prob
        + WEIGHTS["face"] * face_prob
    )

    # High-confidence override: if any single model is very sure, trust it
    max_prob = max(face_prob, dino_prob, eff_prob)
    if max_prob > HIGH_CONFIDENCE_OVERRIDE:
        final_risk = max(final_risk, max_prob)

    return {
        "face_detected": has_face,
        "face_fake_prob": round(face_prob, 4),
        "dino_auth_prob": round(dino_prob, 4),
        "efficientnet_prob": round(eff_prob, 4),
        "final_risk_score": round(final_risk, 4),
        "verdict": explain_risk(final_risk),
    }


if __name__ == "__main__":
    IMAGE_PATH = "test.jpg"

    face_model, dino, eff = load_models()
    result = predict(IMAGE_PATH, face_model, dino, eff)

    print("\n=== IMAGE AUTHENTICITY RESULT ===")
    print("face_detected        :", result["face_detected"])
    print("face_fake_prob       :", result["face_fake_prob"])
    print("dino_auth_prob       :", result["dino_auth_prob"])
    print("efficientnet_prob    :", result["efficientnet_prob"])
    print("final_risk_score     :", result["final_risk_score"])
    print("verdict              :", result["verdict"])
