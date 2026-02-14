import sys
import os

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import torch
from torchvision import transforms
from PIL import Image

from general_ai.dinov2_model import DINOv2ImageDetector

# ---------------- CONFIG ----------------
IMAGE_PATH = "1.png"   # change this to test different images
AI_THRESHOLD = 0.4
# --------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# -------- ImageNet normalization (DINOv2) --------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# -------- Load Model --------
model_path = "models/dinov2_model.pth"
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model not found: {model_path}")

print("Loading model from:", model_path)
print("Model last modified:", os.path.getmtime(model_path))

model = DINOv2ImageDetector().to(device)
model.load_state_dict(
    torch.load(model_path, map_location=device, weights_only=False)
)
model.eval()

def predict_image(image_path):
    image = Image.open(image_path).convert("RGB")

    # 🔎 Debug: confirm image identity
    print("Image size :", image.size)
    print("Image mode :", image.mode)

    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model.forward_logits(image_tensor)
        prob = torch.sigmoid(logits)

    return logits.item(), prob.item()


if __name__ == "__main__":
    if not os.path.exists(IMAGE_PATH):
        raise FileNotFoundError(f"Image not found: {IMAGE_PATH}")

    logits, score = predict_image(IMAGE_PATH)

    print("\n=== DINOv2 GENERAL AI DETECTION (DEBUG) ===")
    print(f"Image path     : {IMAGE_PATH}")
    print(f"Raw logits     : {logits:.4f}")
    print(f"AI probability : {score:.4f}")

    if score >= AI_THRESHOLD:
        print("Prediction     : Likely AI-Generated")
    else:
        print("Prediction     : Likely Real")
