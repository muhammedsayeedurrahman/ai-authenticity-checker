import sys
import os

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import torch
from torchvision import transforms
from PIL import Image
from general_ai.clip_model import GeneralAIImageDetector

# ---------------- CONFIG ----------------
GENERAL_AI_THRESHOLD = 0.15
# ---------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------- Load model --------
model = GeneralAIImageDetector(device).to(device)
model.load_state_dict(
    torch.load("models/general_ai_model.pth", map_location=device, weights_only=False)
)
model.eval()

# -------- CLIP transform --------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.48145466, 0.4578275, 0.40821073),
        std=(0.26862954, 0.26130258, 0.27577711)
    )
])

def predict_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        score = model(image).item()

    return score

# -------- Test --------
if __name__ == "__main__":
    test_image = "images.png"
    score = predict_image(test_image)

    print("AI-generation probability:", round(score, 4))

    if score > GENERAL_AI_THRESHOLD:
        print("Prediction: Likely AI-Generated (Non-Face)")
    else:
        print("Prediction: Likely Real")
