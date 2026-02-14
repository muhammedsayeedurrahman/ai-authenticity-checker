import sys
import os

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import torch
from torchvision import transforms
from PIL import Image

from core_models.face_deepfake_model import FaceDeepfakeModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

model = FaceDeepfakeModel().to(device)
model.load_state_dict(
    torch.load("models/image_face_model.pth", map_location=device, weights_only=False)
)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def predict_face(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        real_prob = model(image).item()

    fake_prob = 1.0 - real_prob
    return real_prob, fake_prob


if __name__ == "__main__":
    Image_path = "images.png"
    real_p, fake_p = predict_face(Image_path)

    print("\n=== FACE MODEL TEST ===")
    print("Image:", Image_path)
    print(f"Real prob: {real_p:.4f}")
    print(f"Fake prob: {fake_p:.4f}")

    if fake_p > 0.6:
        print("Prediction: Likely FAKE")
    elif fake_p < 0.4:
        print("Prediction: Likely REAL")
    else:
        print("Prediction: UNCERTAIN")
