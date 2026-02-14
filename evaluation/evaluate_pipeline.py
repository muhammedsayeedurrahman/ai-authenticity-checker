import sys
import os
from tqdm import tqdm

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    classification_report
)

from pipeline.face_gate import face_present
from core_models.face_deepfake_model import FaceDeepfakeModel
from general_ai.clip_model import GeneralAIImageDetector

# ---------------- CONFIG ----------------
DATA_DIR = "data/pipeline_test"

FACE_FAKE_THRESHOLD = 0.6
GENERAL_AI_THRESHOLD = 0.15
# --------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# -------- Transforms --------
face_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

clip_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.48145466, 0.4578275, 0.40821073),
        std=(0.26862954, 0.26130258, 0.27577711)
    )
])

# -------- Load Models --------
face_model = FaceDeepfakeModel().to(device)
face_model.load_state_dict(
    torch.load("models/image_face_model.pth", map_location=device, weights_only=False)
)
face_model.eval()

general_model = GeneralAIImageDetector(device).to(device)
general_model.load_state_dict(
    torch.load("models/general_ai_model.pth", map_location=device, weights_only=False)
)
general_model.eval()

# -------- Evaluation --------
y_true = []    # 0 = Real, 1 = Fake
y_pred = []    # 0 = Real, 1 = Fake
y_scores = []  # P(Fake)

for label_name, label in [("real", 0), ("fake", 1)]:
    folder = os.path.join(DATA_DIR, label_name)

    for file in tqdm(os.listdir(folder), desc=f"Evaluating {label_name}"):
        image_path = os.path.join(folder, file)
        image = Image.open(image_path).convert("RGB")

        face_detected = face_present(image_path)

        with torch.no_grad():
            # General AI model (always run)
            clip_input = clip_transform(image).unsqueeze(0).to(device)
            general_ai_prob = general_model(clip_input).item()

            # Face deepfake model (only if face detected)
            face_ai_prob = 0.0
            if face_detected:
                face_input = face_transform(image).unsqueeze(0).to(device)
                real_prob = face_model(face_input).item()
                face_ai_prob = 1.0 - real_prob

        # -------- Decision Logic --------
        face_flag = face_ai_prob > FACE_FAKE_THRESHOLD
        general_flag = general_ai_prob > GENERAL_AI_THRESHOLD

        final_flag = face_flag or general_flag   # True = Fake
        final_prob = max(face_ai_prob, general_ai_prob)

        y_true.append(label)
        y_pred.append(1 if final_flag else 0)
        y_scores.append(final_prob)

# -------- Metrics --------
y_true = np.array(y_true)
y_pred = np.array(y_pred)
y_scores = np.array(y_scores)

accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, pos_label=1)
recall = recall_score(y_true, y_pred, pos_label=1)
f1 = f1_score(y_true, y_pred, pos_label=1)
roc_auc = roc_auc_score(y_true, y_scores)

cm = confusion_matrix(y_true, y_pred)

print("\n=== PIPELINE EVALUATION RESULTS ===\n")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1-score : {f1:.4f}")
print(f"ROC-AUC  : {roc_auc:.4f}")

print("\nConfusion Matrix (rows=true, cols=pred):")
print(cm)

print("\nClassification Report:")
print(
    classification_report(
        y_true,
        y_pred,
        target_names=["Real", "Fake"]
    )
)
