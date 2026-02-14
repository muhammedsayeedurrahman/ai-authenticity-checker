import sys
import os

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import torch
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    classification_report
)
from core_models.face_deepfake_model import FaceDeepfakeModel

# ---------------- CONFIG ----------------
TEST_DIR = "data/image"     # using full dataset for now
BATCH_SIZE = 16
REAL_THRESHOLD = 0.4        # probability threshold for REAL class
# ---------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Normalization must match training
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

dataset = datasets.ImageFolder(TEST_DIR, transform=transform)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

# Load model
model = FaceDeepfakeModel().to(device)
model.load_state_dict(
    torch.load("models/image_face_model.pth", map_location=device, weights_only=False)
)
model.eval()

y_true = []
y_pred = []
y_fake_scores = []

with torch.no_grad():
    for images, labels in loader:
        images = images.to(device)

        # Model outputs P(REAL)
        real_probs = model(images).cpu().numpy().flatten()
        fake_probs = 1.0 - real_probs

        # ImageFolder labels:
        # 0 = fake
        # 1 = real
        preds = (real_probs > REAL_THRESHOLD).astype(int)

        y_true.extend(labels.numpy())
        y_pred.extend(preds)
        y_fake_scores.extend(fake_probs)

# Convert to numpy
y_true = np.array(y_true)
y_pred = np.array(y_pred)
y_fake_scores = np.array(y_fake_scores)

# -------- METRICS --------
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, pos_label=0)
recall = recall_score(y_true, y_pred, pos_label=0)
f1 = f1_score(y_true, y_pred, pos_label=0)

# ROC-AUC -> fake is the positive class (label 0)
roc_auc = roc_auc_score((y_true == 0).astype(int), y_fake_scores)

cm = confusion_matrix(y_true, y_pred)

print("\n=== IMAGE DEEPFAKE DETECTION EVALUATION ===\n")
print("Accuracy :", round(accuracy, 4))
print("Precision:", round(precision, 4))
print("Recall   :", round(recall, 4))
print("F1-score :", round(f1, 4))
print("ROC-AUC  :", round(roc_auc, 4))

print("\nConfusion Matrix (rows=true, cols=pred):")
print(cm)

print("\nClassification Report:")
print(
    classification_report(
        y_true,
        y_pred,
        target_names=["Fake", "Real"]
    )
)
