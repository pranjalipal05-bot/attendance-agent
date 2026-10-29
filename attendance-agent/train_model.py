"""Train (or retrain) the engagement model:  python train_model.py"""
from ml_model import train_model, MODEL_PATH, LABELED_CSV
import os

model, acc = train_model(save=True)
source = "your labelled data" if os.path.exists(LABELED_CSV) else "synthetic bootstrap data"
print(f"Trained on {source}. Hold-out accuracy: {acc:.1%}")
print(f"Saved to {MODEL_PATH}")
