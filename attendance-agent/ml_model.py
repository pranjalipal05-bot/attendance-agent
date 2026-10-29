"""
Epic 4 - AI implementation: a small machine-learning engagement classifier.

A RandomForest predicts the probability that a learner is "engaged" from four
features: attention, movement, audio activity and whether audio exists.

Training data
-------------
* If `data/labeled_features.csv` exists (columns: attention, movement, audio,
  has_audio, engaged) the model is trained on YOUR labelled clips.
* Otherwise it is bootstrapped on synthetic data generated from a noisy
  engagement rule, so the project works out of the box. Replace it with real
  labels for better accuracy (see README).

If scikit-learn or the saved model is unavailable, the app silently falls back
to the rule-based score in engagement.py.
"""
import os

import numpy as np

MODEL_PATH = os.path.join("models", "engagement_model.joblib")
LABELED_CSV = os.path.join("data", "labeled_features.csv")
FEATURES = ["attention", "movement", "audio", "has_audio"]


def _synthetic_data(n=4000, seed=42):
    rng = np.random.default_rng(seed)
    attention = rng.random(n)
    movement = rng.random(n)
    has_audio = (rng.random(n) > 0.15).astype(float)
    audio = rng.random(n) * has_audio
    score = 0.6 * attention + 0.2 * np.minimum(1, movement * 2) + 0.2 * np.minimum(1, audio * 1.5)
    score = np.where(has_audio == 1, score, 0.75 * attention + 0.25 * np.minimum(1, movement * 2))
    score += rng.normal(0, 0.08, n)                      # labelling noise
    y = (score > 0.5).astype(int)
    X = np.column_stack([attention, movement, audio, has_audio])
    return X, y


def train_model(save=True):
    """Train the classifier and (optionally) save it. Returns (model, accuracy)."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split

    if os.path.exists(LABELED_CSV):
        import pandas as pd
        df = pd.read_csv(LABELED_CSV)
        X, y = df[FEATURES].to_numpy(float), df["engaged"].to_numpy(int)
    else:
        X, y = _synthetic_data()

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=0)
    model = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=0)
    model.fit(X_tr, y_tr)
    acc = float(model.score(X_te, y_te))

    if save:
        import joblib
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        joblib.dump(model, MODEL_PATH)
    return model, acc


def load_model():
    """Load the saved model (training it first if needed). Returns None on failure."""
    try:
        import joblib
        if os.path.exists(MODEL_PATH):
            try:
                return joblib.load(MODEL_PATH)
            except Exception:
                pass                                   # e.g. scikit-learn version changed
        return train_model(save=True)[0]
    except Exception:
        return None


def predict_engagement(model, attention, movement, audio_ratio):
    """Return an engagement score from 0 to 100 (probability of 'engaged')."""
    has_audio = 0.0 if audio_ratio is None else 1.0
    x = np.array([[attention, movement, min(1.0, (audio_ratio or 0.0)), has_audio]])
    return float(100 * model.predict_proba(x)[0][1])
