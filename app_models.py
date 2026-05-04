# app_models.py
import os
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model

MODEL_DIR = "models"
DNN_PATH = os.path.join(r"C:\Users\yashd\OneDrive\Desktop\Doctor AI\models", "dnn.h5")
SVM_PATH = os.path.join(r"C:\Users\yashd\OneDrive\Desktop\Doctor AI\models", "svc.pkl")      
LE_PATH  = os.path.join(r"C:\Users\yashd\OneDrive\Desktop\Doctor AI\models", "label_encoder.joblib")

# Load label encoder
if os.path.exists(LE_PATH):
    le = joblib.load(LE_PATH)
else:
    le = None

# Load SVM (sklearn)
if os.path.exists(SVM_PATH):
    import pickle
    with open(SVM_PATH, "rb") as f:
        svm = pickle.load(f)
else:
    svm = None

# Load DNN
if os.path.exists(DNN_PATH):
    dnn = load_model(DNN_PATH)
else:
    dnn = None

# -------------------------------
# Prediction functions
# -------------------------------
def predict_svm_proba(X_vec):
    """Predict probabilities with SVM."""
    if svm is None:
        raise RuntimeError("SVM model not loaded")
    X_vec = np.array(X_vec, dtype=float).reshape(1, -1)   # ensure (1, n_features)
    probs = svm.predict_proba(X_vec)[0]      # shape (n_classes,)
    return probs

def predict_dnn_proba(X_vec):
    """Predict probabilities with DNN."""
    if dnn is None:
        raise RuntimeError("DNN model not loaded")
    X_vec = np.array(X_vec, dtype=np.float32).reshape(1, -1)   # ensure (1, n_features)
    probs = dnn.predict(X_vec, verbose=0)[0] # shape (n_classes,)
    return probs

# -------------------------------
# Label decoding
# -------------------------------
def decode_label(code):
    """Map integer class index -> label string using label encoder or svm classes_"""
    if le is not None:
        return le.inverse_transform([code])[0]
    if svm is not None and hasattr(svm, "classes_"):
        return svm.classes_[code]
    return str(code)
