# ensemble.py
import numpy as np
from app_models import predict_dnn_proba, predict_svm_proba, decode_label

def soft_vote(X_vec, weights=(0.5, 0.5)):
    """
    X_vec: 1D or 2D array of shape (n_features,) or (1, n_features)
    Returns: probs_final, pred_index, pred_label
    """
    w_dnn, w_svm = weights

    # Ensure 2D for TF
    X_input = X_vec.reshape(1, -1) if X_vec.ndim == 1 else X_vec

    probs_dnn = predict_dnn_proba(X_input)
    probs_svm = predict_svm_proba(X_input)

    # normalize weights
    s = w_dnn + w_svm
    w_dnn /= s
    w_svm /= s

    probs_final = w_dnn * probs_dnn + w_svm * probs_svm
    pred_idx = int(np.argmax(probs_final))
    pred_label = decode_label(pred_idx)
    return probs_final, pred_idx, pred_label
