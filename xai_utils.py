# xai_utils.py
import numpy as np
import shap
from ensemble_utils import soft_vote
from app_models import predict_dnn_proba, predict_svm_proba
import joblib, os

# helper wrapper used by SHAP/LIME expecting array-like X (n_samples, n_features)
def predict_proba_ensemble(X_array, weights=(0.5,0.5)):
    """
    X_array: shape (n_samples, n_features)
    Returns: (n_samples, n_classes)
    """
    # If 1D array passed, reshape to (1, n_features)
    if X_array.ndim == 1:
        X_array = X_array.reshape(1, -1)

    out = []
    for i in range(X_array.shape[0]):
        probs, _, _ = soft_vote(X_array[i, :], weights=weights)
        out.append(probs)
    return np.array(out)

def shap_explain_instance(X_background, instance, weights=(0.5,0.5), nsamples=100):
    """
    X_background: numpy array of background samples (e.g. small subset of training)
    instance: 1D numpy array
    returns: (feature_names, shap_values_for_instance) — arrays
    """
    # Use KernelExplainer because ensemble is not a single differentiable model
    # KernelExplainer can be slow — use small background
    
    explainer = shap.KernelExplainer(lambda x: predict_proba_ensemble(x, weights=weights), X_background)
    
    # compute shap values for the instance (returns list of arrays per class)
    
    shap_vals = explainer.shap_values(instance.reshape(1, -1), nsamples=nsamples)
    
    # shap_vals is list length n_classes; we can pick predicted class index's shap vector
    
    return shap_vals  # caller decides which class to inspect

"""LIME explainer - currently not used, may use in future"""

# def lime_explain_instance(X_train, feature_names, class_names, instance, num_samples=500):
#     explainer = LimeTabularExplainer(X_train, feature_names=feature_names, class_names=class_names, mode="classification")
#     exp = explainer.explain_instance(instance, predict_proba_ensemble, num_features=10, num_samples=num_samples)
#     return exp  # Lime explanation object; you can call exp.as_list() etc.