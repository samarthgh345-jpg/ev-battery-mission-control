"""
SHAP Explainable AI Module
===========================
Global and local SHAP explanations for the PyTorch Failure Classifier.
"""

import numpy as np
import pandas as pd
import torch
import shap
from src.preprocessing import FEATURE_COLUMNS, get_feature_display_names
from src.prediction_engine import MLPPredictor

def create_explainer(model: MLPPredictor, X_background: pd.DataFrame):
    """
    Create a SHAP DeepExplainer for the PyTorch model.
    X_background should be a small sample (e.g. 100 rows) of the training data.
    """
    X_bg_imputed = model.imputer.transform(X_background[FEATURE_COLUMNS].values)
    X_bg_scaled = model.scaler_x.transform(X_bg_imputed)
    bg_tensor = torch.FloatTensor(X_bg_scaled)
    
    # We use DeepExplainer for PyTorch models
    explainer = shap.DeepExplainer(model.model, bg_tensor)
    return explainer

def explain_global(explainer, model: MLPPredictor, X_sample: pd.DataFrame) -> dict:
    """Compute global SHAP feature importance."""
    X_imputed = model.imputer.transform(X_sample[FEATURE_COLUMNS].values)
    X_scaled = model.scaler_x.transform(X_imputed)
    sample_tensor = torch.FloatTensor(X_scaled)
    
    # DeepExplainer returns a list for outputs, but we only have 1 output (logit)
    shap_values = explainer.shap_values(sample_tensor)
    if isinstance(shap_values, list):
        shap_values = shap_values[0]
        
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    display_names = get_feature_display_names()

    return {
        "feature_names": FEATURE_COLUMNS,
        "display_names": [display_names.get(f, f) for f in FEATURE_COLUMNS],
        "importance": mean_abs_shap.tolist(),
        "shap_values": shap_values,
    }

def explain_local(explainer, model: MLPPredictor, features: dict) -> dict:
    """Compute SHAP contributions for a single prediction."""
    df = pd.DataFrame([features])[FEATURE_COLUMNS]
    X_imputed = model.imputer.transform(df.values)
    X_scaled = model.scaler_x.transform(X_imputed)
    obs_tensor = torch.FloatTensor(X_scaled)
    
    shap_values = explainer.shap_values(obs_tensor)
    if isinstance(shap_values, list):
        shap_values = shap_values[0]
    shap_values = shap_values[0] # Take first (and only) observation
    
    # Expected value from explainer
    base_value = float(explainer.expected_value) if hasattr(explainer, "expected_value") and explainer.expected_value is not None else 0.0
    if isinstance(base_value, np.ndarray):
        base_value = float(base_value[0])
        
    predicted_logit = base_value + float(np.sum(shap_values))
    predicted_prob = 1.0 / (1.0 + np.exp(-predicted_logit))
    
    display_names = get_feature_display_names()

    contributions = []
    for i, feat in enumerate(FEATURE_COLUMNS):
        contributions.append({
            "feature": feat,
            "display_name": display_names.get(feat, feat),
            "value": round(float(features.get(feat, 0)), 4),
            "shap_value": round(float(shap_values[i]), 4),
        })

    # Sort by absolute SHAP value
    contributions.sort(key=lambda x: abs(x["shap_value"]), reverse=True)

    # Generate natural language explanation
    explanation_text = _generate_explanation(contributions)

    return {
        "base_value_logit": round(base_value, 2),
        "predicted_prob": round(predicted_prob, 4),
        "contributions": contributions,
        "explanation_text": explanation_text,
    }

def _generate_explanation(contributions: list) -> str:
    """Generate a natural language explanation from SHAP contributions."""
    top_positive = [c for c in contributions if c["shap_value"] > 0.5][:3]
    top_negative = [c for c in contributions if c["shap_value"] < -0.5][:3]

    parts = []

    if top_positive:
        names = ", ".join([c["display_name"] for c in top_positive])
        parts.append(
            f"The strongest factors INCREASING predicted failure risk are "
            f"**{names}**"
        )

    if top_negative:
        names = ", ".join([c["display_name"] for c in top_negative])
        parts.append(
            f"while **{names}** act{'s' if len(top_negative) == 1 else ''} "
            f"to DECREASE the failure risk"
        )

    if parts:
        return ", ".join(parts) + "."
    return "No significant single-feature drivers identified for this prediction."
