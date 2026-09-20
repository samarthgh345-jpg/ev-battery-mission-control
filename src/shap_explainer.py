"""
SHAP Explainable AI Module
===========================
Global and local SHAP explanations for the XGBoost temperature model.
All SHAP values are computed from the actual trained model — never hard-coded.
"""

import numpy as np
import pandas as pd
import shap
from src.preprocessing import FEATURE_COLUMNS, get_feature_display_names


def create_explainer(model, X_background: pd.DataFrame = None):
    """Create a SHAP TreeExplainer for the XGBoost model."""
    return shap.TreeExplainer(model)


def explain_global(explainer, X_sample: pd.DataFrame) -> dict:
    """
    Compute global SHAP feature importance.

    Returns:
        {
            "feature_names": list[str],
            "display_names": list[str],
            "importance": list[float],  # mean |SHAP value|
            "shap_values": np.ndarray   # full SHAP matrix
        }
    """
    X = X_sample[FEATURE_COLUMNS].copy()
    shap_values = explainer.shap_values(X)

    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    display_names = get_feature_display_names()

    return {
        "feature_names": FEATURE_COLUMNS,
        "display_names": [display_names.get(f, f) for f in FEATURE_COLUMNS],
        "importance": mean_abs_shap.tolist(),
        "shap_values": shap_values,
    }


def explain_local(explainer, features: dict) -> dict:
    """
    Compute SHAP contributions for a single prediction.

    Returns:
        {
            "base_value": float,
            "predicted_value": float,
            "contributions": list[dict],  # sorted by |value|
            "explanation_text": str
        }
    """
    df = pd.DataFrame([features])[FEATURE_COLUMNS]
    shap_values = explainer.shap_values(df)[0]
    base_value = float(explainer.expected_value)
    predicted_value = base_value + float(np.sum(shap_values))
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
        "base_value": round(base_value, 2),
        "predicted_value": round(predicted_value, 2),
        "contributions": contributions,
        "explanation_text": explanation_text,
    }


def _generate_explanation(contributions: list) -> str:
    """Generate a natural language explanation from SHAP contributions."""
    top_positive = [c for c in contributions if c["shap_value"] > 0.1][:3]
    top_negative = [c for c in contributions if c["shap_value"] < -0.1][:3]

    parts = []

    if top_positive:
        names = ", ".join([c["display_name"] for c in top_positive])
        parts.append(
            f"The strongest positive contributors to the predicted temperature are "
            f"**{names}**"
        )

    if top_negative:
        names = ", ".join([c["display_name"] for c in top_negative])
        parts.append(
            f"while **{names}** contribute{'s' if len(top_negative) == 1 else ''} "
            f"to lowering the predicted temperature"
        )

    if parts:
        return ", ".join(parts) + "."
    return "No significant feature contributions identified for this prediction."
