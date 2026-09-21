"""
LIME Explainable AI Module
==========================
Local LIME explanations for the PyTorch MLP failure classifier.
"""

import numpy as np
import pandas as pd
import lime
import lime.lime_tabular
from src.preprocessing import FEATURE_COLUMNS, get_feature_display_names
from src.prediction_engine import MLPPredictor, predict_batch

class LimeThermalExplainer:
    def __init__(self, model: MLPPredictor, background_data: pd.DataFrame):
        """
        Initialize the LIME Tabular Explainer.
        model: MLPPredictor instance.
        background_data: A DataFrame of background training data (e.g., 100-500 samples)
        """
        self.model = model
        self.feature_names = FEATURE_COLUMNS
        
        # Add tiny noise to avoid zero variance (fixes scipy truncnorm error)
        bg_values = background_data[FEATURE_COLUMNS].values.astype(float)
        bg_values += np.random.normal(0, 1e-6, bg_values.shape)

        self.explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=bg_values,
            feature_names=self.feature_names,
            mode="classification", # We are predicting failure probability
            class_names=["Normal", "Failure"],
            random_state=42
        )
        
    def _predict_fn(self, X: np.ndarray) -> np.ndarray:
        """Prediction wrapper for LIME."""
        df = pd.DataFrame(X, columns=self.feature_names)
        probs = predict_batch(self.model, df)
        # LIME classification needs probability for both classes [P(0), P(1)]
        return np.vstack((1 - probs, probs)).T
        
    def explain_prediction(self, features: dict, num_features: int = 5) -> dict:
        """
        Generate a LIME local explanation for a single prediction.
        """
        x_obs = pd.DataFrame([features])[self.feature_names].values[0]
        
        # Generate explanation
        exp = self.explainer.explain_instance(
            data_row=x_obs,
            predict_fn=self._predict_fn,
            num_features=num_features
        )
        
        # Get actual predicted probability for class 1
        predicted_prob = self._predict_fn(np.array([x_obs]))[0][1]
        
        display_names = get_feature_display_names()
        contributions = []
        
        for feat_condition, weight in exp.as_list():
            base_feat = None
            for f in self.feature_names:
                if f in feat_condition:
                    base_feat = f
                    break
                    
            disp_name = display_names.get(base_feat, base_feat) if base_feat else feat_condition
            
            contributions.append({
                "feature": base_feat or feat_condition,
                "condition": feat_condition,
                "display_name": disp_name,
                "weight": round(float(weight), 4),
                "direction": "increases risk" if weight > 0 else "decreases risk"
            })
            
        model_type = "PyTorch MLP Classifier"
            
        return {
            "predicted_prob": round(float(predicted_prob), 4),
            "contributions": contributions,
            "explanation_text": f"LIME local explanation using {model_type} model.",
            "model_type": model_type
        }

def create_lime_explainer(model, dataset_path: str = "data/ev_battery_failure_dataset.csv") -> LimeThermalExplainer:
    """Helper to initialize LIME explainer with a sample of the dataset."""
    try:
        df = pd.read_csv(dataset_path)
        background_data = df.sample(n=500, random_state=42)
        return LimeThermalExplainer(model, background_data)
    except Exception as e:
        print(f"Failed to initialize LIME explainer: {e}")
        return None
