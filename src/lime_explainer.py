"""
LIME Explainable AI Module
==========================
Local LIME explanations for the XGBoost or PyTorch MLP temperature models.
"""

import numpy as np
import pandas as pd
import lime
import lime.lime_tabular
from src.preprocessing import FEATURE_COLUMNS, get_feature_display_names
from src.prediction_engine import MLPPredictor, predict_batch

class LimeThermalExplainer:
    def __init__(self, model, background_data: pd.DataFrame):
        """
        Initialize the LIME Tabular Explainer.
        model: XGBoost or MLPPredictor instance.
        background_data: A DataFrame of background training data (e.g., 100-500 samples)
        """
        self.model = model
        self.feature_names = FEATURE_COLUMNS
        
        # LIME needs a numpy 2d array of training data
        self.explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=background_data[FEATURE_COLUMNS].values,
            feature_names=self.feature_names,
            mode="regression",
            random_state=42
        )
        
    def _predict_fn(self, X: np.ndarray) -> np.ndarray:
        """Prediction wrapper for LIME."""
        df = pd.DataFrame(X, columns=self.feature_names)
        return predict_batch(self.model, df)
        
    def explain_prediction(self, features: dict, num_features: int = 5) -> dict:
        """
        Generate a LIME local explanation for a single prediction.
        
        Returns:
            {
                "predicted_value": float,
                "contributions": list[dict],
                "explanation_text": str,
                "model_type": str
            }
        """
        # Prepare the single observation as a numpy array
        x_obs = pd.DataFrame([features])[self.feature_names].values[0]
        
        # Generate explanation
        exp = self.explainer.explain_instance(
            data_row=x_obs,
            predict_fn=self._predict_fn,
            num_features=num_features
        )
        
        # Get actual predicted value
        predicted_value = self._predict_fn(np.array([x_obs]))[0]
        
        display_names = get_feature_display_names()
        contributions = []
        
        # LIME returns list of tuples: (feature_condition_string, weight)
        # e.g., ('battery_current_A > 10.5', 2.3)
        # We parse this to extract the base feature name if possible, or just pass the string.
        for feat_condition, weight in exp.as_list():
            # Find the actual feature name inside the condition string
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
            
        model_type = "mlp" if isinstance(self.model, MLPPredictor) else "xgboost"
            
        return {
            "predicted_value": round(float(predicted_value), 2),
            "contributions": contributions,
            "explanation_text": f"LIME local explanation using {model_type.upper()} model.",
            "model_type": model_type
        }

def create_lime_explainer(model, dataset_path: str = "data/btms_dataset.csv") -> LimeThermalExplainer:
    """Helper to initialize LIME explainer with a sample of the dataset."""
    try:
        df = pd.read_csv(dataset_path)
        # Use a small sample to initialize LIME for speed
        background_data = df.sample(n=500, random_state=42)
        return LimeThermalExplainer(model, background_data)
    except Exception as e:
        print(f"Failed to initialize LIME explainer: {e}")
        return None
