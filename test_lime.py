import json
from src.lime_explainer import create_lime_explainer
from src.prediction_engine import load_model, get_default_features

print("Loading XGBoost model...")
xgb_model = load_model("models/xgboost_temperature_model.joblib")

print("Initializing LIME explainer...")
lime_explainer = create_lime_explainer(xgb_model)

print("Getting default features...")
features = get_default_features()

print("Generating LIME explanation...")
exp = lime_explainer.explain_prediction(features)

print("LIME Explanation Result:")
print(json.dumps(exp, indent=2))
print("\nTest passed!")
