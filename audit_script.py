import pandas as pd
import numpy as np
import json
import torch
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, confusion_matrix
import os

print("--- 1. Dataset Verification ---")
try:
    df = pd.read_csv("data/ev_battery_failure_dataset.csv")
    print(f"Row count: {len(df)}")
    print(f"Column count: {df.shape[1]}")
    print(f"Target column: battery_failure in cols? {'battery_failure' in df.columns}")
    print(f"Class distribution:\n{df['battery_failure'].value_counts(normalize=True)}")
    print(f"Missing values: {df.isnull().sum().sum()}")
    print(f"Duplicate count: {df.duplicated().sum()}")
except Exception as e:
    print(f"Dataset error: {e}")

print("\n--- 2. Training Verification ---")
models_to_check = [
    ("PyTorch MLP", "models/failure_predictor/mlp.pt"),
    ("XGBoost Baseline", "models/xgboost_failure_model.joblib"),
    ("Autoencoder", "models/autoencoder/ae.pt"),
    ("VAE", "models/vae/vae.pt"),
    ("cGAN Generator", "models/cgan/generator.pt"),
    ("cGAN Discriminator", "models/cgan/discriminator.pt")
]
for name, path in models_to_check:
    print(f"{name}: {'Found' if os.path.exists(path) else 'Missing'}")

print("\n--- 3. Primary Classifier Evaluation ---")
try:
    from src.prediction_engine import load_mlp_model
    from src.preprocessing import FEATURE_COLUMNS, TARGET_COLUMN
    
    mlp = load_mlp_model()
    # evaluate on test set
    np.random.seed(42)
    test_idx = np.random.choice(len(df), size=int(len(df)*0.2), replace=False)
    test_df = df.iloc[test_idx]
    
    X_test = test_df[FEATURE_COLUMNS]
    y_test = test_df[TARGET_COLUMN]
    
    # Preprocess
    X_test_imp = mlp.imputer.transform(X_test)
    X_test_sc = mlp.scaler_x.transform(X_test_imp)
    
    with torch.no_grad():
        X_ten = torch.FloatTensor(X_test_sc)
        logits = mlp.model(X_ten).squeeze()
        probs = torch.sigmoid(logits).numpy()
        preds = (probs > 0.5).astype(int)
        
    print(f"Accuracy: {accuracy_score(y_test, preds):.4f}")
    print(f"Precision: {precision_score(y_test, preds):.4f}")
    print(f"Recall: {recall_score(y_test, preds):.4f}")
    print(f"F1: {f1_score(y_test, preds):.4f}")
    print(f"ROC-AUC: {roc_auc_score(y_test, probs):.4f}")
    print(f"PR-AUC: {average_precision_score(y_test, probs):.4f}")
    print(f"Confusion Matrix:\n{confusion_matrix(y_test, preds)}")
    print(f"Test Set Class Distribution: {y_test.value_counts(normalize=True).to_dict()}")
except Exception as e:
    print(f"Evaluation error: {e}")

print("\n--- 4. Leakage Audit ---")
try:
    from src.preprocessing import FEATURE_COLUMNS
    print(f"Used features: {FEATURE_COLUMNS}")
except Exception as e:
    print(f"Leakage check error: {e}")

print("\n--- 5. SHAP / LIME Verification ---")
try:
    from src.shap_explainer import create_explainer, explain_local
    from src.lime_explainer import create_lime_explainer
    
    bg = test_df[FEATURE_COLUMNS].head(100)
    shap_explainer = create_explainer(mlp, bg)
    test_feat = test_df[FEATURE_COLUMNS].iloc[0].to_dict()
    
    shap_res = explain_local(shap_explainer, mlp, test_feat)
    print("SHAP executed successfully.")
    print(f"SHAP explanation: {shap_res['explanation_text']}")
    
    lime_explainer = create_lime_explainer(mlp)
    lime_res = lime_explainer.explain_prediction(test_feat, num_features=5)
    print("LIME executed successfully.")
except Exception as e:
    print(f"SHAP/LIME error: {e}")

print("\n--- 6. Autoencoder Verification ---")
try:
    from src.anomaly_detection import detect_anomaly, load_autoencoder
    ae_artifacts = load_autoencoder("models/autoencoder")
    ae_model, ae_scaler, ae_imputer, ae_threshold = ae_artifacts
    print(f"Threshold: {ae_threshold}")
    anom_res = detect_anomaly(ae_model, ae_scaler, ae_imputer, ae_threshold, test_feat)
    print(f"Anomaly score: {anom_res['score']}, Label: {anom_res['label']}")
except Exception as e:
    print(f"AE error: {e}")

print("\n--- 7. Git status ---")
