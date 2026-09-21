import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, average_precision_score, 
                             confusion_matrix)
import os
import joblib
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.mlp_predictor import FailureClassifier
from src.preprocessing import load_dataset, get_features_and_target, FEATURE_COLUMNS, TARGET_COLUMN

def train():
    print("Loading dataset...")
    df = load_dataset("data/ev_battery_failure_dataset.csv")
    
    # Train / Val / Test Split
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]
    
    # 70/15/15 split
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.1765, random_state=42, stratify=y_temp) # 0.15/0.85 = 0.1765
    
    print(f"Train size: {len(X_train)}, Val size: {len(X_val)}, Test size: {len(X_test)}")
    print(f"Target distribution (Test): {y_test.value_counts().to_dict()}")

    # Preprocessing
    print("Fitting Imputer and Scaler on Training data only...")
    imputer = SimpleImputer(strategy='median')
    scaler = StandardScaler()
    
    X_train_imputed = imputer.fit_transform(X_train)
    X_train_scaled = scaler.fit_transform(X_train_imputed)
    
    X_val_scaled = scaler.transform(imputer.transform(X_val))
    X_test_scaled = scaler.transform(imputer.transform(X_test))
    
    # Calculate pos_weight for BCEWithLogitsLoss
    num_pos = y_train.sum()
    num_neg = len(y_train) - num_pos
    pos_weight = torch.tensor([num_neg / num_pos], dtype=torch.float32)
    
    train_dataset = TensorDataset(torch.FloatTensor(X_train_scaled), torch.FloatTensor(y_train.values))
    val_dataset = TensorDataset(torch.FloatTensor(X_val_scaled), torch.FloatTensor(y_val.values))
    
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1024, shuffle=False)
    
    model = FailureClassifier(input_dim=len(FEATURE_COLUMNS))
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print("Training PyTorch Failure Classifier...")
    epochs = 20
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            logits = model(batch_X).squeeze()
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                logits = model(batch_X).squeeze()
                loss = criterion(logits, batch_y)
                val_loss += loss.item()
                
        avg_train_loss = total_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            os.makedirs("models/failure_predictor", exist_ok=True)
            torch.save(model.state_dict(), "models/failure_predictor/mlp.pt")
            
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
            
    print("Evaluating best model on Test Set...")
    model.load_state_dict(torch.load("models/failure_predictor/mlp.pt"))
    model.eval()
    
    with torch.no_grad():
        test_logits = model(torch.FloatTensor(X_test_scaled)).squeeze()
        test_probs = torch.sigmoid(test_logits).numpy()
    
    # We can tune threshold on Validation set if we wanted, but default 0.5 is standard
    threshold = 0.5
    test_preds = (test_probs >= threshold).astype(int)
    y_test_true = y_test.values
    
    print("\n--- Test Set Metrics ---")
    print(f"Accuracy:        {accuracy_score(y_test_true, test_preds):.4f}")
    print(f"Precision:       {precision_score(y_test_true, test_preds):.4f}")
    print(f"Recall (Failure):{recall_score(y_test_true, test_preds):.4f}")
    print(f"F1 Score:        {f1_score(y_test_true, test_preds):.4f}")
    print(f"ROC-AUC:         {roc_auc_score(y_test_true, test_probs):.4f}")
    print(f"PR-AUC:          {average_precision_score(y_test_true, test_probs):.4f}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test_true, test_preds))
    
    joblib.dump(imputer, "models/failure_predictor/imputer.pkl")
    joblib.dump(scaler, "models/failure_predictor/scaler_x.pkl")
    print("\nSaved models/failure_predictor/mlp.pt along with imputer and scaler.")
    
if __name__ == "__main__":
    train()
