import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import os
import joblib

from src.models.autoencoder import ThermalAutoencoder
from src.preprocessing import load_dataset, FEATURE_COLUMNS, TARGET_COLUMN

def train():
    print("Loading dataset for Autoencoder...")
    df = load_dataset("data/ev_battery_failure_dataset.csv")
    
    # Train primarily on NORMAL data for anomaly detection
    normal_df = df[df[TARGET_COLUMN] == 0]
    if len(normal_df) < 100:
        print("Warning: Not enough NORMAL data. Using all data.")
        X = df[FEATURE_COLUMNS]
    else:
        X = normal_df[FEATURE_COLUMNS]
        
    X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)
    
    imputer = SimpleImputer(strategy='median')
    scaler = StandardScaler()
    
    X_train_imputed = imputer.fit_transform(X_train)
    X_train_scaled = scaler.fit_transform(X_train_imputed)
    
    X_test_imputed = imputer.transform(X_test)
    X_test_scaled = scaler.transform(X_test_imputed)
    
    train_dataset = TensorDataset(torch.FloatTensor(X_train_scaled), torch.FloatTensor(X_train_scaled))
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    
    model = ThermalAutoencoder(input_dim=len(FEATURE_COLUMNS))
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print("Training Autoencoder...")
    epochs = 20
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            preds = model(batch_X)
            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (epoch+1) % 5 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}")
            
    # Calculate threshold on validation set
    model.eval()
    with torch.no_grad():
        preds = model(torch.FloatTensor(X_test_scaled))
        mse = torch.mean((torch.FloatTensor(X_test_scaled) - preds)**2, dim=1).numpy()
        threshold = np.percentile(mse, 95) # 95th percentile
        
    os.makedirs("models/autoencoder", exist_ok=True)
    torch.save(model.state_dict(), "models/autoencoder/ae.pt")
    joblib.dump(imputer, "models/autoencoder/imputer.pkl")
    joblib.dump(scaler, "models/autoencoder/scaler.pkl")
    with open("models/autoencoder/threshold.txt", "w") as f:
        f.write(str(threshold))
    print(f"Saved Autoencoder models/autoencoder/ae.pt (Threshold: {threshold:.4f})")

if __name__ == "__main__":
    train()
