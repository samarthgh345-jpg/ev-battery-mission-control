import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
import joblib

from models.mlp_predictor import ThermalMLP

def train():
    print("Loading dataset...")
    df = pd.read_csv("data/btms_dataset.csv")
    
    features = [
        "battery_current_A", "battery_voltage_V", "state_of_charge_percent",
        "ambient_temperature_C", "battery_temperature_C", "coolant_inlet_temperature_C",
        "coolant_flow_rate_kg_s", "coolant_pressure_Pa", "nanoparticle_concentration_percent",
        "reynolds_number", "heat_transfer_coefficient_W_m2K", "microchannel_width_mm",
        "microchannel_height_mm", "discharge_rate_C"
    ]
    target = "max_battery_temperature_C"
    
    X = df[features].values
    y = df[target].values
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)
    
    scaler_y = StandardScaler()
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
    y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).flatten()
    
    train_dataset = TensorDataset(torch.FloatTensor(X_train_scaled), torch.FloatTensor(y_train_scaled))
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    
    model = ThermalMLP(input_dim=len(features))
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print("Training MLP Predictor...")
    epochs = 20
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            preds = model(batch_X).squeeze()
            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (epoch+1) % 5 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}")
            
    os.makedirs("models/thermal_predictor", exist_ok=True)
    torch.save(model.state_dict(), "models/thermal_predictor/mlp.pt")
    joblib.dump(scaler_X, "models/thermal_predictor/scaler_x.pkl")
    joblib.dump(scaler_y, "models/thermal_predictor/scaler_y.pkl")
    print("Saved MLP Predictor models/thermal_predictor/mlp.pt")

if __name__ == "__main__":
    train()
