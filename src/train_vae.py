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

from models.vae import VAE

def loss_function(recon_x, x, mu, logvar):
    BCE = nn.functional.mse_loss(recon_x, x, reduction='sum')
    # KL divergence
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD, BCE, KLD

def train():
    print("Loading dataset for VAE...")
    df = pd.read_csv("data/btms_dataset.csv")
    
    features = [
        "battery_current_A", "battery_voltage_V", "state_of_charge_percent",
        "ambient_temperature_C", "battery_temperature_C", "coolant_inlet_temperature_C",
        "coolant_flow_rate_kg_s", "coolant_pressure_Pa", "nanoparticle_concentration_percent",
        "reynolds_number", "heat_transfer_coefficient_W_m2K", "microchannel_width_mm",
        "microchannel_height_mm", "discharge_rate_C"
    ]
    
    X = df[features].values
    X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    train_dataset = TensorDataset(torch.FloatTensor(X_train_scaled))
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    
    model = VAE(input_dim=len(features), latent_dim=4)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print("Training VAE...")
    epochs = 20
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch_idx, (data,) in enumerate(train_loader):
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(data)
            loss, bce, kld = loss_function(recon_batch, data, mu, logvar)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
            
        if (epoch+1) % 5 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {train_loss / len(train_loader.dataset):.4f}")
            
    os.makedirs("models/vae", exist_ok=True)
    torch.save(model.state_dict(), "models/vae/vae.pt")
    joblib.dump(scaler, "models/vae/scaler.pkl")
    print("Saved VAE models/vae/vae.pt")

if __name__ == "__main__":
    train()
