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

from src.models.vae import VAE
from src.preprocessing import load_dataset, FEATURE_COLUMNS

def loss_function(recon_x, x, mu, logvar):
    BCE = nn.functional.mse_loss(recon_x, x, reduction='sum')
    # KL divergence
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD, BCE, KLD

def train():
    print("Loading dataset for VAE...")
    df = load_dataset("data/ev_battery_failure_dataset.csv")
    
    X = df[FEATURE_COLUMNS]
    X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)
    
    imputer = SimpleImputer(strategy='median')
    scaler = StandardScaler()
    
    X_train_imputed = imputer.fit_transform(X_train)
    X_train_scaled = scaler.fit_transform(X_train_imputed)
    
    train_dataset = TensorDataset(torch.FloatTensor(X_train_scaled))
    # Use larger batch size since dataset is 200k
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    
    model = VAE(input_dim=len(FEATURE_COLUMNS), latent_dim=4)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print("Training VAE...")
    epochs = 10  # Reduced epochs for large dataset prototyping
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
            
        print(f"Epoch {epoch+1}/{epochs}, Loss: {train_loss / len(train_loader.dataset):.4f}")
            
    os.makedirs("models/vae", exist_ok=True)
    torch.save(model.state_dict(), "models/vae/vae.pt")
    joblib.dump(imputer, "models/vae/imputer.pkl")
    joblib.dump(scaler, "models/vae/scaler.pkl")
    print("Saved VAE models/vae/vae.pt")

if __name__ == "__main__":
    train()
