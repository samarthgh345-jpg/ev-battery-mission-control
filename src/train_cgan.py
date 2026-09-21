import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import os
import joblib

from src.models.cgan import Generator, Discriminator
from src.preprocessing import load_dataset, FEATURE_COLUMNS, TARGET_COLUMN

def train():
    print("Loading dataset for cGAN...")
    df = load_dataset("data/ev_battery_failure_dataset.csv")
    
    # Filter out NaNs in target just in case, but normally it has none
    df = df.dropna(subset=[TARGET_COLUMN])
    
    X = df[FEATURE_COLUMNS].values
    y = df[TARGET_COLUMN].values.astype(int)
    
    imputer = SimpleImputer(strategy='median')
    scaler = StandardScaler()
    
    X_imputed = imputer.fit_transform(X)
    X_scaled = scaler.fit_transform(X_imputed)
    
    dataset = TensorDataset(torch.FloatTensor(X_scaled), torch.LongTensor(y))
    # Larger batch size
    dataloader = DataLoader(dataset, batch_size=256, shuffle=True)
    
    noise_dim = 16
    num_classes = 2 # 0: NORMAL, 1: FAILURE
    feature_dim = len(FEATURE_COLUMNS)
    
    generator = Generator(noise_dim, num_classes, feature_dim)
    discriminator = Discriminator(num_classes, feature_dim)
    
    criterion = nn.BCELoss()
    opt_g = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    opt_d = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    
    epochs = 10 # Reduced epochs for rapid prototyping
    print("Training Conditional GAN...")
    for epoch in range(epochs):
        g_loss_total = 0
        d_loss_total = 0
        
        for batch_idx, (real_data, labels) in enumerate(dataloader):
            batch_size = real_data.size(0)
            
            # Train Discriminator
            opt_d.zero_grad()
            real_labels_d = torch.ones(batch_size, 1)
            fake_labels_d = torch.zeros(batch_size, 1)
            
            # Real pass
            out_real = discriminator(real_data, labels)
            d_loss_real = criterion(out_real, real_labels_d)
            
            # Fake pass
            noise = torch.randn(batch_size, noise_dim)
            fake_data = generator(noise, labels)
            out_fake = discriminator(fake_data.detach(), labels)
            d_loss_fake = criterion(out_fake, fake_labels_d)
            
            d_loss = (d_loss_real + d_loss_fake) / 2
            d_loss.backward()
            opt_d.step()
            
            # Train Generator
            opt_g.zero_grad()
            out_fake_g = discriminator(fake_data, labels)
            g_loss = criterion(out_fake_g, real_labels_d) # Try to fool discriminator
            g_loss.backward()
            opt_g.step()
            
            d_loss_total += d_loss.item()
            g_loss_total += g_loss.item()
            
        print(f"Epoch {epoch+1}/{epochs} | D Loss: {d_loss_total/len(dataloader):.4f} | G Loss: {g_loss_total/len(dataloader):.4f}")

    os.makedirs("models/cgan", exist_ok=True)
    torch.save(generator.state_dict(), "models/cgan/generator.pt")
    torch.save(discriminator.state_dict(), "models/cgan/discriminator.pt")
    joblib.dump(imputer, "models/cgan/imputer.pkl")
    joblib.dump(scaler, "models/cgan/scaler.pkl")
    print("Saved cGAN models/cgan/generator.pt")

if __name__ == "__main__":
    train()
