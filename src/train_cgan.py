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

from models.cgan import Generator, Discriminator

def train():
    print("Loading dataset for cGAN...")
    df = pd.read_csv("data/btms_dataset.csv")
    
    features = [
        "battery_current_A", "battery_voltage_V", "state_of_charge_percent",
        "ambient_temperature_C", "battery_temperature_C", "coolant_inlet_temperature_C",
        "coolant_flow_rate_kg_s", "coolant_pressure_Pa", "nanoparticle_concentration_percent",
        "reynolds_number", "heat_transfer_coefficient_W_m2K", "microchannel_width_mm",
        "microchannel_height_mm", "discharge_rate_C"
    ]
    
    # Map thermal_risk_label to ints
    label_map = {"NORMAL": 0, "CAUTION": 1, "HIGH": 2, "CRITICAL": 3}
    df["label"] = df["thermal_risk_label"].map(label_map)
    df = df.dropna(subset=["label"])
    
    X = df[features].values
    y = df["label"].values.astype(int)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    dataset = TensorDataset(torch.FloatTensor(X_scaled), torch.LongTensor(y))
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    noise_dim = 16
    num_classes = 4
    feature_dim = len(features)
    
    generator = Generator(noise_dim, num_classes, feature_dim)
    discriminator = Discriminator(num_classes, feature_dim)
    
    criterion = nn.BCELoss()
    opt_g = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    opt_d = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    
    epochs = 20
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
            
        if (epoch+1) % 5 == 0:
            print(f"Epoch {epoch+1}/{epochs} | D Loss: {d_loss_total/len(dataloader):.4f} | G Loss: {g_loss_total/len(dataloader):.4f}")

    os.makedirs("models/cgan", exist_ok=True)
    torch.save(generator.state_dict(), "models/cgan/generator.pt")
    torch.save(discriminator.state_dict(), "models/cgan/discriminator.pt")
    joblib.dump(scaler, "models/cgan/scaler.pkl")
    print("Saved cGAN models/cgan/generator.pt")

if __name__ == "__main__":
    train()
