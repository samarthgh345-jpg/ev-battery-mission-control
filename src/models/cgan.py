import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, noise_dim, num_classes, feature_dim):
        super(Generator, self).__init__()
        self.label_emb = nn.Embedding(num_classes, num_classes)
        
        self.model = nn.Sequential(
            nn.Linear(noise_dim + num_classes, 64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(64, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, feature_dim)
        )
        
    def forward(self, noise, labels):
        c = self.label_emb(labels)
        x = torch.cat([noise, c], dim=1)
        return self.model(x)

class Discriminator(nn.Module):
    def __init__(self, num_classes, feature_dim):
        super(Discriminator, self).__init__()
        self.label_emb = nn.Embedding(num_classes, num_classes)
        
        self.model = nn.Sequential(
            nn.Linear(feature_dim + num_classes, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
    def forward(self, features, labels):
        c = self.label_emb(labels)
        x = torch.cat([features, c], dim=1)
        return self.model(x)
