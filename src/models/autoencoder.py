import torch
import torch.nn as nn

class LightweightAutoencoder(nn.Module):
    """
    Lightweight Autoencoder for IoT/Edge anomaly detection.
    Compresses 38 features (e.g., KDDCup99) down to a bottleneck of 8.
    """
    def __init__(self, input_dim=38):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(True),
            nn.Linear(32, 16),
            nn.ReLU(True),
            nn.Linear(16, 8)
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(True),
            nn.Linear(16, 32),
            nn.ReLU(True),
            nn.Linear(32, input_dim)
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

    def get_reconstruction_error(self, x):
        self.eval()
        with torch.no_grad():
            reconstructed = self.forward(x)
            mse = torch.mean((x - reconstructed)**2, dim=1)
        return mse
