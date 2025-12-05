import torch
import torch.nn as nn
import numpy as np

class FourierFeatureMapping(nn.Module):
    """
    NeurIPS 2020: "Fourier Features Let Networks Learn High Frequency Functions..."
    Implementation from authors:
        x_proj = (2 * pi * x) @ B.T
        [sin(x_proj), cos(x_proj)]
    """
    def __init__(self, input_dim=3, mapping_size=128, scale=10.0):
        super().__init__()
        self.mapping_size = mapping_size
        # B is drawn from N(0, sigma^2)
        # Shape [mapping_size, input_dim] to allow matmul: x @ B.T
        self.register_buffer('B', torch.randn((mapping_size, input_dim)) * scale)
        
    def forward(self, x):
        # x: [N, 3]
        # B: [M, 3]
        # x_proj: [N, M]
        x_proj = (2 * np.pi * x) @ self.B.T
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class MLPTranslator(nn.Module):
    def __init__(self, in_feat_dim=16, use_fourier=True, fourier_scale=10.0, hidden=256, out_dim=1, dropout=0.0):
        super().__init__()
        self.use_fourier = use_fourier
        
        if self.use_fourier:
            # mapping_size=128 result in 128*2 = 256 features
            self.rff = FourierFeatureMapping(input_dim=3, mapping_size=128, scale=fourier_scale)
            combined_dim = in_feat_dim + 256
        else:
            # Only use Anatomix features
            combined_dim = in_feat_dim
        
        print(f"Combined feature dimension: {combined_dim}")
        
        self.net = nn.Sequential(
            nn.Linear(combined_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),

            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            
            nn.Linear(hidden, out_dim),
            nn.Sigmoid(), # Output 0-1
        )

    def forward(self, feats, coords):
        if self.use_fourier:
            pos_enc = self.rff(coords)
            x = torch.cat([feats, pos_enc], dim=1)
        else:
            x = feats
            
        return self.net(x)

class CNNTranslator(nn.Module):
    def __init__(self, in_channels=16, hidden_channels=32, depth=3, final_activation="relu_clamp", dropout=0.0):
        """
        Args:
            in_channels (int): Input feature dimension.
            hidden_channels (int): Number of filters in hidden layers.
            depth (int): Total number of Conv3d layers.
            final_activation (str): "sigmoid", "relu_clamp", or "none".
        """
        super().__init__()
        self.final_activation = final_activation
        
        layers = []
        
        # --- 1. First Layer (Input -> Hidden) ---
        layers.append(nn.Conv3d(in_channels, hidden_channels, kernel_size=3, padding=1))
        layers.append(nn.ReLU(inplace=True))
        if dropout > 0:
            layers.append(nn.Dropout3d(p=dropout))
        
        # --- 2. Middle Layers (Hidden -> Hidden) ---
        # We add (depth - 2) middle layers because first and last are handled separately
        for _ in range(depth - 2):
            layers.append(nn.Conv3d(hidden_channels, hidden_channels, kernel_size=3, padding=1))
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0:
                layers.append(nn.Dropout3d(p=dropout))
            
        # --- 3. Last Layer (Hidden -> 1) ---
        layers.append(nn.Conv3d(hidden_channels, 1, kernel_size=3, padding=1))
        
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        x = self.net(x)
        
        if self.final_activation == "sigmoid":
            return torch.sigmoid(x)
        elif self.final_activation == "relu_clamp":
            return torch.clamp(torch.relu(x), 0, 1)
        elif self.final_activation == "none":
            return x
        else:
            raise ValueError(f"Unknown activation: {self.final_activation}")