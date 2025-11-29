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
    def __init__(self, in_feat_dim=16, use_fourier=True, fourier_scale=10.0, hidden=256, out_dim=1):
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
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
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
    def __init__(self, in_channels=16):
        super().__init__()
        # self.net = nn.Sequential(
        #     nn.Conv3d(in_channels, 32, kernel_size=3, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv3d(32, 64, kernel_size=3, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv3d(64, 32, kernel_size=3, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv3d(32, 1, kernel_size=3, padding=1),
        #     # nn.Sigmoid()
        #     nn.ReLU(inplace=True) 
        # )
        
        self.net = nn.Sequential(
            nn.Conv3d(in_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 1, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.net(x)
        # return x
        return torch.clamp(x, 0, 1)