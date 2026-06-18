import torch
import sys
sys.path.append("/home/minsukc/MRI2CT")
from anatomix.model.network import Unet

m1 = Unet(3, 1, 16, 5, 20, norm="instance", interp="trilinear", pooling="Avg")
print(f"Old config: {sum(p.numel() for p in m1.parameters())}")

m2 = Unet(3, 1, 36, 4, 36, norm="instance", interp="trilinear", pooling="Avg")
print(f"New config: {sum(p.numel() for p in m2.parameters())}")
