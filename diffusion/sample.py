import matplotlib.pyplot as plt

# torch
import torch
import torch.nn.functional as F

import numpy as np

# custom imports
from ddpm import Diffusion
from model import UNet
from util import show, set_seed, CLASS_LABELS
    
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
set_seed()

# Load model
ddpm_cFg = Diffusion(img_size=24, T=500, beta_start=1e-4, beta_end=0.02, device=device)

unet_ddpm_cFg = UNet(num_classes=10, device=device)
unet_ddpm_cFg.eval()
unet_ddpm_cFg.to(device)
unet_ddpm_cFg.load_state_dict(torch.load('weights/DDPM-cfg/model.pth', map_location=device))

# Sample
y = torch.tensor([0,1,2,3,4,5,6,7,8,9], device=device) 
y = F.one_hot(y, num_classes=10).float()
x_new = ddpm_cFg.p_sample_loop(unet_ddpm_cFg, 10, y=y)

fig, axes = plt.subplots(1, 10, figsize=(20, 2))
fig.suptitle("Generated instances for all digits", fontsize=16)
for ax, img, digit in zip(axes, x_new, range(10)):
    ax.imshow(img.squeeze().numpy(), cmap='gray')
    ax.set_title(f'Digit {digit}', fontsize=14)
    ax.axis('off')
plt.tight_layout()
plt.savefig("/home/mlut/probai/assets/sample.png")