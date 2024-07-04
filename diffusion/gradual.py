import matplotlib.pyplot as plt
from ddpm import Diffusion
from model import UNet
from util import show, set_seed, CLASS_LABELS
import torch
import torch.nn.functional as F


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
timesteps_to_save = [1, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500]

ddpm_cFg = Diffusion(img_size=24, T=500, beta_start=1e-4, beta_end=0.02, diff_type='DDPM-cfg', device=device)

unet_ddpm_cFg = UNet(num_classes=10, device=device)
unet_ddpm_cFg.eval()
unet_ddpm_cFg.to(device)
unet_ddpm_cFg.load_state_dict(torch.load('weights/DDPM-cfg/model.pth', map_location=device))

# Generate images
batch_size = 1
choosen_digit = 3
y = F.one_hot(torch.tensor([choosen_digit] * batch_size), num_classes=10).float().to(ddpm_cFg.device)
final_image, intermediate_images = ddpm_cFg.p_sample_loop(unet_ddpm_cFg, batch_size, y=y, timesteps_to_save=timesteps_to_save)

# Plot the images
fig, axes = plt.subplots(1, len(timesteps_to_save), figsize=(20, 2))
fig.suptitle(f"Changes in noise pattern for digit {choosen_digit}")
for ax, img, t in zip(axes, intermediate_images, timesteps_to_save):
    ax.imshow(img.squeeze().numpy(), cmap='gray')
    ax.set_title(f'Timestep {t}')
    ax.axis('off')
plt.tight_layout()
plt.savefig(f"/home/mlut/probai/assets/gradual_change_{choosen_digit}.png")

