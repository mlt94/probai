import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
import random
import torch
from torch.utils.tensorboard import SummaryWriter
import torchvision
from tqdm import tqdm
from torch import optim
import logging

logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")

from ddpm import Diffusion
from model import UNet
from util import prepare_dataloaders

SEED = 1

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def save_images(images, path, show=True, title=None, nrow=10):
    grid = torchvision.utils.make_grid(images, nrow=nrow)
    ndarr = grid.permute(1, 2, 0).to('cpu').numpy()
    if title is not None:
        plt.title(title)
    plt.imshow(ndarr)
    plt.axis('off')
    if path is not None:
        plt.savefig(path, bbox_inches='tight', pad_inches=0)
    if show:
        plt.show()
    plt.close()



def create_result_folders(experiment_name):
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs(os.path.join("models", experiment_name), exist_ok=True)
    os.makedirs(os.path.join("results", experiment_name), exist_ok=True)

def train(device='cuda', T=500, img_size=28, input_channels=1, channels=32, time_dim=256,
          batch_size=100, lr=1e-3, num_epochs=30, experiment_name="ddpm-CG", show=False):
    create_result_folders(experiment_name)
    train_loader, val_loader, test_loader = prepare_dataloaders(batch_size)

    model = UNet(img_size=img_size, c_in=input_channels, c_out=input_channels, 
                 time_dim=time_dim, channels=channels, device=device).to(device)
    diffusion = Diffusion(T=T, beta_start=1e-4, beta_end=0.02, diff_type='DDPM-cg',img_size=16, device=device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    mse = torch.nn.MSELoss()

    logger = SummaryWriter(os.path.join("runs", experiment_name))
    total_steps = len(train_loader)

    for epoch in range(1, num_epochs + 1):
        logging.info(f"Starting epoch {epoch}:")
        pbar = tqdm(train_loader)

        for i, (images, _) in enumerate(pbar):
            images = images.to(device)

            t = diffusion.sample_timesteps(images.shape[0]).to(device)
            x_t, noise = diffusion.q_sample(images, t)
            predicted_noise = model(x_t, t)
            loss = mse(noise, predicted_noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_postfix(MSE=loss.item())
            logger.add_scalar("MSE", loss.item(), global_step=epoch * total_steps + i)

        sampled_images = diffusion.p_sample_loop(model, batch_size=images.shape[0])
        save_images(images=sampled_images, path=os.path.join("results", experiment_name, f"{epoch}.jpg"),
                    show=show, title=f'Epoch {epoch}')
        torch.save(model.state_dict(), os.path.join("models", experiment_name, f"weights-{epoch}.pt"))

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Model will run on {device}")
    set_seed(seed=SEED)
    train(device=device)

if __name__ == '__main__':
    main()