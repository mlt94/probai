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
import torch.nn as nn
import argparse
from torch.optim.lr_scheduler import ReduceLROnPlateau

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

CLASS_LABELS = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]


def create_result_folders(experiment_name):
    os.makedirs("weights", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs(os.path.join("weights", experiment_name), exist_ok=True)
    os.makedirs(os.path.join("results", experiment_name), exist_ok=True)


def train(device='cuda', T=500, input_channels=1, channels=32, time_dim=256,
          batch_size=100, lr=1e-2, num_epochs=30, experiment_name="DDPM-cfg", show=False):
    create_result_folders(experiment_name)
    train_loader, val_loader, test_loader = prepare_dataloaders(batch_size)

    model = UNet(c_in=input_channels, c_out=input_channels, 
                time_dim=time_dim, channels=channels, num_classes=10, device=device).to(device)
    
    diffusion = Diffusion(T=T, beta_start=1e-4, beta_end=0.02, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True) #Learning rate scheduler

    mse = nn.MSELoss()

    logger = SummaryWriter(os.path.join("runs", experiment_name))
    l = len(train_loader)
    min_train_loss = 1e10
    for epoch in range(1, num_epochs + 1):
        logging.info(f"Starting epoch {epoch}:")
        pbar = tqdm(train_loader)
        epoch_loss = 0
        for i, (images, labels) in enumerate(pbar):
            images = images.to(device)

            labels = labels.to(device)
            labels = torch.nn.functional.one_hot(labels, num_classes=10).float() #one-hot encode labels

            #randomly discard labels as pr. step 3 in Algorithm 1 from paper to enable unconditional training
            p_uncod = 0.1
            if torch.rand(1) < p_uncod:
                labels = None

            t = diffusion.sample_timesteps(images.shape[0]).to(device)
            x_t, noise = diffusion.q_sample(images, t) #forward pass (diffusion)
            predicted_noise = model(x_t, t, labels)
            loss = mse(noise, predicted_noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            pbar.set_postfix(MSE=loss.item())
            logger.add_scalar("MSE", loss.item(), global_step=epoch * l + i)

        epoch_loss /= l
        scheduler.step(epoch_loss)
        if epoch_loss <= min_train_loss:
            torch.save(model.state_dict(), os.path.join("weights", experiment_name, f"model.pth"))
            min_train_loss = epoch_loss
            
        #Choose random int between 0-9 to input to p_sample_loop for generation with model, saved under "results/", denoising 
        y = torch.tensor([np.random.randint(0,10)], device=device)
        title = f'Epoch {epoch} with label:{CLASS_LABELS[y.item()]}'
        y = torch.nn.functional.one_hot(y, num_classes=10).float()


        sampled_images = diffusion.p_sample_loop(model, batch_size=images.shape[0], y=y) 
        save_images(images=sampled_images, path=os.path.join("results", experiment_name, f"{epoch}.jpg"),
                    show=show, title=title)
    logger.close()

def main():
    exp_name = 'DDPM-cfg'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Model will run on {device}")

    set_seed()
    train(experiment_name=exp_name, device=device)


if __name__ == '__main__':
    main()