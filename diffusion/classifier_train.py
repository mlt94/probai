import os
from tqdm import tqdm

import numpy as np
import random

import torch
import torch.nn as nn
from torch import optim

from torch.utils.tensorboard import SummaryWriter

from ddpm import Diffusion
from model import Classifier 
from util import set_seed, prepare_dataloaders


def create_result_folders(experiment_name):
    os.makedirs(os.path.join("weights", experiment_name), exist_ok=True)


EPOCHS = 10
IMG_SIZE = 28
INPUT_CHANNELS = 1  
CHANNELS = 32
TIME_DIM = 256
NUM_CLASSES = 10  

def train(device='cpu', T=500, img_size=IMG_SIZE, input_channels=INPUT_CHANNELS, channels=CHANNELS, time_dim=TIME_DIM):

    exp_name = 'classifier'
    create_result_folders(exp_name)  
    train_loader, val_loader, _  = prepare_dataloaders(batch_size=64) 

    diffusion = Diffusion(img_size=img_size, T=T, beta_start=1e-4, beta_end=0.02, device=device)

    model = Classifier(img_size=img_size, c_in=input_channels, labels=NUM_CLASSES, 
        time_dim=time_dim, channels=channels, device=device
    )
    model.to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Trainable parameters: {total_params/1_000_000:.2f}M')
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    softmax = nn.Softmax(dim=-1)
    pbar = tqdm(range(1, EPOCHS + 1), desc='Training')


    for epoch in pbar:
        model.train()
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            t = diffusion.sample_timesteps(images.shape[0]).to(device)
            x_t, noise = diffusion.q_sample(images, t)

            logits = model(x_t, t)

            loss = loss_fn(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    # save checkpoints
    torch.save(model.state_dict(), os.path.join("weights", exp_name, 'model.pth'))

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  
    print(f"Model will run on {device}")
    # set_seed()  # Assuming this function sets the seed for reproducibility
    train(device=device)

if __name__ == '__main__':
    main()