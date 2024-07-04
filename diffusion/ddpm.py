import torch
from IPython import embed
from tqdm import tqdm
import logging
logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")
import math

import torch.nn.functional as F

class Diffusion:
    def __init__(self, T=500, beta_start=1e-4, beta_end=0.02, diff_type='DDPM-cfg', img_size=24,device="cuda"):
        """
        T : total diffusion steps (X_T is pure noise N(0,1))
        beta_start: value of beta for t=0
        b_end: value of beta for t=T

               """

        self.T = T
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.img_size = img_size
        self.device = device
        
        self.betas = self.get_betas().to(device)
        self.alphas = 1. - self.betas
        self.alphas_bar = torch.cumprod(self.alphas, dim=0) 
        self.diff_type = diff_type
        assert diff_type in {"DDPM-cfg"}, 'Invalid diffusion type'
        print(f'Diffusion type: {diff_type}')


    def get_betas(self, schedule='linear'):
        if schedule == 'linear':
            return torch.linspace(self.beta_start, self.beta_end, self.T)
    
    def q_sample(self, x, t):
        """
        x: input image (x0)
        t: timestep: should be torch.tensor

        Forward diffusion process
        q(x_t | x_0) = sqrt(alpha_hat_t) * x0 + sqrt(1-alpha_hat_t) * N(0,1)

        Returns q(x_t | x_0), noise
        """
        sqrt_alpha_bar = torch.sqrt(self.alphas_bar[t])
        sqrt_alpha_bar = sqrt_alpha_bar[:, None, None, None] 
        sqrt_one_minus_alpha_bar = torch.sqrt(1 - self.alphas_bar[t])
        sqrt_one_minus_alpha_bar = sqrt_one_minus_alpha_bar[:, None, None, None]
        
        noise = torch.randn_like(x)
        return sqrt_alpha_bar * x + sqrt_one_minus_alpha_bar * noise, noise
    

    def p_mean_std(self, model, x_t, t, y=None):
        """
        Calculate mean and std of p(x_{t-1} | x_t) using the reverse process and model
        """
        alpha = self.alphas[t][:, None, None, None] # match image dimensions
        alpha_bar = self.alphas_bar[t][:, None, None, None] # match image dimensions 
        beta = self.betas[t][:, None, None, None] # match image dimensions


        predicted_noise = model(x_t, t, y)

        mean = 1 / torch.sqrt(alpha) * (x_t - ((1 - alpha) / (torch.sqrt(1 - alpha_bar))) * predicted_noise) 
        std = torch.sqrt(beta)
 
        return mean, std


    def p_sample(self, model, x_t, t, y=None):
        """
        Sample from p(x{t-1} | x_t) using the reverse process and model
        """
        mean, std = self.p_mean_std(model, x_t, t, y)
        if t[0] > 1:
            noise = torch.randn_like(x_t, device=self.device)
        else:
            noise = torch.zeros_like(x_t, device=self.device)
        return mean + std * noise


    def p_sample_loop(self, model, batch_size, timesteps_to_save=None, y=None, verbose=True):
        """
        y is class label
        """
        if verbose:
            logging.info(f"Sampling {batch_size} new images....")
            pbar = tqdm(reversed(range(1, self.T)), position=0, total=self.T-1)
        else :
            pbar = reversed(range(1, self.T))

    
        model.eval()
        if timesteps_to_save is not None:
            intermediates = []
        with torch.no_grad():
            x = torch.randn((batch_size, 1, self.img_size, self.img_size)).to(self.device)
            for i in pbar:
                t = (torch.ones(batch_size) * i).long().to(self.device)
                x = self.p_sample(model, x, t, y)

                if timesteps_to_save is not None and i in timesteps_to_save:
                    x_itermediate = (x.clamp(-1, 1) + 1) / 2
                    x_itermediate = (x_itermediate * 255).type(torch.uint8)
                    intermediates.append(x_itermediate.cpu())

        model.train()
        x = (x.clamp(-1, 1) + 1) / 2
        x = (x * 255).type(torch.uint8).cpu()

        if timesteps_to_save is not None:
            intermediates.append(x)
            return x, intermediates
        else :
            return x
    

    def sample_timesteps(self, batch_size, upper_limit=None):
        """
        Sample timesteps uniformly for training
        """
        if upper_limit is None:
            return torch.randint(low=1, high=self.T, size=(batch_size,), device=self.device)
        else :
            return torch.randint(low=1, high=upper_limit, size=(batch_size,), device=self.device)
