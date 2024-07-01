import matplotlib.pyplot as plt
import numpy as np
import random

# torch
import torch
import torch.nn.functional as F

# custom imports
from ddpm import Diffusion
from model import Classifier, UNet
from util import show, set_seed, CLASS_LABELS

def set_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def im_normalize(im):
    imn = (im - im.min()) / max((im.max() - im.min()), 1e-8)
    return imn

def tens2image(im):
    tmp = np.squeeze(im.numpy())
    if tmp.ndim == 2:
        return tmp
    else:
        return tmp.transpose((1, 2, 0))

    
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
set_seed()

# Load model
diffusion = Diffusion(img_size=16, T=500, beta_start=1e-4, beta_end=0.02, diff_type='DDPM-cg', device=device)

classifier = Classifier(
    img_size=16, c_in=3, labels=5,
    time_dim=256,channels=32, device=device
)
classifier.to(device)
classifier.eval()
classifier.load_state_dict(torch.load('weights/classifier/model.pth', map_location=device))

unet = UNet(device=device)
unet.eval()
unet.to(device)
unet.load_state_dict(torch.load('weights/DDPM/model.pth', map_location=device))
diffusion.classifier = classifier 

# Sample
y = torch.tensor([0,1,2,3,4], device=device) 
x_new = diffusion.p_sample_loop(unet, 5, y=y)
imgs = [im_normalize(tens2image(x_gen.cpu())) for x_gen in x_new]
show(imgs, fig_titles=CLASS_LABELS, title='Classifier Guidance', save_path='assets/cg_samples.png')
