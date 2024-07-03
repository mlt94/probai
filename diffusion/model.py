# Adapted from : https://github.com/dome272/Diffusion-Models-pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F

def pos_encoding(t, channels, device):
    inv_freq = 1.0 / (
        10000
        ** (torch.arange(0, channels, 2, device=device).float() / channels)
    )
    pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
    pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
    pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
    return pos_enc

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim):
        super(Down, self).__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x, t):
        x = self.pool(x)
        x = self.conv(x)
        return x


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim):
        super(Up, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)
        self.emb_proj = nn.Sequential(
            nn.Linear(emb_dim, out_channels)
        )

    def forward(self, x1, x2, t):
        x1 = self.up(x1)
        # Add positional encoding
        x1 = F.interpolate(x1, size=x2.shape[2:], mode='bilinear', align_corners=True) #bug fix
        t = self.emb_proj(t).unsqueeze(-1).unsqueeze(-1)
        x1 = x1 + t
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, img_size=28, c_in=1, c_out=1, time_dim=256, device="cpu", channels=32, num_classes=10):
        '''Expects one-hot encoded classes '''
        super(UNet, self).__init__()
        self.device = device
        self.time_dim = time_dim
        self.inc = DoubleConv(c_in, channels)
        self.down1 = Down(channels, channels*2, emb_dim=time_dim)
        self.down2 = Down(channels*2, channels*4, emb_dim=time_dim)
        self.down3 = Down(channels*4, channels*8, emb_dim=time_dim)

        self.bot1 = DoubleConv(channels*8, channels*16) 
        self.bot2 = DoubleConv(channels*16, channels*16)  
        self.bot3 = DoubleConv(channels*16, channels*8)  

        self.up1 = Up(channels*8, channels*4, emb_dim=time_dim)  
        self.up2 = Up(channels*4, channels*2, emb_dim=time_dim) 
        self.up3 = Up(channels*2, channels, emb_dim=time_dim)  
        self.outc = nn.Conv2d(channels, c_out, kernel_size=1)

        if num_classes is not None:
            # Project one-hot encoded labels to the time embedding dimension with a two-layer MLP
            self.label_emb = nn.Sequential(
                nn.Linear(num_classes, time_dim),
                nn.GELU(),
                nn.Linear(time_dim, time_dim),
            )

    def forward(self, x, t, y=None):

        t = t.unsqueeze(-1).type(torch.float)
        t = pos_encoding(t, channels=self.time_dim, device=self.device)

        if y is not None:
            # Add label and time embeddings together
            t += self.label_emb(y)
            
        x1 = self.inc(x)
        x2 = self.down1(x1, t)
        x3 = self.down2(x2, t)
        x4 = self.down3(x3, t)
        x4 = self.bot1(x4)
        x4 = self.bot2(x4)
        x4 = self.bot3(x4)
        x = self.up1(x4, x3, t)
        x = self.up2(x, x2, t)
        x = self.up3(x, x1, t)
        output = self.outc(x)

        return output
    
