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


class SelfAttention(nn.Module):
    def __init__(self, in_channels, size):
        super(SelfAttention, self).__init__()
        self.in_channels = in_channels
        self.size = size
        self.query = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.key = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.value = nn.Conv2d(in_channels, in_channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, channels, height, width = x.size()
        query = self.query(x).view(batch_size, -1, height * width).permute(0, 2, 1)
        key = self.key(x).view(batch_size, -1, height * width)
        value = self.value(x).view(batch_size, -1, height * width)

        attention = torch.bmm(query, key)
        attention = F.softmax(attention, dim=-1)

        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(batch_size, channels, height, width)
        out = self.gamma * out + x

        return out


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
        self.sa1 = SelfAttention(channels*2, img_size // 2)
        self.down2 = Down(channels*2, channels*4, emb_dim=time_dim)
        self.sa2 = SelfAttention(channels*4, img_size // 4)
        self.down3 = Down(channels*4, channels*8, emb_dim=time_dim)  # Increased to channels*8
        self.sa3 = SelfAttention(channels*8, img_size // 8)

        self.bot1 = DoubleConv(channels*8, channels*16)  # Increased to channels*16
        self.bot2 = DoubleConv(channels*16, channels*16)  # Increased to channels*16
        self.bot3 = DoubleConv(channels*16, channels*8)  # Increased to channels*8

        self.up1 = Up(channels*8, channels*4, emb_dim=time_dim)  
        self.sa4 = SelfAttention(channels*4, img_size // 4)
        self.up2 = Up(channels*4, channels*2, emb_dim=time_dim) 
        self.sa5 = SelfAttention(channels*2, img_size // 2)
        self.up3 = Up(channels*2, channels, emb_dim=time_dim)  
        self.sa6 = SelfAttention(channels, img_size)
        self.outc = nn.Conv2d(channels, c_out, kernel_size=1)

        if num_classes is not None:
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
        x2 = self.sa1(x2)
        x3 = self.down2(x2, t)
        x3 = self.sa2(x3)
        x4 = self.down3(x3, t)
        x4 = self.sa3(x4)
        x4 = self.bot1(x4)
        x4 = self.bot2(x4)
        x4 = self.bot3(x4)
        x = self.up1(x4, x3, t)
        x = self.sa4(x)
        x = self.up2(x, x2, t)
        x = self.sa5(x)
        x = self.up3(x, x1, t)
        x = self.sa6(x)
        output = self.outc(x)

        return output
    
class Classifier(nn.Module):
    def __init__(self, img_size=28, c_in=1, labels=10, time_dim=256, device="cuda", channels=32):
        super(Classifier, self).__init__()

        self.device = device
        self.time_dim = time_dim

        self.inc = DoubleConv(c_in, channels)
        self.down1 = Down(channels, channels*2, emb_dim=time_dim)
        self.sa1 = SelfAttention(channels*2, img_size//2)
        self.down2 = Down(channels*2, channels*4, emb_dim=time_dim)
        self.sa2 = SelfAttention(channels*4, img_size // 4)
        self.down3 = Down(channels*4, channels*4, emb_dim=time_dim)
        self.sa3 = SelfAttention(channels*4, img_size // 8)

        self.bot1 = DoubleConv(channels*4, channels*8)
        self.bot2 = DoubleConv(channels*8, channels*8)
        self.bot3 = DoubleConv(channels*8, channels*4)

        self.lin1 = nn.Linear(channels * 4 * (img_size // 8) * (img_size // 8), labels)

    def forward(self, x, t):
        t = t.unsqueeze(-1).type(torch.float)
        t = pos_encoding(t, channels=self.time_dim, device=self.device)

        x1 = self.inc(x)
        x2 = self.down1(x1, t)
        x2 = self.sa1(x2)
        x3 = self.down2(x2, t)
        x3 = self.sa2(x3)
        x4 = self.down3(x3, t)
        x4 = self.sa3(x4)
        x4 = self.bot1(x4)
        x4 = self.bot2(x4)
        x4 = self.bot3(x4)

        x = torch.flatten(x4, start_dim=1)
        output = self.lin1(x)

        return output