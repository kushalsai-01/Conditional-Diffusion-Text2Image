import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class SinusoidalPositionEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
    
    def forward(self, time: torch.Tensor) -> torch.Tensor:
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        return embeddings


class TimeEmbedding(nn.Module):
    def __init__(self, time_dim: int, emb_dim: int):
        super().__init__()
        self.sinusoidal = SinusoidalPositionEmbedding(time_dim)
        self.mlp = nn.Sequential(
            nn.Linear(time_dim, emb_dim),
            nn.SiLU(),
            nn.Linear(emb_dim, emb_dim)
        )
    
    def forward(self, time: torch.Tensor) -> torch.Tensor:
        sinusoidal_emb = self.sinusoidal(time)
        return self.mlp(sinusoidal_emb)


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        groups: int = 8
    ):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        groups = min(groups, out_channels)
        while out_channels % groups != 0:
            groups -= 1
        self.norm = nn.GroupNorm(groups, out_channels)
        self.activation = nn.SiLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.norm(self.conv(x)))


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_emb_dim: int, groups: int = 8):
        super().__init__()
        self.conv1 = ConvBlock(in_channels, out_channels, groups=groups)
        self.conv2 = ConvBlock(out_channels, out_channels, groups=groups)
        self.time_mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, out_channels))
        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.skip = nn.Identity()
    
    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x)
        time_emb = self.time_mlp(time_emb)[:, :, None, None]
        h = h + time_emb
        h = self.conv2(h)
        return h + self.skip(x)


class DownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_emb_dim: int, num_blocks: int = 2, downsample: bool = True):
        super().__init__()
        self.blocks = nn.ModuleList()
        self.blocks.append(ResidualBlock(in_channels, out_channels, time_emb_dim))
        for _ in range(num_blocks - 1):
            self.blocks.append(ResidualBlock(out_channels, out_channels, time_emb_dim))
        if downsample:
            self.downsample = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1)
        else:
            self.downsample = nn.Identity()
    
    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        for block in self.blocks:
            x = block(x, time_emb)
        skip = x
        x = self.downsample(x)
        return x, skip


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_emb_dim: int, num_blocks: int = 2, upsample: bool = True):
        super().__init__()
        if upsample:
            self.upsample = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=4, stride=2, padding=1)
        else:
            self.upsample = nn.Identity()
        self.blocks = nn.ModuleList()
        self.blocks.append(ResidualBlock(in_channels * 2, out_channels, time_emb_dim))
        for _ in range(num_blocks - 1):
            self.blocks.append(ResidualBlock(out_channels, out_channels, time_emb_dim))
    
    def forward(self, x: torch.Tensor, skip: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        x = torch.cat([x, skip], dim=1)
        for block in self.blocks:
            x = block(x, time_emb)
        return x


class ConditionalUNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        base_channels: int = 64,
        channel_mults: Tuple[int, ...] = (1, 2, 4, 8),
        num_res_blocks: int = 2,
        time_emb_dim: int = 256,
        text_emb_dim: int = 512,
        num_groups: int = 8
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_channels = base_channels
        self.channel_mults = channel_mults
        self.num_levels = len(channel_mults)
        
        self.time_embedding = TimeEmbedding(time_emb_dim, time_emb_dim)
        
        bottleneck_channels = base_channels * channel_mults[-1]
        self.text_projection = nn.Sequential(
            nn.Linear(text_emb_dim, bottleneck_channels),
            nn.SiLU(),
            nn.Linear(bottleneck_channels, bottleneck_channels)
        )
        
        self.init_conv = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)
        
        self.down_blocks = nn.ModuleList()
        current_channels = base_channels
        for i, mult in enumerate(channel_mults):
            out_ch = base_channels * mult
            is_last = (i == len(channel_mults) - 1)
            self.down_blocks.append(
                DownBlock(
                    in_channels=current_channels,
                    out_channels=out_ch,
                    time_emb_dim=time_emb_dim,
                    num_blocks=num_res_blocks,
                    downsample=not is_last
                )
            )
            current_channels = out_ch
        
        self.bottleneck = nn.ModuleList([
            ResidualBlock(current_channels, current_channels, time_emb_dim),
            ResidualBlock(current_channels, current_channels, time_emb_dim)
        ])
        
        self.up_blocks = nn.ModuleList()
        reversed_mults = list(reversed(channel_mults))
        for i, mult in enumerate(reversed_mults):
            in_ch = base_channels * mult
            out_ch = base_channels * (reversed_mults[i + 1] if i + 1 < len(reversed_mults) else 1)
            self.up_blocks.append(
                UpBlock(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    time_emb_dim=time_emb_dim,
                    num_blocks=num_res_blocks,
                    upsample=not (i == 0)
                )
            )
        
        self.final_conv = nn.Sequential(
            nn.GroupNorm(min(num_groups, base_channels), base_channels),
            nn.SiLU(),
            nn.Conv2d(base_channels, out_channels, kernel_size=3, padding=1)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.zeros_(self.final_conv[-1].weight)
        nn.init.zeros_(self.final_conv[-1].bias)
    
    def forward(self, x: torch.Tensor, t: torch.Tensor, text_emb: torch.Tensor) -> torch.Tensor:
        time_emb = self.time_embedding(t)
        text_features = self.text_projection(text_emb)
        
        x = self.init_conv(x)
        
        skip_connections = []
        for down_block in self.down_blocks:
            x, skip = down_block(x, time_emb)
            skip_connections.append(skip)
        
        for bottleneck_block in self.bottleneck:
            x = bottleneck_block(x, time_emb)
        
        x = x + text_features[:, :, None, None]
        
        for up_block, skip in zip(self.up_blocks, reversed(skip_connections)):
            x = up_block(x, skip, time_emb)
        
        x = self.final_conv(x)
        return x


if __name__ == "__main__":
    print("Testing ConditionalUNet...")
    model = ConditionalUNet(base_channels=32, channel_mults=(1, 2, 4), text_emb_dim=512)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {num_params:,}")
    
    batch_size = 2
    x = torch.randn(batch_size, 3, 64, 64)
    t = torch.randint(0, 1000, (batch_size,))
    text_emb = torch.randn(batch_size, 512)
    
    with torch.no_grad():
        output = model(x, t, text_emb)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print("UNet test passed!")
