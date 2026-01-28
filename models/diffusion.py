import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
from tqdm import tqdm


class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        image_size: int = 64,
        timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        schedule: str = "linear"
    ):
        super().__init__()
        self.model = model
        self.image_size = image_size
        self.timesteps = timesteps
        
        if schedule == "linear":
            betas = torch.linspace(beta_start, beta_end, timesteps)
        elif schedule == "cosine":
            betas = self._cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f"Unknown schedule: {schedule}")
        
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1.0 / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1.0 / alphas_cumprod - 1))
        
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)
        self.register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min=1e-20)))
        self.register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2', (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod))
    
    def _cosine_beta_schedule(self, timesteps: int, s: float = 0.008) -> torch.Tensor:
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clamp(betas, 0.0001, 0.9999)
    
    def _extract(self, a: torch.Tensor, t: torch.Tensor, x_shape: Tuple) -> torch.Tensor:
        batch_size = t.shape[0]
        out = a.gather(-1, t)
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))
    
    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(x_start)
        sqrt_alphas_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    
    def compute_loss(self, x_start: torch.Tensor, text_emb: torch.Tensor, t: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size = x_start.shape[0]
        if t is None:
            t = torch.randint(0, self.timesteps, (batch_size,), device=x_start.device, dtype=torch.long)
        noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start, t, noise)
        predicted_noise = self.model(x_noisy, t, text_emb)
        loss = F.mse_loss(predicted_noise, noise)
        return loss
    
    def predict_start_from_noise(self, x_t: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        sqrt_recip_alphas_cumprod_t = self._extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape)
        sqrt_recipm1_alphas_cumprod_t = self._extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        return sqrt_recip_alphas_cumprod_t * x_t - sqrt_recipm1_alphas_cumprod_t * noise
    
    def q_posterior_mean_variance(self, x_start: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        posterior_mean = (
            self._extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            self._extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = self._extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = self._extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped
    
    @torch.no_grad()
    def p_sample(self, x_t: torch.Tensor, t: torch.Tensor, text_emb: torch.Tensor) -> torch.Tensor:
        batch_size = x_t.shape[0]
        t_tensor = torch.full((batch_size,), t, device=x_t.device, dtype=torch.long)
        predicted_noise = self.model(x_t, t_tensor, text_emb)
        x_start = self.predict_start_from_noise(x_t, t_tensor, predicted_noise)
        x_start = torch.clamp(x_start, -1.0, 1.0)
        posterior_mean, _, posterior_log_variance = self.q_posterior_mean_variance(x_start, x_t, t_tensor)
        noise = torch.randn_like(x_t) if t > 0 else torch.zeros_like(x_t)
        return posterior_mean + torch.exp(0.5 * posterior_log_variance) * noise
    
    @torch.no_grad()
    def sample(self, text_emb: torch.Tensor, batch_size: int = 1, return_all_timesteps: bool = False) -> torch.Tensor:
        device = text_emb.device
        shape = (batch_size, 3, self.image_size, self.image_size)
        x = torch.randn(shape, device=device)
        images = [x] if return_all_timesteps else None
        for t in tqdm(reversed(range(self.timesteps)), desc="Sampling", total=self.timesteps):
            x = self.p_sample(x, t, text_emb)
            if return_all_timesteps:
                images.append(x)
        if return_all_timesteps:
            return torch.stack(images, dim=1)
        return x


class DDIMSampler:
    def __init__(self, diffusion: GaussianDiffusion, ddim_steps: int = 50, eta: float = 0.0):
        self.diffusion = diffusion
        self.ddim_steps = ddim_steps
        self.eta = eta
        self.ddim_timesteps = self._get_ddim_timesteps()
    
    def _get_ddim_timesteps(self) -> torch.Tensor:
        c = self.diffusion.timesteps // self.ddim_steps
        return torch.arange(0, self.diffusion.timesteps, c) + 1
    
    @torch.no_grad()
    def sample(self, text_emb: torch.Tensor, batch_size: int = 1) -> torch.Tensor:
        device = text_emb.device
        shape = (batch_size, 3, self.diffusion.image_size, self.diffusion.image_size)
        x = torch.randn(shape, device=device)
        ddim_timesteps = self.ddim_timesteps.to(device)
        
        for i in tqdm(reversed(range(len(ddim_timesteps))), desc="DDIM Sampling", total=len(ddim_timesteps)):
            t = ddim_timesteps[i]
            t_prev = ddim_timesteps[i - 1] if i > 0 else torch.tensor(0, device=device)
            t_tensor = torch.full((batch_size,), t, device=device, dtype=torch.long)
            
            predicted_noise = self.diffusion.model(x, t_tensor, text_emb)
            
            alpha_cumprod_t = self.diffusion.alphas_cumprod[t]
            alpha_cumprod_t_prev = self.diffusion.alphas_cumprod[t_prev] if t_prev > 0 else torch.tensor(1.0, device=device)
            
            x0_pred = (x - torch.sqrt(1 - alpha_cumprod_t) * predicted_noise) / torch.sqrt(alpha_cumprod_t)
            x0_pred = torch.clamp(x0_pred, -1.0, 1.0)
            
            sigma = self.eta * torch.sqrt((1 - alpha_cumprod_t_prev) / (1 - alpha_cumprod_t)) * torch.sqrt(1 - alpha_cumprod_t / alpha_cumprod_t_prev)
            
            dir_xt = torch.sqrt(1 - alpha_cumprod_t_prev - sigma**2) * predicted_noise
            
            noise = torch.randn_like(x) if i > 0 else torch.zeros_like(x)
            x = torch.sqrt(alpha_cumprod_t_prev) * x0_pred + dir_xt + sigma * noise
        
        return x


if __name__ == "__main__":
    from models.unet import ConditionalUNet
    
    print("Testing GaussianDiffusion...")
    model = ConditionalUNet(base_channels=32, channel_mults=(1, 2), text_emb_dim=512)
    diffusion = GaussianDiffusion(model, image_size=32, timesteps=100)
    
    batch_size = 2
    x = torch.randn(batch_size, 3, 32, 32)
    text_emb = torch.randn(batch_size, 512)
    
    loss = diffusion.compute_loss(x, text_emb)
    print(f"Training loss: {loss.item():.4f}")
    
    print("\nTesting sampling (reduced steps)...")
    diffusion_small = GaussianDiffusion(model, image_size=32, timesteps=10)
    samples = diffusion_small.sample(text_emb[:1], batch_size=1)
    print(f"Sample shape: {samples.shape}")
    
    print("\nDiffusion test passed!")
