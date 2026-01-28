import torch
import torch.nn as nn
from typing import Dict, Any, Optional
import os
from PIL import Image
import numpy as np
from torchvision.utils import make_grid


class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self._register()
    
    def _register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                new_average = self.decay * self.shadow[name] + (1.0 - self.decay) * param.data
                self.shadow[name] = new_average.clone()
    
    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]
    
    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}
    
    def state_dict(self) -> Dict[str, torch.Tensor]:
        return self.shadow.copy()
    
    def load_state_dict(self, state_dict: Dict[str, torch.Tensor]):
        self.shadow = state_dict.copy()


class AverageMeter:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    step: int,
    loss: float,
    save_path: str,
    ema: Optional[EMA] = None,
    scheduler: Optional[Any] = None,
    **kwargs
):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'step': step,
        'loss': loss
    }
    if ema is not None:
        checkpoint['ema_state_dict'] = ema.state_dict()
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    checkpoint.update(kwargs)
    torch.save(checkpoint, save_path)
    print(f"Checkpoint saved: {save_path}")


def load_checkpoint(
    checkpoint_path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    ema: Optional[EMA] = None,
    scheduler: Optional[Any] = None,
    device: str = "cuda"
) -> Dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if ema is not None and 'ema_state_dict' in checkpoint:
        ema.load_state_dict(checkpoint['ema_state_dict'])
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    print(f"Checkpoint loaded: {checkpoint_path}")
    print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}, Step: {checkpoint.get('step', 'N/A')}")
    return checkpoint


def save_images(
    images: torch.Tensor,
    save_path: str,
    nrow: int = 4,
    normalize: bool = True,
    value_range: tuple = (-1, 1)
):
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    if normalize:
        images = (images - value_range[0]) / (value_range[1] - value_range[0])
        images = images.clamp(0, 1)
    grid = make_grid(images, nrow=nrow, padding=2, normalize=False)
    grid_np = grid.permute(1, 2, 0).cpu().numpy()
    grid_np = (grid_np * 255).astype(np.uint8)
    Image.fromarray(grid_np).save(save_path)
    print(f"Images saved: {save_path}")


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    print("Testing helpers...")
    
    model = torch.nn.Linear(10, 10)
    ema = EMA(model, decay=0.999)
    
    for _ in range(10):
        model.weight.data += torch.randn_like(model.weight) * 0.1
        ema.update()
    
    ema.apply_shadow()
    print("EMA applied")
    ema.restore()
    print("EMA restored")
    
    meter = AverageMeter()
    for i in range(10):
        meter.update(i)
    print(f"Average: {meter.avg}")
    
    print("Helpers test passed!")
