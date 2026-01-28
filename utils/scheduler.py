import math
from torch.optim.lr_scheduler import _LRScheduler
from typing import List


class LinearWarmup(_LRScheduler):
    def __init__(self, optimizer, warmup_steps: int, last_epoch: int = -1):
        self.warmup_steps = warmup_steps
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self) -> List[float]:
        if self.last_epoch < self.warmup_steps:
            warmup_factor = (self.last_epoch + 1) / self.warmup_steps
            return [base_lr * warmup_factor for base_lr in self.base_lrs]
        return self.base_lrs


class CosineAnnealingWarmup(_LRScheduler):
    def __init__(
        self,
        optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr: float = 1e-6,
        last_epoch: int = -1
    ):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self) -> List[float]:
        if self.last_epoch < self.warmup_steps:
            warmup_factor = (self.last_epoch + 1) / self.warmup_steps
            return [base_lr * warmup_factor for base_lr in self.base_lrs]
        
        progress = (self.last_epoch - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
        cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
        
        return [
            self.min_lr + (base_lr - self.min_lr) * cosine_decay
            for base_lr in self.base_lrs
        ]


class ExponentialDecay(_LRScheduler):
    def __init__(
        self,
        optimizer,
        decay_rate: float = 0.99,
        decay_steps: int = 1000,
        min_lr: float = 1e-6,
        warmup_steps: int = 0,
        last_epoch: int = -1
    ):
        self.decay_rate = decay_rate
        self.decay_steps = decay_steps
        self.min_lr = min_lr
        self.warmup_steps = warmup_steps
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self) -> List[float]:
        if self.last_epoch < self.warmup_steps:
            warmup_factor = (self.last_epoch + 1) / self.warmup_steps
            return [base_lr * warmup_factor for base_lr in self.base_lrs]
        
        effective_step = self.last_epoch - self.warmup_steps
        decay_factor = self.decay_rate ** (effective_step / self.decay_steps)
        
        return [
            max(self.min_lr, base_lr * decay_factor)
            for base_lr in self.base_lrs
        ]


if __name__ == "__main__":
    import torch
    import matplotlib.pyplot as plt
    
    model = torch.nn.Linear(10, 10)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    warmup_steps = 100
    total_steps = 1000
    
    scheduler = CosineAnnealingWarmup(optimizer, warmup_steps=warmup_steps, total_steps=total_steps, min_lr=1e-6)
    
    lrs = []
    for step in range(total_steps):
        lrs.append(optimizer.param_groups[0]['lr'])
        scheduler.step()
    
    plt.figure(figsize=(10, 5))
    plt.plot(lrs)
    plt.xlabel('Step')
    plt.ylabel('Learning Rate')
    plt.title('Cosine Annealing with Warmup')
    plt.axvline(x=warmup_steps, color='r', linestyle='--', label='Warmup End')
    plt.legend()
    plt.grid(True)
    plt.savefig('lr_schedule.png')
    print("Learning rate schedule saved to lr_schedule.png")
