import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.unet import ConditionalUNet
from models.diffusion import GaussianDiffusion
from text_encoder.text_encoder import get_text_encoder
from data.dataset_loader import get_dataloader
from utils.scheduler import CosineAnnealingWarmup
from utils.helpers import EMA, AverageMeter, save_checkpoint, load_checkpoint, save_images, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Train Text-to-Image Diffusion Model")
    parser.add_argument("--data_dir", type=str, default="./data/cub200", help="Path to dataset")
    parser.add_argument("--output_dir", type=str, default="./outputs", help="Output directory")
    parser.add_argument("--checkpoint", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--image_size", type=int, default=64, help="Image size")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--warmup_steps", type=int, default=1000, help="Warmup steps")
    parser.add_argument("--timesteps", type=int, default=1000, help="Diffusion timesteps")
    parser.add_argument("--base_channels", type=int, default=64, help="UNet base channels")
    parser.add_argument("--ema_decay", type=float, default=0.9999, help="EMA decay rate")
    parser.add_argument("--save_every", type=int, default=5, help="Save checkpoint every N epochs")
    parser.add_argument("--sample_every", type=int, default=5, help="Sample images every N epochs")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--use_synthetic", action="store_true", help="Use synthetic data for testing")
    parser.add_argument("--synthetic_size", type=int, default=1000, help="Synthetic dataset size")
    parser.add_argument("--fp16", action="store_true", help="Use mixed precision training")
    parser.add_argument("--gradient_clip", type=float, default=1.0, help="Gradient clipping value")
    return parser.parse_args()


def train_one_epoch(
    model: nn.Module,
    diffusion: GaussianDiffusion,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    text_encoder,
    ema: EMA,
    epoch: int,
    device: str,
    scaler=None,
    gradient_clip: float = 1.0
):
    model.train()
    loss_meter = AverageMeter()
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for step, batch in enumerate(pbar):
        images = batch["image"].to(device)
        texts = batch["text"]
        
        with torch.no_grad():
            text_embeddings = text_encoder.encode(texts)
        
        optimizer.zero_grad()
        
        if scaler is not None:
            with torch.cuda.amp.autocast():
                loss = diffusion.compute_loss(images, text_embeddings)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss = diffusion.compute_loss(images, text_embeddings)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            optimizer.step()
        
        scheduler.step()
        ema.update()
        
        loss_meter.update(loss.item())
        pbar.set_postfix({
            "loss": f"{loss_meter.avg:.4f}",
            "lr": f"{optimizer.param_groups[0]['lr']:.2e}"
        })
    
    return loss_meter.avg


@torch.no_grad()
def sample_images(
    diffusion: GaussianDiffusion,
    text_encoder,
    prompts,
    save_path: str,
    device: str,
    ema: EMA = None
):
    if ema is not None:
        ema.apply_shadow()
    
    diffusion.model.eval()
    text_embeddings = text_encoder.encode(prompts)
    samples = diffusion.sample(text_embeddings, batch_size=len(prompts))
    save_images(samples, save_path, nrow=2)
    
    if ema is not None:
        ema.restore()


def main():
    args = parse_args()
    set_seed(args.seed)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "samples"), exist_ok=True)
    
    print("Loading text encoder...")
    text_encoder = get_text_encoder(device=device)
    text_emb_dim = text_encoder.embed_dim
    
    print("Building model...")
    model = ConditionalUNet(
        in_channels=3,
        out_channels=3,
        base_channels=args.base_channels,
        channel_mults=(1, 2, 4, 8),
        num_res_blocks=2,
        time_emb_dim=256,
        text_emb_dim=text_emb_dim
    ).to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    
    diffusion = GaussianDiffusion(
        model=model,
        image_size=args.image_size,
        timesteps=args.timesteps,
        schedule="cosine"
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    
    print("Loading dataset...")
    if args.use_synthetic:
        dataloader = get_dataloader(
            data_dir=args.data_dir,
            image_size=args.image_size,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            use_synthetic=True,
            synthetic_size=args.synthetic_size
        )
    else:
        dataloader = get_dataloader(
            data_dir=args.data_dir,
            image_size=args.image_size,
            batch_size=args.batch_size,
            num_workers=args.num_workers
        )
    
    total_steps = len(dataloader) * args.epochs
    scheduler = CosineAnnealingWarmup(
        optimizer,
        warmup_steps=args.warmup_steps,
        total_steps=total_steps,
        min_lr=1e-6
    )
    
    ema = EMA(model, decay=args.ema_decay)
    
    scaler = torch.cuda.amp.GradScaler() if args.fp16 and device == "cuda" else None
    
    start_epoch = 0
    if args.checkpoint:
        checkpoint = load_checkpoint(
            args.checkpoint, model, optimizer, ema, scheduler, device
        )
        start_epoch = checkpoint.get("epoch", 0) + 1
    
    sample_prompts = [
        "A small yellow bird with black wings",
        "A red cardinal sitting on a branch",
        "A blue jay in flight",
        "A colorful hummingbird near flowers"
    ]
    
    print(f"\nStarting training from epoch {start_epoch}...")
    print(f"Total epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Total steps: {total_steps}")
    
    for epoch in range(start_epoch, args.epochs):
        avg_loss = train_one_epoch(
            model=model,
            diffusion=diffusion,
            dataloader=dataloader,
            optimizer=optimizer,
            scheduler=scheduler,
            text_encoder=text_encoder,
            ema=ema,
            epoch=epoch,
            device=device,
            scaler=scaler,
            gradient_clip=args.gradient_clip
        )
        
        print(f"Epoch {epoch} completed. Average loss: {avg_loss:.4f}")
        
        if (epoch + 1) % args.save_every == 0:
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                step=len(dataloader) * (epoch + 1),
                loss=avg_loss,
                save_path=os.path.join(args.output_dir, "checkpoints", f"checkpoint_epoch_{epoch}.pt"),
                ema=ema,
                scheduler=scheduler
            )
        
        if (epoch + 1) % args.sample_every == 0:
            sample_path = os.path.join(args.output_dir, "samples", f"samples_epoch_{epoch}.png")
            sample_images(diffusion, text_encoder, sample_prompts, sample_path, device, ema)
    
    save_checkpoint(
        model=model,
        optimizer=optimizer,
        epoch=args.epochs - 1,
        step=total_steps,
        loss=avg_loss,
        save_path=os.path.join(args.output_dir, "checkpoints", "checkpoint_final.pt"),
        ema=ema,
        scheduler=scheduler
    )
    
    print("Training completed!")


if __name__ == "__main__":
    main()
