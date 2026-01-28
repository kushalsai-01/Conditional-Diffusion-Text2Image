import argparse
import os
import torch
from typing import List

from models.unet import ConditionalUNet
from models.diffusion import GaussianDiffusion, DDIMSampler
from text_encoder.text_encoder import get_text_encoder
from utils.helpers import load_checkpoint, save_images, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Generate images from text prompts")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--output_dir", type=str, default="./generated", help="Output directory")
    parser.add_argument("--prompts", type=str, nargs="+", default=None, help="Text prompts")
    parser.add_argument("--prompt_file", type=str, default=None, help="File with prompts (one per line)")
    parser.add_argument("--num_samples", type=int, default=4, help="Number of samples per prompt")
    parser.add_argument("--image_size", type=int, default=64, help="Image size")
    parser.add_argument("--timesteps", type=int, default=1000, help="Diffusion timesteps")
    parser.add_argument("--base_channels", type=int, default=64, help="UNet base channels")
    parser.add_argument("--use_ddim", action="store_true", help="Use DDIM sampling")
    parser.add_argument("--ddim_steps", type=int, default=50, help="DDIM sampling steps")
    parser.add_argument("--eta", type=float, default=0.0, help="DDIM eta parameter")
    parser.add_argument("--use_ema", action="store_true", help="Use EMA weights")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--cfg_scale", type=float, default=1.0, help="Classifier-free guidance scale")
    return parser.parse_args()


def load_prompts(args) -> List[str]:
    if args.prompts:
        return args.prompts
    elif args.prompt_file:
        with open(args.prompt_file, 'r') as f:
            return [line.strip() for line in f if line.strip()]
    else:
        return [
            "A small yellow bird with black wings sitting on a tree branch",
            "A red cardinal bird with a distinctive crest",
            "A blue jay bird with white and blue feathers",
            "A hummingbird hovering near colorful flowers",
            "A robin with an orange-red breast on green grass",
            "A sparrow perched on a wooden fence",
            "A colorful parrot with green and red feathers",
            "An owl with large eyes in a dark forest"
        ]


@torch.no_grad()
def generate_images(
    diffusion: GaussianDiffusion,
    text_encoder,
    prompts: List[str],
    output_dir: str,
    num_samples: int = 4,
    use_ddim: bool = False,
    ddim_steps: int = 50,
    eta: float = 0.0,
    cfg_scale: float = 1.0,
    device: str = "cuda"
):
    os.makedirs(output_dir, exist_ok=True)
    diffusion.model.eval()
    
    for i, prompt in enumerate(prompts):
        print(f"\nGenerating images for prompt {i+1}/{len(prompts)}:")
        print(f"  \"{prompt}\"")
        
        text_emb = text_encoder.encode([prompt] * num_samples)
        
        if use_ddim:
            sampler = DDIMSampler(diffusion, ddim_steps=ddim_steps, eta=eta)
            samples = sampler.sample(text_emb, batch_size=num_samples)
        else:
            samples = diffusion.sample(text_emb, batch_size=num_samples)
        
        prompt_slug = prompt[:50].replace(" ", "_").replace("/", "_")
        save_path = os.path.join(output_dir, f"{i:03d}_{prompt_slug}.png")
        save_images(samples, save_path, nrow=2)
        
        for j in range(num_samples):
            individual_path = os.path.join(output_dir, f"{i:03d}_{j:02d}_{prompt_slug}.png")
            save_images(samples[j:j+1], individual_path, nrow=1)


def main():
    args = parse_args()
    set_seed(args.seed)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
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
    
    diffusion = GaussianDiffusion(
        model=model,
        image_size=args.image_size,
        timesteps=args.timesteps,
        schedule="cosine"
    ).to(device)
    
    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = load_checkpoint(args.checkpoint, model, device=device)
    
    if args.use_ema and 'ema_state_dict' in checkpoint:
        print("Using EMA weights...")
        ema_state = checkpoint['ema_state_dict']
        for name, param in model.named_parameters():
            if name in ema_state:
                param.data = ema_state[name].to(device)
    
    prompts = load_prompts(args)
    print(f"\nLoaded {len(prompts)} prompts")
    
    sampling_method = "DDIM" if args.use_ddim else "DDPM"
    steps = args.ddim_steps if args.use_ddim else args.timesteps
    print(f"Sampling method: {sampling_method} with {steps} steps")
    
    generate_images(
        diffusion=diffusion,
        text_encoder=text_encoder,
        prompts=prompts,
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        use_ddim=args.use_ddim,
        ddim_steps=args.ddim_steps,
        eta=args.eta,
        cfg_scale=args.cfg_scale,
        device=device
    )
    
    print(f"\nGeneration complete! Images saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
