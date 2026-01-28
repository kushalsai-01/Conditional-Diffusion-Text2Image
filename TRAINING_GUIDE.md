# Training Guide

Complete step-by-step guide for training the Text-to-Image Diffusion Model.

---

## Prerequisites

- **Python 3.8+**
- **GPU recommended** (NVIDIA with CUDA) - CPU training is very slow
- **16GB+ RAM** for full training
- **~2GB disk space** for checkpoints and outputs

---

## Setup Instructions

### Step 1: Clone the Repository

```bash
git clone https://github.com/kushalsai-01/Conditional-Diffusion-Text2Image.git
cd Conditional-Diffusion-Text2Image
```

### Step 2: Create Virtual Environment

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Linux/Mac:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

**Verify installation:**
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import transformers; print('Transformers: OK')"
```

---

## Training Options

### Option 1: Quick Test with Synthetic Data (Recommended for Testing)

**Best for:** Testing the pipeline without downloading dataset

**Time:** 1-2 hours on CPU, 10-20 minutes on GPU

```bash
python train.py \
    --use_synthetic \
    --synthetic_size 500 \
    --epochs 10 \
    --batch_size 4 \
    --base_channels 32 \
    --timesteps 100 \
    --image_size 64 \
    --save_every 5 \
    --sample_every 5
```

### Option 2: Full Training with CUB-200 Dataset

**Best for:** Production-quality results

**Time:** 6-12 hours on GPU (RTX 3060+)

**First, download the dataset:**
- Option A: [CUB-200 from HuggingFace](https://huggingface.co/datasets/alkzar90/CC6204-Hackaton-Cub-Dataset)
- Option B: [Original CUB-200](https://www.vision.caltech.edu/datasets/cub_200_2011/)

Place dataset in `./data/cub200/` with structure:
```
data/cub200/
├── images/
│   ├── 001.Black_footed_Albatross/
│   └── ...
└── text/
    ├── 001.Black_footed_Albatross/
    └── ...
```

**Run training:**
```bash
python train.py \
    --root_dir ./data/cub200 \
    --epochs 100 \
    --batch_size 16 \
    --image_size 64 \
    --base_channels 64 \
    --timesteps 1000 \
    --lr 1e-4 \
    --warmup_steps 1000 \
    --save_every 10 \
    --sample_every 5
```

### Option 3: Training with Mixed Precision (Faster on GPU)

```bash
python train.py \
    --use_synthetic \
    --epochs 20 \
    --batch_size 16 \
    --fp16 \
    --image_size 64
```

---

## Training Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--root_dir` | `./data/cub200` | Path to dataset |
| `--output_dir` | `./outputs` | Where to save checkpoints & samples |
| `--use_synthetic` | `False` | Use synthetic data (no dataset needed) |
| `--synthetic_size` | `1000` | Number of synthetic samples |
| `--epochs` | `100` | Number of training epochs |
| `--batch_size` | `16` | Batch size |
| `--image_size` | `64` | Image resolution (64×64) |
| `--timesteps` | `1000` | Diffusion timesteps |
| `--base_channels` | `64` | UNet base channels (32 for testing) |
| `--lr` | `1e-4` | Learning rate |
| `--warmup_steps` | `1000` | LR warmup steps |
| `--ema_decay` | `0.9999` | EMA decay rate |
| `--save_every` | `5` | Save checkpoint every N epochs |
| `--sample_every` | `5` | Generate samples every N epochs |
| `--fp16` | `False` | Use mixed precision training |
| `--gradient_clip` | `1.0` | Gradient clipping value |

---

## Output Files

### During Training:

**Checkpoints** (saved to `outputs/checkpoints/`):
- `checkpoint_epoch_5.pt`
- `checkpoint_epoch_10.pt`
- ...
- `checkpoint_final.pt`

**Sample Images** (saved to `outputs/samples/`):
- `samples_epoch_4.png`
- `samples_epoch_9.png`
- ...

**File Sizes:**
- Each checkpoint: ~30-100 MB
- Each sample image: ~50-200 KB

---

## Generating Images After Training

### Basic Generation

```bash
python sample.py \
    --checkpoint outputs/checkpoints/checkpoint_final.pt \
    --prompts "a red bird with black wings" \
    --num_samples 4
```

### Generate Multiple Prompts

```bash
python sample.py \
    --checkpoint outputs/checkpoints/checkpoint_final.pt \
    --prompts "a red bird with black wings" "a blue bird on a branch" "a yellow bird in flight" \
    --num_samples 4 \
    --output_dir ./my_generated_images
```

### Fast Generation with DDIM

```bash
python sample.py \
    --checkpoint outputs/checkpoints/checkpoint_final.pt \
    --prompts "a colorful parrot" \
    --num_samples 8 \
    --use_ddim \
    --ddim_steps 50 \
    --use_ema
```

**Generated images saved to:** `./generated/`

---

## Pushing Results to GitHub

### If Checkpoint Files Are Small (<100MB)

```bash
git add outputs/samples/
git add outputs/checkpoints/checkpoint_final.pt
git commit -m "Add training results and model checkpoint"
git push origin main
```

### If Checkpoint Files Are Large (>100MB)

**Option A: Use Git LFS**

```bash
# Install Git LFS first: https://git-lfs.github.com/
git lfs install
git lfs track "*.pt"
git add .gitattributes
git add outputs/checkpoints/*.pt
git add outputs/samples/
git commit -m "Add model checkpoint with LFS"
git push origin main
```

**Option B: Upload to Cloud & Link in README**

```bash
# Only push sample images
git add outputs/samples/
git commit -m "Add training sample images"
git push origin main

# Upload checkpoint_final.pt to:
# - Google Drive
# - Dropbox
# - HuggingFace Hub
# Then add download link in README.md
```

### Push Only Sample Images (Recommended)

```bash
git add outputs/samples/*.png
git commit -m "Add generated samples from training"
git push origin main
```

---

## Expected Training Time

| Hardware | Dataset | Epochs | Batch Size | Time |
|----------|---------|--------|------------|------|
| NVIDIA RTX 4090 | CUB-200 (11K) | 100 | 32 | ~4 hours |
| NVIDIA RTX 3090 | CUB-200 (11K) | 100 | 16 | ~6 hours |
| NVIDIA RTX 3060 | CUB-200 (11K) | 100 | 16 | ~12 hours |
| NVIDIA RTX 3060 | Synthetic (500) | 10 | 8 | ~20 mins |
| CPU (Intel i7) | Synthetic (100) | 5 | 4 | ~2 hours |

---

## Monitoring Training

### Watch Terminal Output

```
Epoch 0: 100%|████████| 63/63 [01:23<00:00, loss=0.3421, lr=1.00e-05]
Epoch 0 completed. Average loss: 0.3421
Checkpoint saved: outputs/checkpoints/checkpoint_epoch_0.pt
Images saved: outputs/samples/samples_epoch_4.png

Epoch 5: 100%|████████| 63/63 [01:20<00:00, loss=0.1234, lr=1.00e-04]
...
```

### Check Sample Quality

Generated samples improve over epochs:
- **Epochs 1-10:** Blurry colored blobs
- **Epochs 10-30:** Basic shapes and forms emerge
- **Epochs 30-60:** Recognizable objects
- **Epochs 60-100:** Clear, detailed images

---

## Troubleshooting

### Out of Memory (OOM) Error

```bash
# Reduce batch size
python train.py --batch_size 4

# Reduce model size
python train.py --base_channels 32

# Reduce image size
python train.py --image_size 32
```

### Training Too Slow

```bash
# Use mixed precision
python train.py --fp16

# Reduce timesteps
python train.py --timesteps 500

# Use fewer epochs for testing
python train.py --epochs 10
```

### CUDA Not Available

```bash
# Check CUDA
python -c "import torch; print(torch.cuda.is_available())"

# If False, training will use CPU (very slow)
# Consider using Google Colab for free GPU
```

---

## Quick Start Summary

```bash
# 1. Clone
git clone https://github.com/kushalsai-01/Conditional-Diffusion-Text2Image.git
cd Conditional-Diffusion-Text2Image

# 2. Setup
python -m venv venv
venv\Scripts\activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 3. Train (quick test)
python train.py --use_synthetic --epochs 10 --batch_size 4

# 4. Generate images
python sample.py --checkpoint outputs/checkpoints/checkpoint_final.pt --prompts "a red bird"

# 5. Push results
git add outputs/samples/
git commit -m "Training complete"
git push origin main
```

---

## Need Help?

- Check [README.md](README.md) for architecture details
- Open an issue on GitHub
- Review error messages in terminal output

**Happy Training! 🚀**
