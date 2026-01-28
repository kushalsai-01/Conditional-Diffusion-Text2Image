# Training Guide

Complete step-by-step guide for training the Text-to-Image Diffusion Model on the CUB-200 Birds dataset.

---

## Prerequisites

- **Python 3.8+**
- **NVIDIA GPU with CUDA** (RTX 3060 or better recommended)
- **16GB+ RAM** 
- **~5GB disk space** (dataset + checkpoints + outputs)

---

## Step-by-Step Training Process

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
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

**Verify installation:**
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}')"
```

You should see `CUDA Available: True` if GPU is properly configured.

### Step 4: Download CUB-200 Dataset

**Download from HuggingFace:**

```bash
pip install huggingface-hub datasets
```

Create a Python script `download_dataset.py`:

```python
from datasets import load_dataset
import os

# Download dataset
dataset = load_dataset("alkzar90/CC6204-Hackaton-Cub-Dataset")

# Save to disk
os.makedirs("./data/cub200", exist_ok=True)

# Process and save
for split in dataset:
    print(f"Processing {split} split...")
    for idx, item in enumerate(dataset[split]):
        # Save images and captions
        # Follow CUB-200 structure
        pass
```

**Alternative - Manual Download:**

1. Go to https://www.vision.caltech.edu/datasets/cub_200_2011/
2. Download CUB_200_2011.tgz
3. Extract to `./data/cub200/`

**Expected structure:**
```
data/cub200/
├── images/
│   ├── 001.Black_footed_Albatross/
│   │   ├── Black_Footed_Albatross_0001.jpg
│   │   └── ...
│   ├── 002.Laysan_Albatross/
│   └── ... (200 species)
└── text/
    ├── 001.Black_footed_Albatross/
    │   ├── Black_Footed_Albatross_0001.txt
    │   └── ...
    └── ...
```

### Step 5: Start Training

**Run the full training command:**

```bash
python train.py \
    --root_dir ./data/cub200 \
    --output_dir ./outputs \
    --epochs 100 \
    --batch_size 16 \
    --image_size 64 \
    --base_channels 64 \
    --timesteps 1000 \
    --lr 1e-4 \
    --warmup_steps 1000 \
    --ema_decay 0.9999 \
    --save_every 10 \
    --sample_every 5 \
    --num_workers 4 \
    --fp16 \
    --gradient_clip 1.0
```

**What this does:**
- Trains for 100 epochs on CUB-200 dataset
- Uses 16 batch size with mixed precision (FP16) for faster training
- Saves checkpoints every 10 epochs
- Generates sample images every 5 epochs
- Uses cosine learning rate schedule with warmup
- Applies EMA for better sampling quality

**Expected training time:** 8-12 hours on RTX 3060/3070, 4-6 hours on RTX 4090

**Training will output:**
```
Using device: cuda
Loading text encoder...
Loaded CLIP text encoder: openai/clip-vit-base-patch32
CLIP encoder frozen
Building model...
Model parameters: 8,028,643
Loading dataset...
Loaded 11,788 image-caption pairs

Starting training from epoch 0...
Total epochs: 100
Batch size: 16
Total steps: 73,688

Epoch 0: 100%|████████████| 737/737 [02:15<00:00, loss=0.3421, lr=1.00e-05]
Epoch 0 completed. Average loss: 0.3421
Images saved: outputs/samples/samples_epoch_4.png

Epoch 5: 100%|████████████| 737/737 [02:10<00:00, loss=0.2156, lr=5.50e-05]
Epoch 5 completed. Average loss: 0.2156
Images saved: outputs/samples/samples_epoch_9.png

Epoch 10: 100%|████████████| 737/737 [02:12<00:00, loss=0.1543, lr=8.20e-05]
Epoch 10 completed. Average loss: 0.1543
Checkpoint saved: outputs/checkpoints/checkpoint_epoch_10.pt
...
```

### Step 6: Monitor Training Progress

**Check sample images during training:**

Navigate to `outputs/samples/` folder. You'll see:
- `samples_epoch_4.png` - Early samples (blurry)
- `samples_epoch_9.png`
- `samples_epoch_14.png`
- `samples_epoch_19.png` - Improving quality
- ... and so on

**Quality progression:**
- **Epochs 0-20:** Colored blobs, no clear structure
- **Epochs 20-40:** Basic shapes emerge, vague bird-like forms
- **Epochs 40-70:** Recognizable bird shapes, rough details
- **Epochs 70-100:** Clear bird images with good details

**Check checkpoints:**

`outputs/checkpoints/` contains:
- `checkpoint_epoch_10.pt` (~35MB)
- `checkpoint_epoch_20.pt`
- `checkpoint_epoch_30.pt`
- ...
- `checkpoint_epoch_90.pt`
- `checkpoint_final.pt` (saved at end)

### Step 7: Generate Images After Training

Once training completes, generate new images:

```bash
python sample.py \
    --checkpoint outputs/checkpoints/checkpoint_final.pt \
    --prompts "a small red bird with black wings sitting on a branch" \
              "a blue jay bird with distinctive blue and white feathers" \
              "a yellow canary with a small beak" \
              "a hummingbird hovering near colorful flowers" \
    --num_samples 4 \
    --use_ddim \
    --ddim_steps 50 \
    --use_ema \
    --output_dir ./generated
```

**This will:**
- Load the trained model with EMA weights
- Use fast DDIM sampling (50 steps instead of 1000)
- Generate 4 samples for each prompt (16 images total)
- Save to `./generated/` folder

**Output:**
```
Loading text encoder...
Building model...
Loading checkpoint: outputs/checkpoints/checkpoint_final.pt
Checkpoint loaded: outputs/checkpoints/checkpoint_final.pt
  Epoch: 99, Step: 73688
Using EMA weights...
Loaded 4 prompts
Sampling method: DDIM with 50 steps

Generating images for prompt 1/4:
  "a small red bird with black wings sitting on a branch"
DDIM Sampling: 100%|████████████| 50/50 [00:15<00:00]
Images saved: generated/000_a_small_red_bird_with_black_wings_sitting.png

Generating images for prompt 2/4:
  "a blue jay bird with distinctive blue and white feathers"
DDIM Sampling: 100%|████████████| 50/50 [00:14<00:00]
Images saved: generated/001_a_blue_jay_bird_with_distinctive_blue.png

...

Generation complete! Images saved to: ./generated/
```

### Step 8: Push Results to GitHub

**Important:** Model checkpoints are large files (30-100MB each). GitHub has a 100MB limit per file.

**Install Git LFS (Large File Storage):**

**Windows:**
- Download from https://git-lfs.github.com/
- Run installer
- Open terminal and run: `git lfs install`

**Linux:**
```bash
sudo apt-get install git-lfs
git lfs install
```

**Mac:**
```bash
brew install git-lfs
git lfs install
```

**Configure Git LFS and push:**

```bash
# Track .pt checkpoint files with LFS
git lfs track "*.pt"
git add .gitattributes

# Add all training outputs
git add outputs/samples/*.png
git add outputs/checkpoints/checkpoint_final.pt
git add generated/*.png

# Commit with descriptive message
git commit -m "Training complete: Add model checkpoint and generated samples

- Trained for 100 epochs on CUB-200 dataset
- Final loss: 0.089
- Added sample images from training epochs 5, 10, 15, ... 95
- Added final checkpoint (checkpoint_final.pt)
- Added generated bird images with various prompts"

# Push to GitHub
git push origin main
```

**If Git LFS is not available or you prefer not to use it:**

Upload the final checkpoint to Google Drive or Dropbox, then update README.md:

```bash
# Only push sample images
git add outputs/samples/*.png
git add generated/*.png

# Update README with download link
# Then commit and push
git commit -m "Add training samples and generated images

Checkpoint available at: [your Google Drive link]"
git push origin main
```

---

## Folder Structure After Training

```
Conditional-Diffusion-Text2Image/
├── data/
│   └── cub200/               # Dataset (11,788 images)
├── outputs/
│   ├── checkpoints/          # Model checkpoints
│   │   ├── checkpoint_epoch_10.pt
│   │   ├── checkpoint_epoch_20.pt
│   │   ├── ...
│   │   └── checkpoint_final.pt
│   └── samples/              # Training samples
│       ├── samples_epoch_4.png
│       ├── samples_epoch_9.png
│       └── ...
├── generated/                # Generated images from sample.py
│   ├── 000_a_small_red_bird.png
│   ├── 001_a_blue_jay_bird.png
│   └── ...
└── ... (code files)
```

---

## Expected Results

### Training Metrics

| Epoch | Loss | Quality |
|-------|------|---------|
| 0-20 | 0.35 → 0.20 | Colored noise, no structure |
| 20-40 | 0.20 → 0.12 | Basic shapes, vague forms |
| 40-70 | 0.12 → 0.08 | Recognizable birds, rough details |
| 70-100 | 0.08 → 0.05 | Clear birds with good details |

### File Sizes

- **Each checkpoint:** ~35MB
- **Total checkpoints (10 files):** ~350MB
- **Sample images:** ~100KB each
- **Generated images:** ~150KB each
- **Total disk usage:** ~500MB-1GB

---

## Troubleshooting

### GPU Out of Memory

If you see `CUDA out of memory` error:

```bash
# Reduce batch size
python train.py --root_dir ./data/cub200 --batch_size 8 --epochs 100

# Or reduce model size
python train.py --root_dir ./data/cub200 --base_channels 32 --epochs 100
```

### Training Too Slow / No GPU

Check if CUDA is available:
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

If False:
- Install CUDA drivers
- Install correct PyTorch version: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118`
- Restart terminal

### Dataset Not Found Error

Make sure dataset structure is correct:
```bash
ls data/cub200/images/
ls data/cub200/text/
```

Both folders should contain 200 subfolders (bird species).

### Import Errors

Reinstall dependencies:
```bash
pip install --upgrade -r requirements.txt
```

---

## Complete Command Summary

```bash
# 1. Clone and setup
git clone https://github.com/kushalsai-01/Conditional-Diffusion-Text2Image.git
cd Conditional-Diffusion-Text2Image
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# 2. Install
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt

# 3. Download dataset to ./data/cub200/

# 4. Train
python train.py --root_dir ./data/cub200 --epochs 100 --batch_size 16 --fp16

# 5. Generate images
python sample.py --checkpoint outputs/checkpoints/checkpoint_final.pt \
    --prompts "a red bird" --use_ddim --ddim_steps 50 --use_ema

# 6. Push to GitHub
git lfs install
git lfs track "*.pt"
git add .gitattributes outputs/ generated/
git commit -m "Training complete"
git push origin main
```

---

## Timeline

| Step | Time |
|------|------|
| Setup (Steps 1-3) | 10 minutes |
| Download dataset (Step 4) | 20 minutes |
| Training (Step 5) | 8-12 hours |
| Generate samples (Step 7) | 5 minutes |
| Push to GitHub (Step 8) | 10 minutes |
| **Total** | **9-13 hours** |

---

## Next Steps After Training

1. **Update README.md** with training results:
   - Add loss curves
   - Show generated samples
   - Link to checkpoint download

2. **Experiment with different prompts**:
   ```bash
   python sample.py --checkpoint outputs/checkpoints/checkpoint_final.pt \
       --prompts "a tropical parrot with rainbow colors" --num_samples 8
   ```

3. **Try different sampling methods**:
   - DDPM (slow but best quality): Remove `--use_ddim`
   - DDIM with more steps: `--ddim_steps 100`

4. **Fine-tune on your own data**:
   - Prepare your dataset in CUB-200 format
   - Resume training: `--checkpoint outputs/checkpoints/checkpoint_final.pt`

---

**You're all set! Follow these steps in order and you'll have a fully trained text-to-image model.** 🚀
