# Text-to-Image Generation Using Conditional Diffusion Models

![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=flat-square&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![CLIP](https://img.shields.io/badge/CLIP-OpenAI-412991?style=flat-square&logo=openai&logoColor=white)

From-scratch implementation of Text-to-Image Diffusion Model using DDPM + CLIP for conditional image generation.

**Input:** `"a small bird with red feathers and black wings"`  
**Output:** 64×64 RGB image matching the description

---

## How It Works

### Diffusion Process

```mermaid
%%{init: {'theme':'dark'}}%%
graph LR
    A[Clean Image] -->|Add Noise| B[x₁]
    B -->|Add Noise| C[x₂]
    C -->|Add Noise| D[xₜ]
    D -->|UNet Denoise| C
    C -->|UNet Denoise| B
    B -->|UNet Denoise| A
    
    style A fill:#1a472a,stroke:#2e7d32,color:#fff
    style D fill:#5c2e2e,stroke:#d32f2f,color:#fff
```

### Training Objective

$$\mathcal{L} = \mathbb{E}_{x_0, \epsilon, t} \left[ \| \epsilon - \epsilon_\theta(x_t, t, c) \|^2 \right]$$

Neural network predicts the noise $\epsilon$ added at each timestep $t$ conditioned on text $c$.

---

## Architecture

```mermaid
%%{init: {'theme':'dark'}}%%
flowchart TB
    T[Text Prompt] --> CLIP[CLIP Encoder]
    CLIP --> EMB[Embedding 512-D]
    N[Noise] --> UNET[Conditional UNet]
    EMB --> UNET
    TIME[Timestep] --> UNET
    UNET --> PRED[Predicted Noise]
    PRED --> IMG[Generated Image]
    
    style T fill:#1a472a,stroke:#2e7d32,color:#fff
    style IMG fill:#1a3d5c,stroke:#1976d2,color:#fff
    style UNET fill:#5c2e2e,stroke:#d32f2f,color:#fff
```

### UNet Details

```mermaid
%%{init: {'theme':'dark'}}%%
graph TB
    X[Noisy Image] --> CONV[Init Conv]
    T[Timestep] --> TIME[Time Embed]
    C[Text Embed] --> PROJ[Projection]
    
    CONV --> DOWN[Encoder]
    TIME --> DOWN
    DOWN --> BOTTLE[Bottleneck]
    PROJ --> BOTTLE
    BOTTLE --> UP[Decoder]
    TIME --> UP
    UP --> OUT[Output Conv]
    OUT --> NOISE[Predicted Noise]
    
    style X fill:#1a472a,stroke:#2e7d32,color:#fff
    style NOISE fill:#5c2e2e,stroke:#d32f2f,color:#fff
    style BOTTLE fill:#3d2e5c,stroke:#7b1fa2,color:#fff
```

---

## Project Structure

```
Conditional-Diffusion-Text2Image/
├── data/                # Dataset loading
├── models/              # UNet + Diffusion
├── text_encoder/        # CLIP wrapper
├── utils/               # Helpers & schedulers
├── train.py            # Training script
├── sample.py           # Generation script
└── requirements.txt
```

---

## Dataset

**CUB-200-2011 Birds Dataset** (11,788 images, 200 species)

**HuggingFace:**
```python
from datasets import load_dataset
dataset = load_dataset("alkzar90/CC6204-Hackaton-Cub-Dataset")
```

**Links:**
- [CUB-200 HuggingFace](https://huggingface.co/datasets/alkzar90/CC6204-Hackaton-Cub-Dataset)
- [CUB Train Dataset](https://huggingface.co/datasets/Multimodal-Fatima/CUB_train)
- [Original Dataset](https://www.vision.caltech.edu/datasets/cub_200_2011/)

---

## Installation

```bash
git clone https://github.com/kushalsai-01/Conditional-Diffusion-Text2Image.git
cd Conditional-Diffusion-Text2Image
pip install -r requirements.txt
```

---

## Training

### Quick Start (Synthetic Data)

```bash
python train.py --use_synthetic --epochs 20 --batch_size 8
```

### Full Training (CUB-200)

```bash
python train.py \
    --root_dir ./dataset \
    --epochs 100 \
    --batch_size 16 \
    --timesteps 1000 \
    --lr 1e-4
```

### Training Flow

```mermaid
%%{init: {'theme':'dark'}}%%
flowchart TD
    START([Train]) --> BATCH[Get Batch]
    BATCH --> ENCODE[Text → Embedding]
    ENCODE --> NOISE[Add Noise to Image]
    NOISE --> PREDICT[UNet Predicts Noise]
    PREDICT --> LOSS[MSE Loss]
    LOSS --> UPDATE[Update Weights]
    UPDATE --> CHECK{More Data?}
    CHECK -->|Yes| BATCH
    CHECK -->|No| END([Done])
    
    style START fill:#1a472a,stroke:#2e7d32,color:#fff
    style END fill:#1a472a,stroke:#2e7d32,color:#fff
    style LOSS fill:#5c2e2e,stroke:#d32f2f,color:#fff
```

---

## Generation

### Basic

```bash
python sample.py \
    --checkpoint checkpoints/model_final.pt \
    --prompts "a red bird with black wings"
```

### DDIM (Faster)

```bash
python sample.py \
    --checkpoint checkpoints/model_final.pt \
    --prompts "a blue bird on a branch" \
    --use_ddim \
    --ddim_steps 50
```

---

## Results

### Training Progress

![Training Loss](outputs/samples/training_loss.png)

| Epoch | Loss | Sample |
|-------|------|--------|
| 5 | 0.312 | ![](outputs/samples/samples_epoch_4.png) |
| 10 | 0.187 | ![](outputs/samples/samples_epoch_9.png) |
| 15 | 0.123 | ![](outputs/samples/samples_epoch_14.png) |
| 20 | 0.089 | ![](outputs/samples/samples_epoch_19.png) |

### Generated Samples

**Prompt:** "a small yellow bird with black wings"

![Generated Samples](outputs/samples/samples_final.png)

---

## Technical Details

| Component | Configuration |
|-----------|--------------|
| Model | Conditional UNet |
| Parameters | ~8M |
| Text Encoder | CLIP ViT-B/32 (frozen) |
| Image Size | 64×64 |
| Timesteps | 1000 (DDPM) / 50 (DDIM) |
| Batch Size | 16 |
| Learning Rate | 1e-4 |
| Scheduler | Cosine with warmup |

---

## References

- [DDPM](https://arxiv.org/abs/2006.11239) - Ho et al., 2020
- [DDIM](https://arxiv.org/abs/2010.02502) - Song et al., 2021
- [CLIP](https://arxiv.org/abs/2103.00020) - Radford et al., 2021

---

## License

MIT License
