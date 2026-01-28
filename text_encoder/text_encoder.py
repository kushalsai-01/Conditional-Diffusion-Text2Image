import torch
import torch.nn as nn
from typing import List, Optional, Union


class SimpleTextEncoder(nn.Module):
    def __init__(self, vocab_size: int = 10000, embed_dim: int = 512, max_length: int = 77):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_length = max_length
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.positional_embedding = nn.Embedding(max_length, embed_dim)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=8,
                dim_feedforward=embed_dim * 4,
                dropout=0.1,
                batch_first=True
            ),
            num_layers=4
        )
        self.final_proj = nn.Linear(embed_dim, embed_dim)
        self._init_weights()
    
    def _init_weights(self):
        nn.init.normal_(self.embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding.weight, std=0.02)
    
    def tokenize(self, texts: List[str]) -> torch.Tensor:
        batch_size = len(texts)
        tokens = torch.zeros(batch_size, self.max_length, dtype=torch.long)
        for i, text in enumerate(texts):
            text_tokens = [hash(word) % (self.embedding.num_embeddings - 2) + 1 for word in text.lower().split()]
            text_tokens = text_tokens[:self.max_length - 2]
            text_tokens = [0] + text_tokens + [self.embedding.num_embeddings - 1]
            for j, token in enumerate(text_tokens):
                tokens[i, j] = token
        return tokens
    
    def forward(self, text: Union[torch.Tensor, List[str]]) -> torch.Tensor:
        if isinstance(text, list):
            tokens = self.tokenize(text)
            tokens = tokens.to(next(self.parameters()).device)
        else:
            tokens = text
        positions = torch.arange(tokens.size(1), device=tokens.device).unsqueeze(0).expand(tokens.size(0), -1)
        x = self.embedding(tokens) + self.positional_embedding(positions)
        x = self.transformer(x)
        x = x[:, 0, :]
        x = self.final_proj(x)
        return x


class TextEncoder(nn.Module):
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32", freeze: bool = True):
        super().__init__()
        self.model_name = model_name
        self.freeze = freeze
        self.use_clip = False
        self.embed_dim = 512
        
        try:
            from transformers import CLIPTextModel, CLIPTokenizer
            self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
            self.encoder = CLIPTextModel.from_pretrained(model_name)
            self.embed_dim = self.encoder.config.hidden_size
            self.use_clip = True
            print(f"Loaded CLIP text encoder: {model_name}")
            if freeze:
                for param in self.encoder.parameters():
                    param.requires_grad = False
                print("CLIP encoder frozen")
        except Exception as e:
            print(f"Could not load CLIP: {e}")
            print("Using simple text encoder as fallback")
            self.encoder = SimpleTextEncoder(embed_dim=self.embed_dim)
            self.tokenizer = None
    
    def forward(self, text: Union[torch.Tensor, List[str]]) -> torch.Tensor:
        if self.use_clip:
            if isinstance(text, torch.Tensor):
                raise ValueError("CLIP encoder requires text strings, not tensors")
            tokens = self.tokenizer(
                text,
                padding="max_length",
                max_length=77,
                truncation=True,
                return_tensors="pt"
            )
            tokens = {k: v.to(next(self.encoder.parameters()).device) for k, v in tokens.items()}
            with torch.no_grad() if self.freeze else torch.enable_grad():
                outputs = self.encoder(**tokens)
            return outputs.pooler_output
        else:
            return self.encoder(text)
    
    def encode(self, text: Union[str, List[str]]) -> torch.Tensor:
        if isinstance(text, str):
            text = [text]
        return self.forward(text)


def get_text_encoder(model_name: str = "openai/clip-vit-base-patch32", freeze: bool = True, device: str = "cuda") -> TextEncoder:
    encoder = TextEncoder(model_name=model_name, freeze=freeze)
    encoder = encoder.to(device)
    encoder.eval()
    return encoder


if __name__ == "__main__":
    print("Testing TextEncoder...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    encoder = get_text_encoder(device=device)
    print(f"Encoder embed_dim: {encoder.embed_dim}")
    print(f"Using CLIP: {encoder.use_clip}")
    
    test_texts = [
        "A beautiful sunset over the ocean",
        "A small red bird sitting on a branch"
    ]
    
    with torch.no_grad():
        embeddings = encoder.encode(test_texts)
    
    print(f"Input texts: {test_texts}")
    print(f"Output shape: {embeddings.shape}")
    print(f"Embedding norm: {embeddings.norm(dim=1)}")
    print("TextEncoder test passed!")
