import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import random
from typing import Optional, Tuple, List, Callable


class TextImageDataset(Dataset):
    def __init__(
        self,
        root_dir: str = "./dataset",
        image_size: int = 64,
        transform: Optional[Callable] = None,
        use_synthetic: bool = False,
        synthetic_size: int = 1000
    ):
        self.root_dir = root_dir
        self.image_size = image_size
        self.use_synthetic = use_synthetic
        self.synthetic_size = synthetic_size
        
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
        else:
            self.transform = transform
        
        if use_synthetic:
            self.samples = self._create_synthetic_samples()
        else:
            self.samples = self._load_dataset()
    
    def _load_dataset(self) -> List[Tuple[str, str]]:
        samples = []
        images_dir = os.path.join(self.root_dir, "images")
        text_dir = os.path.join(self.root_dir, "text")
        
        if os.path.exists(images_dir) and os.path.exists(text_dir):
            for class_folder in os.listdir(images_dir):
                class_img_path = os.path.join(images_dir, class_folder)
                class_txt_path = os.path.join(text_dir, class_folder)
                
                if os.path.isdir(class_img_path) and os.path.isdir(class_txt_path):
                    for img_file in os.listdir(class_img_path):
                        if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                            img_path = os.path.join(class_img_path, img_file)
                            txt_file = os.path.splitext(img_file)[0] + ".txt"
                            txt_path = os.path.join(class_txt_path, txt_file)
                            
                            if os.path.exists(txt_path):
                                samples.append((img_path, txt_path))
        
        captions_dir = os.path.join(self.root_dir, "captions")
        if len(samples) == 0 and os.path.exists(images_dir):
            for img_file in os.listdir(images_dir):
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(images_dir, img_file)
                    txt_file = os.path.splitext(img_file)[0] + ".txt"
                    txt_path = os.path.join(captions_dir, txt_file)
                    
                    if os.path.exists(txt_path):
                        samples.append((img_path, txt_path))
        
        if len(samples) == 0:
            print("No dataset found. Using synthetic data for demonstration.")
            self.use_synthetic = True
            return self._create_synthetic_samples()
        
        print(f"Loaded {len(samples)} image-caption pairs from {self.root_dir}")
        return samples
    
    def _create_synthetic_samples(self) -> List[Tuple[torch.Tensor, str]]:
        samples = []
        captions = [
            "a small bird with red feathers and black wings",
            "a blue bird with white belly sitting on a branch",
            "a yellow songbird with brown markings",
            "a green parrot with orange beak",
            "a black crow with shiny feathers",
            "a white dove with gray wing tips",
            "a brown sparrow on the ground",
            "a colorful hummingbird near flowers",
            "a large eagle with sharp talons",
            "a pink flamingo standing in water",
            "a small finch with striped pattern",
            "a woodpecker with red head",
            "a robin with orange breast",
            "a cardinal with bright red plumage",
            "a bluejay with blue and white feathers",
        ]
        
        for i in range(self.synthetic_size):
            caption = random.choice(captions)
            samples.append((None, caption))
        
        print(f"Created {len(samples)} synthetic samples for testing")
        return samples
    
    def _generate_synthetic_image(self) -> torch.Tensor:
        image = torch.zeros(3, self.image_size, self.image_size)
        base_color = torch.rand(3, 1, 1)
        x = torch.linspace(0, 1, self.image_size).view(1, 1, -1)
        y = torch.linspace(0, 1, self.image_size).view(1, -1, 1)
        image = base_color * (0.5 + 0.5 * torch.sin(x * 6.28) * torch.cos(y * 6.28))
        image = image + 0.1 * torch.randn_like(image)
        image = torch.clamp(image * 2 - 1, -1, 1)
        return image
    
    def _read_caption(self, txt_path: str) -> str:
        with open(txt_path, 'r', encoding='utf-8') as f:
            captions = f.readlines()
        captions = [c.strip() for c in captions if c.strip()]
        if len(captions) == 0:
            return "an image"
        return random.choice(captions)
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        if self.use_synthetic:
            _, caption = self.samples[idx]
            image = self._generate_synthetic_image()
        else:
            img_path, txt_path = self.samples[idx]
            image = Image.open(img_path).convert('RGB')
            image = self.transform(image)
            caption = self._read_caption(txt_path)
        return image, caption


def get_dataloader(
    root_dir: str = "./dataset",
    batch_size: int = 16,
    image_size: int = 64,
    num_workers: int = 4,
    use_synthetic: bool = False,
    synthetic_size: int = 1000,
    shuffle: bool = True,
    pin_memory: bool = True
) -> DataLoader:
    dataset = TextImageDataset(
        root_dir=root_dir,
        image_size=image_size,
        use_synthetic=use_synthetic,
        synthetic_size=synthetic_size
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True
    )
    
    return dataloader


if __name__ == "__main__":
    print("Testing dataset loader...")
    dataloader = get_dataloader(use_synthetic=True, synthetic_size=100, batch_size=8, num_workers=0)
    images, captions = next(iter(dataloader))
    print(f"Batch loaded successfully!")
    print(f"Images shape: {images.shape}")
    print(f"Images range: [{images.min():.2f}, {images.max():.2f}]")
    print(f"Sample caption: '{captions[0]}'")
