import os
import sys
import time
import urllib.request
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageDraw, ImageFont

# Set paths
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from ocr.cnn_model import CharClassifierCNN

# =============================================================================
# CONFIGURATION
# =============================================================================
FONT_URL = "https://raw.githubusercontent.com/Gutenberg-Labo/GL-Nummernschild/main/fonts/ttf/GL-Nummernschild-Mtl.ttf"
FONT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data")
FONT_PATH = os.path.join(FONT_DIR, "GL-Nummernschild-Mtl.ttf")

EMNIST_LABELS = (
    '0123456789'
    'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    'abdefghnqrt'
)
NUM_CLASSES = len(EMNIST_LABELS)  # 47 classes
SAVE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "char_cnn_synthetic.pth")

# Normalization stats matching EMNIST
EMNIST_MEAN = 0.1751
EMNIST_STD = 0.3277
# =============================================================================

def download_font():
    """Downloads the FE-Schrift font if it does not exist."""
    if os.path.exists(FONT_PATH):
        print(f"[INFO] Font already exists at: {FONT_PATH}")
        return True
    
    os.makedirs(FONT_DIR, exist_ok=True)
    print(f"[INFO] Downloading FE-Schrift font from: {FONT_URL}")
    try:
        urllib.request.urlretrieve(FONT_URL, FONT_PATH)
        print(f"[OK] Font downloaded successfully to: {FONT_PATH}")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to download font: {e}")
        return False

def get_text_size(draw, text, font):
    """Pillow version-compatible text size helper."""
    if hasattr(draw, "textbbox"):
        bbox = draw.textbbox((0, 0), text, font=font)
        return bbox[2] - bbox[0], bbox[3] - bbox[1]
    else:
        return draw.textsize(text, font=font)

def generate_base_char(ch: str, font_path: str, size: int) -> np.ndarray:
    """
    Renders a character in a larger canvas, extracts its exact bounding box,
    resizes it to keep aspect ratio, and centers it in a 28x28 canvas.
    This mimics the binarized output of the character segmenter perfectly.
    """
    # 1. Render character in a larger 64x64 canvas
    img = Image.new('L', (64, 64), color=0)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype(font_path, size=size)
    except Exception:
        # Fallback to default load if specific size fails
        font = ImageFont.load_default()
        
    tw, th = get_text_size(draw, ch, font)
    # Centering on 64x64 canvas
    tx = (64 - tw) // 2
    ty = (64 - th) // 2
    draw.text((tx, ty), ch, fill=255, font=font)
    
    # Convert to numpy to find bounding box
    img_np = np.array(img, dtype=np.uint8)
    coords = np.argwhere(img_np > 0)
    
    if len(coords) == 0:
        # Empty space / nothing rendered
        return np.zeros((28, 28), dtype=np.uint8)
    
    # Bounding box of character
    ymin, xmin = coords.min(axis=0)
    ymax, xmax = coords.max(axis=0)
    
    char_crop = img_np[ymin:ymax+1, xmin:xmax+1]
    ch_h, ch_w = char_crop.shape[:2]
    
    # 2. Resize maintaining aspect ratio to fit inside 28x28 (e.g. height between 16 and 22 px)
    target_max = np.random.randint(18, 23)
    scale = target_max / max(ch_h, ch_w)
    new_w = max(1, int(ch_w * scale))
    new_h = max(1, int(ch_h * scale))
    
    resized = cv2.resize(char_crop, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # 3. Paste centered on 28x28 canvas
    canvas = np.zeros((28, 28), dtype=np.uint8)
    dx = (28 - new_w) // 2
    dy = (28 - new_h) // 2
    canvas[dy:dy+new_h, dx:dx+new_w] = resized
    
    return canvas


def apply_augmentations(img: np.ndarray) -> np.ndarray:
    """Applies random rotations, shifts, blurs, stroke modifications, and noise."""
    # Convert to PIL Image for easy rotation and affine transforms
    pil_img = Image.fromarray(img)
    w, h = pil_img.size
    
    # 1. Random Rotation (-12 to 12 degrees)
    angle = np.random.uniform(-12, 12)
    pil_img = pil_img.rotate(angle, resample=Image.BICUBIC, expand=False)
    
    # 2. Random Shear / Skew (simulates camera perspective)
    if np.random.rand() < 0.5:
        shear_factor = np.random.uniform(-0.15, 0.15)
        # Affine matrix: [1, a, 0, 0, 1, 0]
        pil_img = pil_img.transform((w, h), Image.AFFINE, (1, shear_factor, 0, 0, 1, 0), resample=Image.BICUBIC)
        
    img = np.array(pil_img, dtype=np.uint8)
    
    # 3. Random Translation / Shift (-2 to 2 pixels)
    tx = np.random.randint(-2, 3)
    ty = np.random.randint(-2, 3)
    M = np.float32([[1, 0, tx], [0, 1, ty]])
    img = cv2.warpAffine(img, M, (28, 28), flags=cv2.INTER_NEAREST)
    
    # 4. Stroke Thickness Variation (Dilate/Erode)
    if np.random.rand() < 0.25:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        img = cv2.dilate(img, kernel, iterations=1)
    elif np.random.rand() < 0.25:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        img = cv2.erode(img, kernel, iterations=1)
        
    # 5. Random Blur (Gaussian Blur)
    if np.random.rand() < 0.3:
        ksize = np.random.choice([3, 5])
        img = cv2.GaussianBlur(img, (ksize, ksize), 0)
        
    # 6. Random Noise
    if np.random.rand() < 0.2:
        noise = np.random.normal(0, np.random.uniform(5, 15), img.shape).astype(np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
    return img


class SyntheticCharDataset(Dataset):
    """
    Dataset of synthetic characters rendered using FE-Schrift and standard sans-serif fonts
    with heavy data augmentation, cached in memory.
    """
    def __init__(self, font_path: str, samples_per_class: int, is_train: bool = True):
        self.samples_per_class = samples_per_class
        self.is_train = is_train
        
        # Cargar y mezclar tipografias
        self.fonts = [font_path]
        if os.name == 'nt':
            win_fonts = [
                'C:/Windows/Fonts/arialbd.ttf',
                'C:/Windows/Fonts/calibrib.ttf',
                'C:/Windows/Fonts/tahomabd.ttf',
            ]
            for f in win_fonts:
                if os.path.exists(f):
                    self.fonts.append(f)
        else:
            linux_fonts = [
                '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',
                '/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf',
                '/usr/share/fonts/truetype/ubuntu/Ubuntu-B.ttf',
            ]
            for f in linux_fonts:
                if os.path.exists(f):
                    self.fonts.append(f)
                    
        print(f"[SyntheticCharDataset] Available fonts for training: {[os.path.basename(f) for f in self.fonts]}")
        
        self.images = []
        self.labels = []
        
        print(f"[SyntheticCharDataset] Generating {'Training' if is_train else 'Validation'} set ({samples_per_class} samples/class)...")
        start_t = time.time()
        
        for idx, ch in enumerate(EMNIST_LABELS):
            for _ in range(samples_per_class):
                # Seleccionar tipografia segun tipo de caracter
                if ch.islower():
                    # FE-Schrift no tiene minusculas, usamos Arial/Calibri/DejaVu
                    if len(self.fonts) > 1:
                        f_path = np.random.choice(self.fonts[1:])
                    else:
                        f_path = self.fonts[0]
                else:
                    # Para mayusculas y digitos (placas reales), mezclamos:
                    # 40% FE-Schrift (placas nuevas) y 60% tipografias standard negritas (placas clasicas)
                    if len(self.fonts) > 1 and np.random.rand() < 0.6:
                        f_path = np.random.choice(self.fonts[1:])
                    else:
                        f_path = self.fonts[0]
                
                # Random font size
                font_size = np.random.randint(28, 38)
                char_img = generate_base_char(ch, f_path, font_size)
                
                # Apply augmentations
                if is_train:
                    char_img = apply_augmentations(char_img)
                else:
                    # Light augmentation for validation set
                    if np.random.rand() < 0.3:
                        char_img = apply_augmentations(char_img)
                
                # Normalize values matching EMNIST loader (ToTensor maps 0-255 to 0.0-1.0)
                tensor = torch.tensor(char_img, dtype=torch.float32) / 255.0
                tensor = (tensor - EMNIST_MEAN) / EMNIST_STD
                tensor = tensor.unsqueeze(0)  # [1, 28, 28]
                
                self.images.append(tensor)
                self.labels.append(idx)
                
        print(f"[OK] Generated {len(self.images)} total samples in {time.time() - start_t:.1f}s")
        
    def __len__(self):
        return len(self.images)
        
    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

def train(epochs: int = 25, batch_size: int = 128, lr: float = 0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")
    
    # Download font
    if not download_font():
        print("[CRITICAL] Cannot train without FE-Schrift font.")
        return
        
    # Generate datasets
    train_dataset = SyntheticCharDataset(FONT_PATH, samples_per_class=1200, is_train=True)
    val_dataset = SyntheticCharDataset(FONT_PATH, samples_per_class=200, is_train=False)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
    
    # Initialize model
    model = CharClassifierCNN(num_classes=NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    
    print(f"\n[INFO] Starting training for {epochs} epochs...")
    print(f"[INFO] Saving weights to: {SAVE_PATH}")
    print("-" * 65)
    
    best_acc = 0.0
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # Gradient clipping
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
        scheduler.step()
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = (correct / total) * 100
        
        # Validation phase
        model.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0.0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
                
        val_epoch_loss = val_loss / len(val_loader.dataset)
        val_acc = (val_correct / val_total) * 100
        
        print(f"Epoch [{epoch+1:>2}/{epochs}] | "
              f"Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:>5.2f}% | "
              f"Val Loss: {val_epoch_loss:.4f} | Val Acc: {val_acc:>5.2f}%")
              
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"  --> Saved new best checkpoint with Val Acc: {val_acc:.2f}%")
            
    print(f"\n[OK] Training finished. Best Validation Accuracy: {best_acc:.2f}%")
    print(f"Model saved to {SAVE_PATH}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train CharClassifierCNN on synthetic FE-Schrift characters.")
    parser.add_argument("--epochs", type=int, default=25, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    args = parser.parse_args()
    
    train(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
