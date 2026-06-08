import os
import sys
import argparse
import time
import numpy as np
import cv2
import urllib.request
from PIL import Image, ImageDraw, ImageFont

# Set paths to ensure import works if needed
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

FONT_URL = "https://raw.githubusercontent.com/Gutenberg-Labo/GL-Nummernschild/main/fonts/ttf/GL-Nummernschild-Mtl.ttf"
FONT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data")
FONT_PATH = os.path.join(FONT_DIR, "GL-Nummernschild-Mtl.ttf")

EMNIST_LABELS = (
    '0123456789'
    'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    'abdefghnqrt'
)

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
    
    # 2. Resize maintaining aspect ratio to fit inside 28x28
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
    """Applies random rotations, shifts, blurs, stroke modifications, noise, cuts, spots, and tampering."""
    pil_img = Image.fromarray(img)
    w, h = pil_img.size
    
    # 1. Random Rotation (-12 to 12 degrees)
    angle = np.random.uniform(-12, 12)
    pil_img = pil_img.rotate(angle, resample=Image.BICUBIC, expand=False)
    
    # 2. Random Shear / Skew (simulates camera perspective)
    if np.random.rand() < 0.5:
        shear_factor = np.random.uniform(-0.15, 0.15)
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
        
    # 5. Entrecortes / Fading (horizontal/vertical lines cutting through the character)
    if np.random.rand() < 0.35:
        num_cuts = np.random.randint(1, 3)
        for _ in range(num_cuts):
            if np.random.rand() < 0.5:
                # Horizontal cut
                y = np.random.randint(2, 26)
                thickness = np.random.choice([1, 2])
                img[y:y+thickness, :] = 0
            else:
                # Vertical cut
                x = np.random.randint(2, 26)
                thickness = np.random.choice([1, 2])
                img[:, x:x+thickness] = 0

    # 6. Manchas / Suciedad (spots: dark spots simulating dirt or white spots simulating peeling paint)
    if np.random.rand() < 0.35:
        num_spots = np.random.randint(1, 4)
        for _ in range(num_spots):
            spot_x = np.random.randint(0, 28)
            spot_y = np.random.randint(0, 28)
            spot_r = np.random.randint(1, 3)
            # Mostly black spots (dirt) on character, occasionally white spots (peeling/highlights)
            color = 0 if np.random.rand() < 0.7 else 255
            cv2.circle(img, (spot_x, spot_y), spot_r, color, -1)

    # 7. Adulteramiento ligero / Artefactos (white lines/shapes simulating screws, plate lines, or severe degradation)
    if np.random.rand() < 0.25:
        # Draw a small random line crossing part of the character
        x1, y1 = np.random.randint(2, 26), np.random.randint(2, 26)
        x2, y2 = x1 + np.random.randint(-5, 6), y1 + np.random.randint(-5, 6)
        cv2.line(img, (x1, y1), (x2, y2), 255, thickness=np.random.choice([1, 2]))

    if np.random.rand() < 0.2:
        # Erase a small block of pixels randomly
        x = np.random.randint(4, 24)
        y = np.random.randint(4, 24)
        w_erase = np.random.randint(2, 5)
        h_erase = np.random.randint(2, 5)
        img[y:y+h_erase, x:x+w_erase] = 0

    # 8. Random Blur (Gaussian Blur)
    if np.random.rand() < 0.3:
        ksize = np.random.choice([3, 5])
        img = cv2.GaussianBlur(img, (ksize, ksize), 0)
        
    # 9. Random Noise
    if np.random.rand() < 0.25:
        noise = np.random.normal(0, np.random.uniform(5, 15), img.shape).astype(np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
    return img


def get_folder_name(ch: str) -> str:
    """Returns a unique case-safe folder name for each class to support Windows."""
    if ch.isdigit():
        return ch
    elif ch.isupper():
        return f"upper_{ch}"
    elif ch.islower():
        return f"lower_{ch}"
    else:
        return f"char_{ord(ch)}"

def main():
    parser = argparse.ArgumentParser(description="Physically generate synthetic character images.")
    parser.add_argument("--samples", type=int, default=100, help="Number of samples to generate per class")
    parser.add_argument("--output_dir", type=str, default=None, 
                        help="Output directory path (defaults to data/synthetic_dataset)")
    parser.add_argument("--no_augment", action="store_true", help="Disable augmentations (save clean characters)")
    args = parser.parse_args()

    # Determine paths
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if args.output_dir is None:
        output_dir = os.path.join(base_dir, "data", "synthetic_dataset")
    else:
        output_dir = os.path.abspath(args.output_dir)

    print(f"[INFO] Output Directory: {output_dir}")
    print(f"[INFO] Samples per class: {args.samples}")
    print(f"[INFO] Apply Augmentations: {not args.no_augment}")

    # Ensure font exists
    if not download_font():
        print("[ERROR] Cannot proceed without the font file.")
        sys.exit(1)

    # Initialize fonts list
    fonts = [FONT_PATH]
    if os.name == 'nt':
        win_fonts = [
            'C:/Windows/Fonts/arialbd.ttf',
            'C:/Windows/Fonts/calibrib.ttf',
            'C:/Windows/Fonts/tahomabd.ttf',
        ]
        for f in win_fonts:
            if os.path.exists(f):
                fonts.append(f)
    else:
        linux_fonts = [
            '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',
            '/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf',
            '/usr/share/fonts/truetype/ubuntu/Ubuntu-B.ttf',
        ]
        for f in linux_fonts:
            if os.path.exists(f):
                fonts.append(f)

    print(f"[INFO] Available fonts for generation: {[os.path.basename(f) for f in fonts]}")

    os.makedirs(output_dir, exist_ok=True)
    
    # Save mapping file
    mapping_path = os.path.join(output_dir, "class_mapping.txt")
    with open(mapping_path, "w", encoding="utf-8") as f_map:
        f_map.write("Class Index | Character | Folder Name\n")
        f_map.write("-" * 40 + "\n")
        for idx, ch in enumerate(EMNIST_LABELS):
            f_map.write(f"{idx:<11} | {ch:<9} | {get_folder_name(ch)}\n")
    print(f"[INFO] Class mapping written to: {mapping_path}")

    start_time = time.time()
    total_generated = 0

    for idx, ch in enumerate(EMNIST_LABELS):
        class_folder_name = get_folder_name(ch)
        class_dir = os.path.join(output_dir, class_folder_name)
        os.makedirs(class_dir, exist_ok=True)

        print(f"Generating class {idx + 1}/{len(EMNIST_LABELS)}: '{ch}' (Folder: '{class_folder_name}')")

        for s_idx in range(args.samples):
            # Font selection rules
            if ch.islower():
                # FE-Schrift does not have lowercase, use sans-serif fonts if available
                if len(fonts) > 1:
                    f_path = np.random.choice(fonts[1:])
                else:
                    f_path = fonts[0]
            else:
                # 40% FE-Schrift, 60% sans-serif fonts
                if len(fonts) > 1 and np.random.rand() < 0.6:
                    f_path = np.random.choice(fonts[1:])
                else:
                    f_path = fonts[0]

            # Generate character image
            font_size = np.random.randint(28, 38)
            char_img = generate_base_char(ch, f_path, font_size)

            # Apply augmentations if requested
            if not args.no_augment:
                char_img = apply_augmentations(char_img)

            # Save as PNG image
            img_filename = f"sample_{s_idx:04d}.png"
            img_path = os.path.join(class_dir, img_filename)
            cv2.imwrite(img_path, char_img)
            total_generated += 1

    elapsed = time.time() - start_time
    print("-" * 50)
    print(f"[SUCCESS] Generated {total_generated} images in {elapsed:.2f} seconds.")
    print(f"[SUCCESS] Dataset stored at: {output_dir}")

if __name__ == "__main__":
    main()
