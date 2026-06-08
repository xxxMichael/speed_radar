import os
import sys
import cv2

sys.path.insert(0, 'src')
from ocr.plate_ocr import PlateOCR

def run_comparison():
    brain_dir = r"C:\Users\Mayck\.gemini\antigravity-ide\brain\103540da-7918-421e-a934-0997093de6cb"
    if not os.path.exists(brain_dir):
        print(f"Brain dir not found: {brain_dir}")
        return

    # Load both models
    print("[INFO] Loading EMNIST Model...")
    ocr_emnist = PlateOCR(model_path="src/char_cnn.pth")
    
    print("\n[INFO] Loading Synthetic FE-Schrift Model...")
    ocr_synth = PlateOCR(model_path="src/char_cnn_synthetic.pth")

    print("\n" + "=" * 80)
    print("  OCR COMPARISON: EMNIST MODEL vs SYNTHETIC FE-SCHRIFT MODEL")
    print("=" * 80)
    print(f"{'Image File':30s} | {'EMNIST OCR':15s} | {'Synthetic OCR':15s} | {'Status'}")
    print("-" * 80)

    for f in os.listdir(brain_dir):
        if not (f.endswith(".jpg") or f.endswith(".png")):
            continue
        
        image_path = os.path.join(brain_dir, f)
        img = cv2.imread(image_path)
        if img is None:
            continue
            
        # 1. Run EMNIST OCR
        plate_emnist, _ = ocr_emnist.read_plate(img, [0, 0, img.shape[1], img.shape[0]])
        
        # 2. Run Synthetic OCR
        plate_synth, _ = ocr_synth.read_plate(img, [0, 0, img.shape[1], img.shape[0]])
        
        # Determine comparison
        status = "Different"
        if plate_emnist == plate_synth:
            status = "Identical"
            
        print(f"{f:30s} | {plate_emnist:15s} | {plate_synth:15s} | {status}")

if __name__ == "__main__":
    run_comparison()
