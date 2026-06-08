import os
import sys
import cv2
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from ocr.cnn_model import CharClassifierCNN
from ocr.plate_segmentation import PlateSegmenter
from ocr.plate_ocr import idx_to_char

def load_model(model_path, device):
    model = CharClassifierCNN(num_classes=47).to(device)
    if os.path.exists(model_path):
        state = torch.load(model_path, map_location=device)
        model.load_state_dict(state)
        model.eval()
        return model
    else:
        print(f"[ERROR] Modelo no encontrado en {model_path}")
        return None

def main():
    image_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "imagenes", "image copy.png")
    
    # 1. Cargar la imagen
    img = cv2.imread(image_path)
    if img is None:
        print(f"[ERROR] No se pudo cargar la imagen de prueba: {image_path}")
        return
    print(f"[INFO] Imagen de prueba cargada exitosamente: {image_path}")
    print(f"[INFO] Tamaño de imagen: {img.shape}")

    # Extraer la porción donde está la placa heurísticamente si es el auto completo
    # O asumir que la imagen es de un tamaño y la placa está en la zona inferior
    # Como queremos probar segmentation, vamos a intentar segmentar directamente o recortar la placa a mano
    # si la imagen es todo el auto. Vamos a usar un ROI simple basado en la imagen provista (taxi con placa abajo).
    # La imagen es de 314 x 262 (según vi en artifacts thumbnail).
    # O tal vez PlateSegmenter funciona directamente. Vamos a intentar recortar la mitad inferior.
    from ocr.plate_detector import PlateDetector
    
    # Encontrar placa usando YOLO en lugar de un ROI estático
    detector = PlateDetector()
    plate_crop, bbox = detector.find_plate(img)
    
    if plate_crop is None:
        print("[WARN] YOLO no detectó la placa. Usando la mitad inferior de la imagen...")
        h, w = img.shape[:2]
        plate_crop = img[int(h*0.7):, int(w*0.2):]

    segmenter = PlateSegmenter(target_size=(28, 28))
    char_imgs, debug_img, _ = segmenter.segment_characters(plate_crop)

    if not char_imgs:
        print("[WARN] No se detectaron caracteres en el ROI. Intentando en toda la imagen...")
        char_imgs, debug_img, _ = segmenter.segment_characters(img)
    
    if not char_imgs:
        print("[ERROR] OpenCV no pudo segmentar ningún carácter de la placa.")
        return

    print(f"[INFO] OpenCV segmentó {len(char_imgs)} caracteres.")

    # Guardar imagen debug
    debug_path = "debug_segmentation.jpg"
    cv2.imwrite(debug_path, debug_img)
    print(f"[INFO] Imagen de debug de segmentación guardada en {debug_path}")

    # 2. Cargar modelos
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Usando dispositivo: {device}")

    base_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
    model_emnist = load_model(os.path.join(base_path, 'char_cnn.pth'), device)
    model_synth = load_model(os.path.join(base_path, 'char_cnn_synthetic.pth'), device)

    if not model_emnist or not model_synth:
        return

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1751,), (0.3277,)),
    ])

    # 3. Evaluar
    pred_emnist = ""
    conf_emnist_sum = 0
    pred_synth = ""
    conf_synth_sum = 0

    print("\n---------------------------------------------------------")
    print(f"| {'Idx':<3} | {'EMNIST':<15} | {'SINTÉTICO':<15} |")
    print("---------------------------------------------------------")

    for i, char_img in enumerate(char_imgs):
        pil_img = Image.fromarray(char_img)
        tensor = transform(pil_img).unsqueeze(0).to(device)

        with torch.no_grad():
            # EMNIST
            logits_e = model_emnist(tensor)
            probs_e = F.softmax(logits_e, dim=1)
            conf_e, pred_idx_e = probs_e.max(dim=1)
            char_e = idx_to_char(pred_idx_e.item())
            pred_emnist += char_e
            conf_emnist_sum += conf_e.item()

            # Synthetic
            logits_s = model_synth(tensor)
            probs_s = F.softmax(logits_s, dim=1)
            conf_s, pred_idx_s = probs_s.max(dim=1)
            char_s = idx_to_char(pred_idx_s.item())
            pred_synth += char_s
            conf_synth_sum += conf_s.item()

        print(f"| {i:<3} | {char_e} ({conf_e.item():.2f})       | {char_s} ({conf_s.item():.2f})       |")

    print("---------------------------------------------------------")
    print("\n[RESULTADOS FINALES PARA LA PLACA]")
    
    avg_conf_emnist = conf_emnist_sum / len(char_imgs)
    avg_conf_synth = conf_synth_sum / len(char_imgs)

    print(f"Modelo EMNIST:      Texto inferido = '{pred_emnist}', Confianza Media = {avg_conf_emnist:.2f}")
    print(f"Modelo SINTÉTICO:   Texto inferido = '{pred_synth}', Confianza Media = {avg_conf_synth:.2f}")

    if avg_conf_synth > avg_conf_emnist:
        print("\n=> DECISIÓN: El modelo SINTÉTICO obtiene mejor confianza. Será usado por defecto.")
    else:
        print("\n=> DECISIÓN: El modelo EMNIST obtiene mejor confianza. Será usado por defecto.")

if __name__ == "__main__":
    main()
