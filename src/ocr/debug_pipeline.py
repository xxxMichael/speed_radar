"""
Script de diagnostico del pipeline OCR.
Muestra CADA etapa del procesamiento para identificar donde falla.
Ejecutar: .\.venv\Scripts\python.exe src/ocr/debug_pipeline.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms

from ocr.cnn_model import CharClassifierCNN
from ocr.plate_segmentation import PlateSegmenter

# ── Capturar un frame de la camara ──────────────────────────────────────────
print("[1] Abriendo camara (indice 1)...")
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
if not cap.isOpened():
    print("   ERROR: No se pudo abrir camara 1, intentando 0...")
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

print("   Presiona ESPACIO para capturar frame, Q para salir.")
frame = None
while True:
    ret, f = cap.read()
    if ret:
        cv2.imshow("Captura - ESPACIO para congelar", f)
    k = cv2.waitKey(1) & 0xFF
    if k == ord(' ') and ret:
        frame = f.copy()
        print("   Frame capturado.")
        break
    elif k == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

if frame is None:
    print("No se capturo frame. Saliendo.")
    sys.exit(0)

# ── 1. Region heuristica de placa ────────────────────────────────────────────
print("\n[2] Aplicando heuristica de localizacion de placa...")
h, w = frame.shape[:2]
print(f"   Frame: {w}x{h}")

# La heuristica usa la mitad inferior del FRAME COMPLETO
# (en el sistema real, se usa el bbox del vehiculo, pero aqui lo simplificamos)
vehicle_crop = frame.copy()  # Simular que el frame ES el vehiculo completo
vh, vw = vehicle_crop.shape[:2]
search_y = vh // 2
roi = vehicle_crop[search_y:, :]
print(f"   ROI de busqueda (mitad inferior): {vw}x{roi.shape[0]}")

gray   = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
blur   = cv2.bilateralFilter(gray, 9, 75, 75)
edges  = cv2.Canny(blur, 30, 200)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

roi_h, roi_w = roi.shape[:2]
plate_candidates = []
for cnt in contours:
    x, y, cw, ch = cv2.boundingRect(cnt)
    if ch < 8 or cw < 20:
        continue
    aspect = cw / float(ch)
    area   = cw * ch
    if 1.8 <= aspect <= 6.5 and area > roi_h * roi_w * 0.01:
        plate_candidates.append((area, x, y, cw, ch))

plate_candidates.sort(reverse=True)
print(f"   Candidatos de placa encontrados: {len(plate_candidates)}")
for i, (area, x, y, cw, ch) in enumerate(plate_candidates[:5]):
    print(f"     #{i+1}: pos=({x},{y}) size={cw}x{ch} aspect={cw/float(ch):.2f} area={area}")

# Mostrar roi con candidatos
roi_vis = roi.copy()
for area, x, y, cw, ch in plate_candidates[:5]:
    cv2.rectangle(roi_vis, (x, y), (x+cw, y+ch), (0, 255, 0), 2)

# ── 2. Extraer el mejor candidato ────────────────────────────────────────────
plate_crop = None
if plate_candidates:
    area, x, y, cw, ch = plate_candidates[0]
    pad = 4
    x1 = max(0, x - pad); y1 = max(0, y - pad)
    x2 = min(roi_w, x + cw + pad); y2 = min(roi_h, y + ch + pad)
    plate_crop = roi[y1:y2, x1:x2]
    print(f"\n[3] Mejor candidato de placa: {plate_crop.shape[1]}x{plate_crop.shape[0]} px")
else:
    fallback_h = max(20, vh // 4)
    plate_crop = vehicle_crop[vh - fallback_h:, :]
    print(f"\n[3] No se encontro candidato - usando fallback (franja inferior): {plate_crop.shape}")

# ── 3. Segmentacion de caracteres ────────────────────────────────────────────
print("\n[4] Segmentando caracteres...")
segmenter = PlateSegmenter(target_size=(28, 28))
char_imgs, debug_img, _stages = segmenter.segment_characters(plate_crop)
print(f"   Caracteres segmentados: {len(char_imgs)}")

# Mostrar la imagen binaria internamente
gray_plate  = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
blur_plate  = cv2.GaussianBlur(gray_plate, (5, 5), 0)
_, binary   = cv2.threshold(blur_plate, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# Estadisticas de la imagen de placa
print(f"   Tam placa: {plate_crop.shape[1]}x{plate_crop.shape[0]} px")
print(f"   Min/Max gris: {gray_plate.min()}/{gray_plate.max()}")
print(f"   Umbral Otsu: N/A (se calcula internamente)")

# Contornos que detecta el segmentador
contours2, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
h_img, w_img = binary.shape
min_h, max_h = int(h_img * 0.3), int(h_img * 0.95)
print(f"   Contornos totales en placa: {len(contours2)}")
print(f"   Filtro altura: {min_h}px - {max_h}px, aspecto: 0.2 - 1.0")
for cnt in contours2:
    x, y, cw, ch = cv2.boundingRect(cnt)
    aspect = cw / float(ch)
    pasa = (min_h < ch < max_h) and (0.2 < aspect < 1.0)
    print(f"     x={x} y={y} w={cw} h={ch} aspect={aspect:.2f}  -> {'PASA' if pasa else 'RECHAZADO'}")

# ── 4. Clasificacion CNN ─────────────────────────────────────────────────────
print("\n[5] Clasificando con CNN...")
model_path = os.path.join(os.path.dirname(__file__), '..', 'char_cnn.pth')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"   Dispositivo: {device}")
print(f"   Modelo: {model_path}")

model = CharClassifierCNN(num_classes=47).to(device)
if os.path.exists(model_path):
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    print("   Pesos cargados OK")
else:
    print("   ERROR: no se encontro char_cnn.pth")
model.eval()

EMNIST_LABELS = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabdefghnqrt'

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1751,), (0.3277,)),
])

from PIL import Image
for i, char_img in enumerate(char_imgs):
    pil = Image.fromarray(char_img)
    tensor = transform(pil).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)
        probs  = F.softmax(logits, dim=1)
    top3 = torch.topk(probs, 3, dim=1)
    print(f"   Char {i+1}: ", end='')
    for conf, idx in zip(top3.values[0], top3.indices[0]):
        char = EMNIST_LABELS[idx.item()] if idx.item() < len(EMNIST_LABELS) else '?'
        print(f"  {char}({conf.item()*100:.1f}%)", end='')
    print()

# ── 5. Mostrar visualizaciones ───────────────────────────────────────────────
print("\n[6] Mostrando visualizaciones. Presiona cualquier tecla para cerrar cada ventana.")

cv2.imshow("Frame completo", frame)
cv2.waitKey(0)

cv2.imshow("ROI busqueda + candidatos", roi_vis)
cv2.waitKey(0)

cv2.imshow("Placa extraida (color)", plate_crop)
cv2.waitKey(0)

# Mostrar binaria escalada para ver bien
binary_bgr = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
binary_big  = cv2.resize(binary_bgr, (binary.shape[1]*4, binary.shape[0]*4), interpolation=cv2.INTER_NEAREST)
cv2.imshow("Placa binarizada (Otsu) x4", binary_big)
cv2.waitKey(0)

if debug_img is not None:
    debug_big = cv2.resize(debug_img, (debug_img.shape[1]*4, debug_img.shape[0]*4), interpolation=cv2.INTER_NEAREST)
    cv2.imshow("Segmentacion caracteres x4", debug_big)
    cv2.waitKey(0)

# Mostrar caracteres individuales
if char_imgs:
    row = np.hstack([cv2.resize(c, (84, 84), interpolation=cv2.INTER_NEAREST) for c in char_imgs])
    cv2.imshow(f"Caracteres segmentados ({len(char_imgs)} total)", row)
    cv2.waitKey(0)

cv2.destroyAllWindows()
print("\n[OK] Diagnostico completado.")
