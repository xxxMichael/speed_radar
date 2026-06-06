"""
Modulo de reconocimiento de placas vehiculares.

Pipeline completo:
  1. Recibe el frame completo + bounding box del vehiculo (de YOLO).
  2. Recorta y preprocesa la zona del vehiculo para localizar la placa.
  3. Usa PlateSegmenter (OpenCV clasico) para extraer los caracteres.
  4. Clasifica cada caracter con la CNN entrenada (CharClassifierCNN).
  5. Devuelve el string de la placa reconstruido.

Restricciones del proyecto:
  - Sin EasyOCR ni Tesseract.
  - Solo CNN propia + OpenCV para segmentacion de caracteres.
"""

import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms

# Importar modulos del proyecto (soporte de ejecucion desde src/ o raiz)
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from ocr.cnn_model import CharClassifierCNN
from ocr.plate_segmentation import PlateSegmenter


# ---------------------------------------------------------------------------
# Mapeo de indices de salida de la CNN → caracteres visibles
# EMNIST Balanced tiene 47 clases en este orden:
#   0-9   -> digitos 0-9
#   10-35 -> mayusculas A-Z
#   36-46 -> minusculas a, b, d, e, f, g, h, n, q, r, t
# Para placas vehiculares usamos mayusculas y digitos (indices 0-35).
# ---------------------------------------------------------------------------
EMNIST_LABELS = (
    '0123456789'
    'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    'abdefghnqrt'
)


def idx_to_char(idx: int) -> str:
    """Convierte un indice de clase EMNIST al caracter correspondiente."""
    if 0 <= idx < len(EMNIST_LABELS):
        return EMNIST_LABELS[idx]
    return '?'


class PlateOCR:
    """
    Reconocedor de placas vehiculares usando OpenCV + CNN personalizada.

    Uso:
        ocr = PlateOCR(model_path='char_cnn.pth')
        plate_str, debug_img = ocr.read_plate(frame, bbox)
    """

    def __init__(self, model_path: str = 'char_cnn.pth', device: str = None):
        """
        Carga la CNN entrenada y configura el pipeline.

        Args:
            model_path (str): Ruta al archivo .pth con los pesos entrenados.
            device (str | None): 'cuda', 'cpu', o None (auto-detecta CUDA).
        """
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        print(f"[PlateOCR] Cargando CNN desde: {model_path}  |  Dispositivo: {self.device}")

        # Modelo CNN
        self.model = CharClassifierCNN(num_classes=47).to(self.device)

        if os.path.exists(model_path):
            state = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state)
            print(f"[PlateOCR] Pesos cargados correctamente.")
        else:
            print(f"[PlateOCR] ADVERTENCIA: no se encontro {model_path}. Usando pesos aleatorios.")

        self.model.eval()

        # Segmentador de caracteres (OpenCV clasico)
        self.segmenter = PlateSegmenter(target_size=(28, 28))

        # Transformacion de preprocesado para la CNN
        # Mismos estadisticos que se usaron en el entrenamiento (EMNIST balanced)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1751,), (0.3277,)),
        ])

        # Umbral de confianza minimo por caracter (0-1)
        # Caracteres con confianza menor se ignoran
        self.char_conf_threshold = 0.35

    # ------------------------------------------------------------------
    # Utilidades de deteccion de placa dentro del bbox del vehiculo
    # ------------------------------------------------------------------

    def _find_plate_region(self, vehicle_crop: np.ndarray) -> np.ndarray | None:
        """
        Intenta localizar la region de la placa dentro del recorte del vehiculo.

        Heuristica:
        - Las placas suelen estar en la mitad inferior del vehiculo.
        - Busca rectangulos con relacion de aspecto entre 2:1 y 6:1.
        - Devuelve el recorte de la mejor candidata, o el propio crop si no encuentra.

        Args:
            vehicle_crop: Imagen BGR del vehiculo recortado.

        Returns:
            numpy.ndarray: Recorte BGR de la placa, o None si el crop es invalido.
        """
        if vehicle_crop is None or vehicle_crop.size == 0:
            return None

        h, w = vehicle_crop.shape[:2]
        if h < 10 or w < 10:
            return None

        # Zona de busqueda: mitad inferior del vehiculo
        search_y = h // 2
        roi = vehicle_crop[search_y:, :]

        gray  = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blur  = cv2.bilateralFilter(gray, 9, 75, 75)
        edges = cv2.Canny(blur, 30, 200)

        # Cerrar huecos en los bordes
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best_crop   = None
        best_area   = 0
        roi_h, roi_w = roi.shape[:2]

        for cnt in contours:
            x, y, cw, ch = cv2.boundingRect(cnt)
            if ch < 8 or cw < 20:
                continue

            aspect = cw / float(ch)
            area   = cw * ch

            # Placa tipica: relacion ancho/alto entre 2 y 6, y area > 1% del ROI
            if 1.8 <= aspect <= 6.5 and area > roi_h * roi_w * 0.01:
                if area > best_area:
                    best_area = area
                    # Pequeno padding
                    pad = 4
                    x1 = max(0, x - pad)
                    y1 = max(0, y - pad)
                    x2 = min(roi_w, x + cw + pad)
                    y2 = min(roi_h, y + ch + pad)
                    best_crop = roi[y1:y2, x1:x2]

        # Si no encontro una placa, devolver la franja inferior del vehiculo
        # (es mejor que devolver nada para el prototipo)
        if best_crop is None:
            fallback_h = max(20, h // 4)
            best_crop  = vehicle_crop[h - fallback_h:, :]

        return best_crop

    # ------------------------------------------------------------------
    # Clasificacion de un caracter con la CNN
    # ------------------------------------------------------------------

    def _classify_char(self, char_img: np.ndarray) -> tuple[str, float]:
        """
        Clasifica un caracter de 28x28 px con la CNN.

        Args:
            char_img: Imagen en escala de grises 28x28 (uint8).

        Returns:
            tuple[str, float]: (caracter predicho, confianza 0-1).
        """
        # Convertir a PIL Image para la transformacion
        from PIL import Image
        pil_img = Image.fromarray(char_img)

        tensor = self.transform(pil_img).unsqueeze(0).to(self.device)  # [1, 1, 28, 28]

        with torch.no_grad():
            logits = self.model(tensor)               # [1, 47]
            probs  = F.softmax(logits, dim=1)         # [1, 47]
            conf, pred_idx = probs.max(dim=1)

        char = idx_to_char(pred_idx.item())
        return char, conf.item()

    # ------------------------------------------------------------------
    # API publica principal
    # ------------------------------------------------------------------

    def read_plate(self, frame: np.ndarray, bbox: list) -> tuple[str, np.ndarray | None]:
        """
        Lee la placa de un vehiculo dado su bounding box en el frame.

        Args:
            frame (numpy.ndarray): Frame BGR completo.
            bbox (list): [x1, y1, x2, y2] del vehiculo en pixeles.

        Returns:
            tuple[str, numpy.ndarray | None]:
                - Texto de la placa (str). Puede ser vacio si no se detectaron caracteres.
                - Imagen de debug de la segmentacion (puede ser None).
        """
        x1, y1, x2, y2 = bbox
        fh, fw = frame.shape[:2]

        # Recortar vehiculo con margen de seguridad
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(fw, x2)
        y2 = min(fh, y2)

        vehicle_crop = frame[y1:y2, x1:x2]

        # Localizar region de placa dentro del recorte
        plate_crop = self._find_plate_region(vehicle_crop)
        if plate_crop is None or plate_crop.size == 0:
            return '', None

        # Segmentar caracteres
        char_imgs, debug_img = self.segmenter.segment_characters(plate_crop)

        if not char_imgs:
            return '', debug_img

        # Clasificar cada caracter
        plate_chars = []
        for char_img in char_imgs:
            char, conf = self._classify_char(char_img)
            if conf >= self.char_conf_threshold:
                plate_chars.append(char)

        plate_str = ''.join(plate_chars).upper()
        return plate_str, debug_img
