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
from ocr.plate_detector import PlateDetector


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

# ---------------------------------------------------------------------------
# Correcciones posicionales para placas ecuatorianas (formato AAA-NNNN)
# La CNN confunde ciertos caracteres; esta tabla corrige segun la posicion.
# Posiciones 0-2: deben ser LETRAS  → digitos se corrigen a letras similares
# Posiciones 3-6: deben ser DIGITOS → letras se corrigen a digitos similares
# ---------------------------------------------------------------------------
_DIGIT_TO_LETTER = {
    '0': 'O', '1': 'I', '2': 'Z', '5': 'S', '8': 'B', '6': 'G',
    '3': 'E', '4': 'A', '7': 'T', '9': 'P'
}
_LETTER_TO_DIGIT = {
    'O': '0', 'I': '1', 'Z': '2', 'S': '5', 'B': '8', 'G': '6',
    'D': '0', 'Q': '0', 'U': '0', 'A': '4', 'T': '7', 'l': '1',
    'J': '1', 'Y': '7', 'H': '4', 'F': '7', 'E': '3', 'P': '9'
}


def idx_to_char(idx: int) -> str:
    """Convierte un indice de clase EMNIST al caracter correspondiente."""
    if 0 <= idx < len(EMNIST_LABELS):
        return EMNIST_LABELS[idx]
    return '?'


def normalize_plate(raw: str) -> str:
    """
    Normaliza la placa leida aplicando el formato de placa ecuatoriana: AAA-NNNN.

    Usa un algoritmo de busqueda de subsecuencia optima para ser tolerante a ruido
    y segmentacion de caracteres extra (como el guion, tornillos o bordes).
    """
    # Solo alfanumericos, mayusculas
    clean = ''.join(c.upper() for c in raw if c.isalnum())

    if len(clean) < 4:
        return clean   # Demasiado corto para normalizar

    # Si la longitud es mayor a 7, buscar la subsecuencia optima de longitud 7
    if len(clean) > 7:
        import itertools
        best_candidate = clean[:7]
        best_score = -1

        # Generar todas las subsecuencias de longitud 7 en orden relativo original
        s_len = len(clean)
        indices = list(range(s_len))

        for comb in itertools.combinations(indices, 7):
            candidate = ''.join(clean[idx] for idx in comb)

            # Evaluar score del candidato
            score = 0
            for i, ch in enumerate(candidate):
                if i < 3:  # Letras
                    if ch.isalpha() or ch in _DIGIT_TO_LETTER:
                        score += 2 if ch.isalpha() else 1
                else:      # Digitos
                    if ch.isdigit() or ch in _LETTER_TO_DIGIT:
                        score += 2 if ch.isdigit() else 1

            if score > best_score:
                best_score = score
                best_candidate = candidate

        clean = best_candidate

    chars = list(clean)

    # Correccion posicional
    for i, ch in enumerate(chars):
        if i < 3:   # Posicion de letra
            if ch.isdigit():
                chars[i] = _DIGIT_TO_LETTER.get(ch, ch)
        else:       # Posicion de digito
            if ch.isalpha():
                chars[i] = _LETTER_TO_DIGIT.get(ch, ch)

    result = ''.join(chars)

    # Formatear con guion si tiene exactamente 7 chars
    if len(result) == 7:
        return result[:3] + '-' + result[3:]
    return result



class PlateOCR:
    """
    Reconocedor de placas vehiculares usando OpenCV + CNN personalizada.

    Uso:
        ocr = PlateOCR(model_path='char_cnn.pth')
        plate_str, debug_img = ocr.read_plate(frame, bbox)
    """

    def __init__(self, model_path: str = None, device: str = None):
        """
        Carga la CNN entrenada y configura el pipeline.

        Args:
            model_path (str | None): Ruta al archivo .pth con los pesos entrenados.
                                     Si es None, busca 'char_cnn_synthetic.pth' y luego 'char_cnn.pth'.
            device (str | None): 'cuda', 'cpu', o None (auto-detecta CUDA).
        """
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        # Si no se especifica ruta, resolver automaticamente priorizando la CNN sintetica
        if model_path is None:
            possible_names = ['char_cnn_synthetic.pth', 'char_cnn.pth']
            for name in possible_names:
                # Probar ruta relativa al archivo plate_ocr.py
                local_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', name)
                if os.path.exists(local_path):
                    model_path = local_path
                    break
                # Probar ruta en el directorio de trabajo actual
                if os.path.exists(name):
                    model_path = name
                    break
            if model_path is None:
                model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'char_cnn_synthetic.pth')

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

        # Detector de placa con YOLOv8 (con fallback heuristico)
        self.plate_detector = PlateDetector(conf_threshold=0.25)

        # Segmentador de caracteres (OpenCV clasico)
        self.segmenter = PlateSegmenter(target_size=(28, 28))

        # Transformacion de preprocesado para la CNN
        # NOTA: Las imagenes de EMNIST en el dataset ya estaban transpuestas, por eso
        # el entrenamiento aplica transpose_image para corregirlas y el modelo aprendio
        # a clasificar caracteres en orientacion NORMAL.
        # Para inferencia sobre placas reales (caracteres en orientacion correcta)
        # NO se aplica la transposicion: el modelo ya espera caracteres normales.
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1751,), (0.3277,)),
        ])

        # Umbral de confianza minimo por caracter (0-1)
        # Bajado a 0.15 para no descartar caracteres validos en placas reales
        # (EMNIST entrenado en tipografias standard puede tener menor confianza en fuentes de placa)
        self.char_conf_threshold = 0.15

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

        # Localizar region de placa con YOLOv8 preentrenado (o fallback heuristico)
        plate_crop, _plate_bbox = self.plate_detector.find_plate(vehicle_crop)
        if plate_crop is None or plate_crop.size == 0:
            return '', None

        # Segmentar caracteres
        char_imgs, debug_img = self.segmenter.segment_characters(plate_crop)

        if not char_imgs:
            return '', debug_img

        # Clasificar cada caracter (con confianza)
        plate_chars = []
        for char_img in char_imgs:
            char, conf = self._classify_char(char_img)
            if conf >= self.char_conf_threshold:
                plate_chars.append(char)

        raw_str  = ''.join(plate_chars).upper()
        # Aplicar normalizacion de formato (placa ecuatoriana AAA-NNNN)
        plate_str = normalize_plate(raw_str)
        return plate_str, debug_img
