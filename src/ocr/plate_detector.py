"""
Detector de regiones de placa vehicular usando YOLOv8 preentrenado.

El modelo 'keremberke/yolov8n-license-plate-detection' fue entrenado
especificamente para detectar el bounding box exacto de la matricula/placa
dentro de la imagen de un vehiculo.

Pipeline:
  1. Recibe el recorte de la imagen del vehiculo (bbox del auto ya extraido).
  2. Corre inferencia con YOLOv8 para localizar la(s) placa(s).
  3. Devuelve el recorte BGR de la placa con mayor confianza.
  4. Si no detecta ninguna placa, cae en la heuristica clasica como fallback.
"""

import os
import cv2
import numpy as np
import threading
import torch

try:
    from ultralytics import YOLO
    _YOLO_AVAILABLE = True
except ImportError:
    _YOLO_AVAILABLE = False
    print("[PlateDetector] WARNING: ultralytics no esta instalado. Se usara fallback heuristico.")


# Nombre del modelo en el hub de Ultralytics/HuggingFace
# Al ser la primera ejecucion, se descargara automaticamente (~6 MB)
_MODEL_REPO = "yasirfaizahmed/license-plate-object-detection"
_MODEL_ALIAS = "best.pt"

# Ruta local donde se cachea el modelo descargado
_CACHE_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'models')
_CACHE_PATH = os.path.join(_CACHE_DIR, 'plate_detector_yolov8n.pt')


class PlateDetector:
    """
    Localiza con precision el bounding box de la matricula/placa dentro
    del recorte del vehiculo usando YOLOv8 fine-tuned para placas.

    Uso:
        detector = PlateDetector()
        plate_crop, plate_bbox = detector.find_plate(vehicle_crop)
    """

    def __init__(self, conf_threshold: float = 0.25):
        """
        Inicializa el detector y descarga el modelo si no existe en cache.

        Args:
            conf_threshold: Confianza minima (0-1) para aceptar una deteccion.
                            Valores bajos detectan mas placas con mas falsos positivos.
                            Valores altos son mas estrictos.
        """
        self.conf_threshold = conf_threshold
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = None
        self._loaded = False
        self._is_base_model = False  # True cuando se usa yolov8n.pt en lugar del modelo fine-tuned
        self.lock = threading.Lock()

        if not _YOLO_AVAILABLE:
            print("[PlateDetector] ultralytics no disponible, usando fallback heuristico.")
            return

        # Intentar cargar modelo desde cache local primero
        if os.path.exists(_CACHE_PATH):
            self._load_model(_CACHE_PATH)
        
        # Si no cargo nada, o si cargo el modelo base COCO (no especializado), intentamos descargar
        if not self._loaded or self._is_base_model:
            self._download_and_cache()


    def _download_and_cache(self):
        """
        Descarga el modelo de deteccion de placas con 3 estrategias en orden:
          1. HuggingFace Hub via huggingface_hub (requiere token si el repo es privado).
          2. Descarga directa via urllib desde HuggingFace CDN (sin token, repos publicos).
          3. Reutilizar yolov8n.pt del proyecto como base (siempre funciona — fallback final).
        """
        print("[PlateDetector] Buscando modelo de deteccion de placas...")
        os.makedirs(_CACHE_DIR, exist_ok=True)

        # --- Estrategia 1: huggingface_hub con HF_TOKEN si esta definido ---
        hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
        if hf_token:
            try:
                from huggingface_hub import hf_hub_download
                print(f"  [1/3] Descargando desde HuggingFace Hub con token...")
                path = hf_hub_download(
                    repo_id=_MODEL_REPO,
                    filename=_MODEL_ALIAS,
                    local_dir=_CACHE_DIR,
                    token=hf_token,
                )
                import shutil
                if os.path.abspath(path) != os.path.abspath(_CACHE_PATH):
                    shutil.copy2(path, _CACHE_PATH)
                print("[PlateDetector] Descarga via HuggingFace Hub completada.")
                self._load_model(_CACHE_PATH)
                return
            except Exception as e:
                print(f"  [1/3] Fallo: {str(e)[:80]}")

        # --- Estrategia 2: URL directa sin token (HuggingFace CDN publico) ---
        try:
            import urllib.request
            # URL directa al archivo best.pt sin autenticacion
            url = (
                f"https://huggingface.co/{_MODEL_REPO}"
                f"/resolve/main/{_MODEL_ALIAS}"
            )
            print(f"  [2/3] Descarga directa desde: {url}")
            urllib.request.urlretrieve(url, _CACHE_PATH)
            if os.path.getsize(_CACHE_PATH) > 1_000_000:  # > 1 MB => valido
                print("[PlateDetector] Descarga directa completada.")
                self._load_model(_CACHE_PATH)
                return
            else:
                os.remove(_CACHE_PATH)
                print("  [2/3] Archivo descargado muy pequeno, posiblemente pagina de error.")
        except Exception as e:
            print(f"  [2/3] Fallo: {str(e)[:80]}")

        # --- Estrategia 3: Usar yolov8n.pt existente en el proyecto ---
        # Este modelo base de YOLOv8 no detecta placas como clase separada, pero
        # el modulo lo cargara como base para no crashear. El fallback heuristico
        # tomara el relevo en find_plate() cuando YOLO no encuentre 'license_plate'.
        print("  [3/3] Usando yolov8n.pt del proyecto como modelo base...")
        local_candidates = [
            os.path.join(os.path.dirname(__file__), '..', '..', 'yolov8n.pt'),
            os.path.join(os.getcwd(), 'yolov8n.pt'),
        ]
        for candidate in local_candidates:
            candidate = os.path.normpath(candidate)
            if os.path.exists(candidate):
                import shutil
                shutil.copy2(candidate, _CACHE_PATH)
                print(f"  [3/3] Copiado desde: {candidate}")
                self._load_model(_CACHE_PATH)
                # Marcar que es el modelo base (no fine-tuned para placas)
                self._is_base_model = True
                return

        print("[PlateDetector] No se encontro ningun modelo. Se usara solo el fallback heuristico.")

    def _load_model(self, path: str):
        """Carga el modelo YOLO desde un archivo .pt local."""
        try:
            self.model = YOLO(path)
            self._loaded = True
            print(f"[PlateDetector] Modelo cargado: {os.path.basename(path)} en dispositivo: {self.device}")

            # Verificar si el modelo tiene clases de deteccion de placas
            # Si solo tiene clases COCO (80) sin 'license_plate', es el modelo base
            names = self.model.names if self.model.names else {}
            has_plate_class = any(
                'plate' in n.lower() or 'license' in n.lower() or 'matricula' in n.lower()
                for n in names.values()
            )
            if not has_plate_class:
                self._is_base_model = True
                print(f"[PlateDetector] Modelo base COCO detectado ({len(names)} clases, sin clase 'plate').")
                print(f"[PlateDetector] Se usara fallback heuristico para localizar placas.")
            else:
                self._is_base_model = False
                print(f"[PlateDetector] Modelo fine-tuned para placas detectado. Clases: {list(names.values())}")
        except Exception as e:
            print(f"[PlateDetector] WARNING: No se pudo cargar el modelo en {path}: {e}")
            print("[PlateDetector] Se usara fallback heuristico.")

    def find_plate(self, vehicle_crop: np.ndarray) -> tuple[np.ndarray | None, list | None]:
        """
        Detecta la region de la placa dentro del recorte del vehiculo.

        Args:
            vehicle_crop: Imagen BGR del vehiculo (ya recortada del frame principal).

        Returns:
            tuple:
                - np.ndarray | None: Imagen BGR recortada de la placa con mayor confianza.
                                     None si no se detecto ninguna placa y el fallback tambien fallo.
                - list | None: Bounding box [x1, y1, x2, y2] de la placa RELATIVO al crop,
                               o None si no se detecto.
        """
        if vehicle_crop is None or vehicle_crop.size == 0:
            self.last_detection_method = "None"
            self.last_detection_conf = 0.0
            return None, None

        h, w = vehicle_crop.shape[:2]
        if h < 10 or w < 10:
            self.last_detection_method = "None"
            self.last_detection_conf = 0.0
            return None, None

        # --- Deteccion con YOLOv8 ---
        if self._loaded and self.model is not None:
            plate_crop, plate_bbox = self._detect_with_yolo(vehicle_crop)
            if plate_crop is not None:
                return plate_crop, plate_bbox
            else:
                # Si tenemos el modelo especializado y no detectó nada, confiamos en YOLO y abortamos
                if not self._is_base_model:
                    self.last_detection_method = "None (YOLO rejected)"
                    self.last_detection_conf = 0.0
                    return None, None

        # --- Fallback heuristico (Solo si YOLO no está cargado o es el modelo base sin entrenamiento) ---
        return self._heuristic_fallback(vehicle_crop)

    def _detect_with_yolo(self, vehicle_crop: np.ndarray) -> tuple[np.ndarray | None, list | None]:
        """
        Corre el modelo YOLOv8 sobre el recorte del vehiculo.
        Devuelve el recorte con mayor confianza si supera el umbral.
        Si el modelo cargado es el modelo base (yolov8n.pt sin fine-tuning para placas),
        retorna None para que el fallback heuristico tome el relevo.
        """
        # Si es el modelo base sin fine-tuning para placas, no ejecutar YOLO
        # (detectaria objetos COCO sin relacion con placas vehiculares)
        if self._is_base_model:
            return None, None

        try:
            with self.lock:
                results = self.model.predict(
                    vehicle_crop,
                    conf=self.conf_threshold,
                    verbose=False,
                    imgsz=480,    # Reducido de 640 a 480 para optimizar tráfico cercano/lento
                    device=self.device
                )

            if not results or results[0].boxes is None:
                return None, None

            boxes = results[0].boxes
            if len(boxes) == 0:
                return None, None

            h, w = vehicle_crop.shape[:2]

            # Tomar la deteccion con mayor confianza (en lugar de mayor area)
            best_crop = None
            best_bbox = None
            best_conf_score = 0.0

            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0])

                if conf < self.conf_threshold:
                    continue

                if conf > best_conf_score:
                    best_conf_score = conf
                    # Padding de 2px para no cortar bordes de caracteres
                    pad = 2
                    rx1 = max(0, x1 - pad)
                    ry1 = max(0, y1 - pad)
                    rx2 = min(w, x2 + pad)
                    ry2 = min(h, y2 + pad)
                    best_crop = vehicle_crop[ry1:ry2, rx1:rx2]
                    best_bbox = [rx1, ry1, rx2, ry2]

            if best_crop is not None:
                self.last_detection_method = "YOLO"
                self.last_detection_conf = best_conf_score
            return best_crop, best_bbox

        except Exception as e:
            print(f"[PlateDetector] Error durante inferencia YOLO: {e}")
            return None, None

    def _heuristic_fallback(self, vehicle_crop: np.ndarray) -> tuple[np.ndarray, list | None]:
        """
        Fallback heuristico mejorado: detecta la region rectangular brillante (placa)
        dentro del recorte del vehiculo usando multiples estrategias.

        Returns:
            tuple: (plate_crop, plate_bbox_or_None)
        """
        h, w = vehicle_crop.shape[:2]

        # --- Estrategia 1: Deteccion de rectangulo blanco/claro con aspecto de placa ---
        gray = cv2.cvtColor(vehicle_crop, cv2.COLOR_BGR2GRAY)
        # CLAHE para normalizar iluminacion
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray  = clahe.apply(gray)

        # Umbral: zonas muy claras (fondo blanco de la placa ecuatoriana)
        _, bright = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
        # Operaciones morfologicas para unir la zona blanca de la placa
        kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
        bright   = cv2.morphologyEx(bright, cv2.MORPH_CLOSE, kernel_h)

        contours, _ = cv2.findContours(bright, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best_crop  = None
        best_bbox  = None
        best_score = 0.0

        for cnt in contours:
            x, y, cw, ch = cv2.boundingRect(cnt)
            if cw < 30 or ch < 8:
                continue
            aspect = cw / float(ch)
            area   = cw * ch
            if 2.0 <= aspect <= 8.0 and area > h * w * 0.005:
                # Score: aspecto cercano a 4:1 y area grande
                aspect_score = 1.0 - abs(aspect - 4.0) / 4.0
                area_score   = min(1.0, area / (h * w * 0.10))
                score        = aspect_score * 0.5 + area_score * 0.5
                if score > best_score:
                    best_score = score
                    pad = 5
                    bx1 = max(0, x - pad);  by1 = max(0, y - pad)
                    bx2 = min(w, x+cw+pad); by2 = min(h, y+ch+pad)
                    best_crop = vehicle_crop[by1:by2, bx1:bx2]
                    best_bbox = [bx1, by1, bx2, by2]

        if best_crop is not None:
            self.last_detection_method = "Heuristic (Threshold)"
            self.last_detection_conf = 0.0
            return best_crop, best_bbox

        # --- Estrategia 2: Busqueda de bordes en la mitad inferior ---
        search_y = max(0, int(h * 0.35))
        roi      = vehicle_crop[search_y:, :]
        roi_h, roi_w = roi.shape[:2]

        gray2  = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blur2  = cv2.GaussianBlur(gray2, (5, 5), 0)
        edges  = cv2.Canny(blur2, 25, 150)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 3))
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        contours2, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best_score2 = 0
        for cnt in contours2:
            x, y, cw, ch = cv2.boundingRect(cnt)
            if ch < 6 or cw < 15:
                continue
            aspect = cw / float(ch)
            area   = cw * ch
            if 1.5 <= aspect <= 8.0 and area > roi_h * roi_w * 0.002:
                aspect_score = 1.0 - abs(aspect - 4.0) / 4.0
                area_score   = area / (roi_h * roi_w)
                score        = aspect_score * 0.45 + area_score * 0.55
                if score > best_score2:
                    best_score2 = score
                    pad = 4
                    x1 = max(0, x-pad);    y1 = max(0, y-pad)
                    x2 = min(roi_w, x+cw+pad); y2 = min(roi_h, y+ch+pad)
                    best_crop  = roi[y1:y2, x1:x2]
                    best_bbox  = [x1, search_y+y1, x2, search_y+y2]

        if best_crop is not None:
            self.last_detection_method = "Heuristic (Edges)"
            self.last_detection_conf = 0.0
            return best_crop, best_bbox

        # --- Fallback final: franja inferior del vehiculo ---
        fallback_h = max(20, h // 4)
        fc = vehicle_crop[h-fallback_h:, :]
        
        self.last_detection_method = "Heuristic (Blind Crop)"
        self.last_detection_conf = 0.0
        return fc, None

    def draw_detection(self, vehicle_crop: np.ndarray, plate_bbox: list | None) -> np.ndarray:
        """
        Dibuja el bounding box de la placa detectada sobre el recorte del vehiculo.
        Util para visualizacion/debug.

        Args:
            vehicle_crop: Imagen BGR del vehiculo.
            plate_bbox: [x1, y1, x2, y2] relativo al crop, o None.

        Returns:
            np.ndarray: Imagen con el bbox de la placa pintado en verde.
        """
        vis = vehicle_crop.copy()
        if plate_bbox is not None:
            x1, y1, x2, y2 = plate_bbox
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 80), 2)
            cv2.putText(vis, "PLACA", (x1, max(0, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 80), 1)
        return vis
