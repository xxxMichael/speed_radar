"""
Test de reconocimiento de placas en tiempo real via camara.

Modo de uso:
  1. Coloca la placa dentro del recuadro verde que aparece en pantalla.
  2. Presiona ESPACIO para capturar y analizar la placa.
  3. El resultado del OCR se muestra en pantalla y en consola.
  4. Presiona 'c' para modo continuo (analiza cada 1 segundo automaticamente).

Controles:
    ESPACIO   -> Capturar y reconocer placa en el recuadro
    C         -> Activar/desactivar modo continuo
    D         -> Mostrar/ocultar imagen de debug (segmentacion de caracteres)
    +/-       -> Agrandar/achicar el recuadro de captura
    S         -> Guardar screenshot del frame actual
    Q / ESC   -> Salir
"""

import cv2
import time
import sys
import os
import numpy as np

# Rutas de importacion
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from ocr.plate_ocr import PlateOCR
from ocr.plate_detector import PlateDetector


# =============================================================================
# CONFIGURACION
# =============================================================================
CAMERA_INDEX    = 1             # DroidCam USB. Cambiar a 0 para webcam integrada.
CAMERA_BACKEND  = cv2.CAP_DSHOW

# Ruta al modelo CNN entrenado (prioriza el modelo sintético si existe)
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'char_cnn_synthetic.pth')
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'char_cnn.pth')

WINDOW_NAME      = "Test OCR de Placa  (Q=salir)"
WINDOW_DEBUG     = "Debug - Segmentacion de Caracteres"

# Ancho inicial del recuadro de captura en pixeles (alto proporcional ~1:3)
ROI_WIDTH_INIT  = 320
ROI_ASPECT      = 3.2           # ancho / alto de una placa tipica
# =============================================================================


class PlateOCRTest:
    """
    Herramienta de prueba del reconocimiento de placas via camara.
    Muestra un recuadro de captura ajustable y ejecuta el OCR sobre esa region.
    """

    def __init__(self):
        self.ocr         = None
        self.roi_w       = ROI_WIDTH_INIT
        self.roi_h       = max(30, int(self.roi_w / ROI_ASPECT))

        # Ultimo resultado de OCR
        self.last_plate  = ''
        self.last_confs  = []     # Confianza por caracter (no usado visualmente aqui)
        self.last_debug  = None   # Imagen de debug del segmentador
        self.result_time = 0.0    # Cuando se obtuvo el resultado (para FadeOut)

        # Historial de lecturas
        self.history: list[str] = []

        # Modo continuo: analiza cada segundo
        self.continuous      = False
        self.last_auto_time  = 0.0
        self.auto_interval   = 1.0   # segundos entre capturas automaticas

        # Mostrar debug
        self.show_debug = True

        self.screenshot_n = 0
        self.fps_counter  = 0
        self.fps_display  = 0.0
        self.fps_timer    = time.time()

        # Detector de posicion de placa con YOLOv8
        self.plate_detector: PlateDetector | None = None   # Se inicializa en run()
        # Ultimo bbox detectado de la placa RELATIVO al roi (para dibujarlo en pantalla)
        self.last_plate_bbox_in_frame: list | None = None

    # ------------------------------------------------------------------
    # OCR sobre una region del frame
    # ------------------------------------------------------------------

    def _analyze_roi(self, frame: np.ndarray, roi_rect: tuple) -> str:
        """
        Ejecuta la deteccion de placa con YOLOv8/heuristico y el OCR sobre la region roi_rect.

        1. Recorta el ROI del frame.
        2. Usa PlateDetector para localizar la placa dentro del ROI.
        3. Segmenta y clasifica los caracteres con la CNN OCR.

        Args:
            frame: Frame BGR completo.
            roi_rect: (x1, y1, x2, y2) de la region de captura.

        Returns:
            str: Texto de la placa reconocida.
        """
        rx1, ry1, rx2, ry2 = roi_rect
        roi_crop = frame[ry1:ry2, rx1:rx2].copy()

        if roi_crop.size == 0:
            return ''

        # --- Paso 1: Detectar bbox exacto de la placa dentro del ROI ---
        plate_crop = None
        self.last_plate_bbox_in_frame = None

        if self.plate_detector is not None:
            result = self.plate_detector.find_plate(roi_crop)
            plate_crop_det, plate_bbox_roi = result
            if plate_crop_det is not None and plate_crop_det.size > 0:
                plate_crop = plate_crop_det
                # Convertir bbox relativo al ROI a coordenadas absolutas del frame
                if plate_bbox_roi is not None:
                    px1, py1, px2, py2 = plate_bbox_roi
                    self.last_plate_bbox_in_frame = [
                        rx1 + px1, ry1 + py1,
                        rx1 + px2, ry1 + py2
                    ]

        # Si no se localizo placa, usar el ROI completo directamente
        if plate_crop is None:
            plate_crop = roi_crop

        # --- Paso 2: Segmentacion de caracteres con OpenCV ---
        char_imgs, debug_img = self.ocr.segmenter.segment_characters(plate_crop)
        self.last_debug = debug_img

        if not char_imgs:
            return ''

        # --- Paso 3: Clasificacion CNN por caracter ---
        plate_chars = []
        for char_img in char_imgs:
            char, conf = self.ocr._classify_char(char_img)
            if conf >= self.ocr.char_conf_threshold:
                plate_chars.append((char, conf))

        plate_str = ''.join(c for c, _ in plate_chars).upper()
        return plate_str

    # ------------------------------------------------------------------
    # Renderizado
    # ------------------------------------------------------------------

    def _get_roi_rect(self, frame_w: int, frame_h: int) -> tuple:
        """Calcula las coordenadas del recuadro centrado en el frame."""
        cx = frame_w // 2
        cy = frame_h // 2
        x1 = cx - self.roi_w // 2
        y1 = cy - self.roi_h // 2
        x2 = x1 + self.roi_w
        y2 = y1 + self.roi_h
        return (
            max(0, x1), max(0, y1),
            min(frame_w, x2), min(frame_h, y2)
        )

    def _draw_roi_box(self, frame: np.ndarray, roi_rect: tuple, active: bool):
        """Dibuja el recuadro de captura con esquinas marcadas."""
        x1, y1, x2, y2 = roi_rect
        color  = (0, 255, 0) if not active else (0, 200, 255)
        thick  = 2
        corner = 20   # Longitud de cada esquina

        # Esquinas
        for px, py, dx, dy in [
            (x1, y1,  1,  1),
            (x2, y1, -1,  1),
            (x1, y2,  1, -1),
            (x2, y2, -1, -1),
        ]:
            cv2.line(frame, (px, py), (px + dx * corner, py), color, thick + 1)
            cv2.line(frame, (px, py), (px, py + dy * corner), color, thick + 1)

        # Borde completo semitransparente
        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 1)
        cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)

        # Etiqueta
        label = "CAPTURA CONTINUA" if self.continuous else "Coloca la placa aqui"
        cv2.putText(frame, label, (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)

    def _draw_result(self, frame: np.ndarray):
        """Muestra el resultado del OCR en la parte inferior del frame."""
        h, w = frame.shape[:2]

        # Fondo del resultado
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h - 80), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

        if self.last_plate:
            # Tiempo transcurrido desde el ultimo resultado (FadeOut visual)
            age   = time.time() - self.result_time
            alpha = max(0.0, 1.0 - age / 8.0)   # Desvanece en 8 segundos

            r = int(255 * alpha)
            g = int(220 * alpha)
            b = 0

            cv2.putText(frame, "PLACA DETECTADA:",
                        (10, h - 55), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (150, 150, 150), 1)
            cv2.putText(frame, self.last_plate,
                        (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (b, g, r), 3)
        else:
            cv2.putText(frame, "Sin resultado - presiona ESPACIO o activa continuo [C]",
                        (10, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (100, 100, 100), 1)

        # Historial (ultimas 4 lecturas)
        if self.history:
            cv2.putText(frame, "Historial: " + "  |  ".join(self.history[-4:]),
                        (10, h - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (120, 120, 120), 1)

    def _draw_hud(self, frame: np.ndarray):
        """Panel de estado en la esquina superior derecha."""
        h, w = frame.shape[:2]
        panel_x = w - 270

        overlay = frame.copy()
        cv2.rectangle(overlay, (panel_x - 5, 0), (w, 125), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

        cont_color = (0, 255, 100) if self.continuous else (130, 130, 130)
        dbg_color  = (0, 200, 255) if self.show_debug  else (130, 130, 130)

        row, step = 20, 21
        cv2.putText(frame, f"FPS: {self.fps_display:.1f}",
                    (panel_x, row), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (200, 200, 200), 1); row += step
        cv2.putText(frame, f"ROI: {self.roi_w}x{self.roi_h} px  [+/-]",
                    (panel_x, row), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1); row += step
        cv2.putText(frame, f"Continuo: {'ON' if self.continuous else 'OFF'}  [C]",
                    (panel_x, row), cv2.FONT_HERSHEY_SIMPLEX, 0.48, cont_color, 1); row += step
        cv2.putText(frame, f"Debug seg: {'ON' if self.show_debug else 'OFF'}  [D]",
                    (panel_x, row), cv2.FONT_HERSHEY_SIMPLEX, 0.48, dbg_color, 1); row += step
        cv2.putText(frame, "[ESPACIO] Capturar",
                    (panel_x, row), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 255, 180), 1); row += step
        cv2.putText(frame, "[S] Screenshot  [Q] Salir",
                    (panel_x, row), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (150, 150, 150), 1)

    def _show_debug_window(self):
        """Muestra la imagen de debug del segmentador en una ventana separada."""
        if not self.show_debug or self.last_debug is None:
            return

        debug = self.last_debug
        # Escalar para visualizacion si es muy pequena
        dh, dw = debug.shape[:2]
        if dw < 300:
            scale = 300 / dw
            debug = cv2.resize(debug, (int(dw * scale), int(dh * scale)),
                               interpolation=cv2.INTER_NEAREST)

        cv2.imshow(WINDOW_DEBUG, debug)

    # ------------------------------------------------------------------
    # Bucle principal
    # ------------------------------------------------------------------

    def run(self):
        print("\n" + "=" * 60)
        print("  TEST DE RECONOCIMIENTO DE PLACAS (OCR)")
        print("=" * 60)
        print(f"  Camara     : indice {CAMERA_INDEX}")
        print(f"  Modelo CNN : {MODEL_PATH}")
        print("-" * 60)
        print("  1. Coloca la placa dentro del recuadro verde")
        print("  2. Presiona ESPACIO para leer la placa")
        print("  3. Presiona C para modo continuo (1 lectura/segundo)")
        print("  4. Presiona D para ver la segmentacion de caracteres")
        print("=" * 60 + "\n")

        # Abrir camara
        cap = cv2.VideoCapture(CAMERA_INDEX, CAMERA_BACKEND)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not cap.isOpened():
            print(f"[ERROR] No se pudo abrir la camara {CAMERA_INDEX}.")
            return

        fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"[INFO] Camara abierta: {fw}x{fh}")

        # Cargar OCR
        print(f"[INFO] Cargando CNN OCR desde: {MODEL_PATH}")
        self.ocr = PlateOCR(model_path=MODEL_PATH)
        print("[OK] OCR listo.")

        # Cargar detector de placa (YOLOv8 fine-tuned)
        print("[INFO] Iniciando detector de posicion de placa (YOLOv8)...")
        self.plate_detector = PlateDetector(conf_threshold=0.25)
        print("[OK] PlateDetector listo.\n")

        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, min(fw, 1280), min(fh, 720))

        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                time.sleep(0.05)
                continue

            now = time.time()

            # FPS
            self.fps_counter += 1
            if now - self.fps_timer >= 1.0:
                self.fps_display = self.fps_counter / (now - self.fps_timer)
                self.fps_counter = 0
                self.fps_timer   = now

            roi_rect = self._get_roi_rect(fw, fh)

            # Modo continuo: analizar automaticamente
            if self.continuous and (now - self.last_auto_time) >= self.auto_interval:
                self.last_auto_time = now
                plate = self._analyze_roi(frame, roi_rect)
                if plate:
                    self.last_plate  = plate
                    self.result_time = now
                    self.history.append(plate)
                    ts = time.strftime("%H:%M:%S")
                    print(f"[OCR] {ts}  Placa: {plate}")

            # Dibujar
            self._draw_roi_box(frame, roi_rect, active=self.continuous)

            # Dibujar bbox de la placa detectada por YOLOv8 (amarillo brillante)
            if self.last_plate_bbox_in_frame is not None:
                px1, py1, px2, py2 = self.last_plate_bbox_in_frame
                cv2.rectangle(frame, (px1, py1), (px2, py2), (0, 220, 255), 2)
                cv2.putText(frame, "PLACA DETECTADA",
                            (px1, max(0, py1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 255), 2)

            self._draw_result(frame)
            self._draw_hud(frame)

            cv2.imshow(WINDOW_NAME, frame)
            self._show_debug_window()

            # Teclado
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q') or key == 27:
                break

            elif key == ord(' '):           # ESPACIO: captura manual
                plate = self._analyze_roi(frame, roi_rect)
                if plate:
                    self.last_plate  = plate
                    self.result_time = now
                    self.history.append(plate)
                    ts = time.strftime("%H:%M:%S")
                    print(f"[OCR] {ts}  Placa: {plate}")
                    print(f"      Chars detectados: {len(plate)}")
                else:
                    self.last_plate = ''
                    print(f"[OCR] No se detectaron caracteres. Ajusta la posicion de la placa.")

            elif key == ord('c'):           # C: modo continuo
                self.continuous = not self.continuous
                print(f"[INFO] Modo continuo: {'ON' if self.continuous else 'OFF'}")

            elif key == ord('d'):           # D: debug
                self.show_debug = not self.show_debug
                if not self.show_debug:
                    cv2.destroyWindow(WINDOW_DEBUG)
                print(f"[INFO] Debug segmentacion: {'ON' if self.show_debug else 'OFF'}")

            elif key == ord('+') or key == ord('='):
                self.roi_w = min(fw - 20, self.roi_w + 20)
                self.roi_h = max(30, int(self.roi_w / ROI_ASPECT))
                print(f"[INFO] ROI: {self.roi_w}x{self.roi_h} px")

            elif key == ord('-'):
                self.roi_w = max(80, self.roi_w - 20)
                self.roi_h = max(30, int(self.roi_w / ROI_ASPECT))
                print(f"[INFO] ROI: {self.roi_w}x{self.roi_h} px")

            elif key == ord('s'):           # S: screenshot
                fname = f"placa_test_{self.screenshot_n:03d}.jpg"
                cv2.imwrite(fname, frame)
                print(f"[INFO] Screenshot guardado: {fname}")
                self.screenshot_n += 1

        cap.release()
        cv2.destroyAllWindows()

        print("\n" + "=" * 60)
        print(f"  Lecturas totales: {len(self.history)}")
        if self.history:
            print(f"  Placas leidas : {' | '.join(self.history)}")
        print("=" * 60)


if __name__ == "__main__":
    PlateOCRTest().run()
