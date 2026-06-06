"""
Script de prueba del sistema de deteccion de vehiculos y calculo de velocidad.

Controles:
    Clic x2   -> Fijar las dos lineas de medicion
    L         -> Cambiar orientacion de lineas: VERTICAL (movimiento horizontal)
                 / HORIZONTAL (movimiento vertical)
    + / -     -> Aumentar/disminuir distancia real entre lineas (metros)
    . / ,     -> Aumentar/disminuir limite de velocidad (km/h)
    D         -> Activar/desactivar modo diagnostico (muestra todos los objetos)
    R         -> Resetear lineas, velocidades y contadores
    S         -> Guardar screenshot del frame actual
    Q         -> Salir
"""

import cv2
import time
import threading
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from detection.vehicle_tracker import VehicleTracker
from detection.speed_calculator import SpeedCalculator
from ocr.plate_ocr import PlateOCR


# =============================================================================
# CONFIGURACION INICIAL (ajustable en la interfaz)
# =============================================================================
CAMERA_INDEX       = 1          # DroidCam USB (cambiar a 0 si es webcam integrada)
CAMERA_BACKEND     = cv2.CAP_DSHOW

SPEED_LIMIT_KMH    = 30.0       # Limite inicial (ajustable con . y ,)
SPEED_LIMIT_STEP   = 5.0        # Cuanto cambia el limite con cada pulsacion

REAL_DISTANCE_M    = 5.0        # Distancia real inicial entre lineas (ajustable con + y -)
REAL_DISTANCE_STEP = 0.5

YOLO_CONF          = 0.15       # Umbral de confianza YOLO

WINDOW_NAME        = "Speed Radar - Deteccion (Q=salir)"
# =============================================================================


class InteractiveSpeedTest:
    """
    Herramienta interactiva de deteccion y medicion de velocidad.

    Soporta lineas de medicion VERTICALES (movimiento horizontal de vehiculos)
    y HORIZONTALES (movimiento vertical de vehiculos), intercambiables con 'L'.
    El limite de velocidad y la distancia real se ajustan desde la interfaz.
    """

    # Modos de orientacion de lineas
    MODE_VERTICAL   = 'vertical'    # Lineas | |  →  miden movimiento izquierda/derecha
    MODE_HORIZONTAL = 'horizontal'  # Lineas ═ ═  →  miden movimiento arriba/abajo

    def __init__(self, camera_index: int, real_distance_m: float, speed_limit_kmh: float):
        self.camera_index    = camera_index
        self.real_distance_m = real_distance_m
        self.speed_limit_kmh = speed_limit_kmh

        # Orientacion de las lineas de medicion
        self.line_mode = self.MODE_VERTICAL   # Por defecto: lineas verticales

        # Posicion de las dos lineas (en pixeles)
        # En modo VERTICAL  usan coordenada X (columna)
        # En modo HORIZONTAL usan coordenada Y (fila)
        self.line1_pos = None
        self.line2_pos = None
        self.click_n   = 0

        # Modulos de deteccion
        self.tracker    = None
        self.speed_calc = None

        # Historial de velocidades medidas: {track_id: speed_kmh}
        self.speeds: dict[int, float] = {}

        # Estado de diagnostico
        self.diag_mode      = False
        self.detected_count = 0

        # Metricas de FPS
        self.fps_display = 0.0
        self.fps_counter = 0
        self.fps_timer   = time.time()

        self.screenshot_n = 0

        # --- Infracciones ---
        self.infraction_log: list[dict] = []
        self.alert_until: float = 0.0

        # --- OCR de placas ---
        self.plate_ocr: PlateOCR | None = None   # Se inicializa en run()
        # Placas ya leidas: {track_id: plate_str}
        self.plates: dict[int, str] = {}
        # Set de track_ids cuyo OCR ya esta corriendo en hilo (evita duplicados)
        self._ocr_running: set[int] = set()

    # ------------------------------------------------------------------
    # OCR de placas (hilo en segundo plano)
    # ------------------------------------------------------------------

    def _run_ocr_async(self, tid: int, frame_copy: 'np.ndarray', bbox: list, speed: float):
        """
        Ejecuta el OCR de la placa en un hilo separado para no bloquear el video.
        Al terminar, registra la placa e imprime el resultado en consola.

        Args:
            tid: track_id del vehiculo.
            frame_copy: Copia del frame actual (para que no cambie mientras se procesa).
            bbox: Bounding box [x1, y1, x2, y2] del vehiculo.
            speed: Velocidad calculada en km/h.
        """
        try:
            plate, _ = self.plate_ocr.read_plate(frame_copy, bbox)
        except Exception as e:
            plate = ''
            print(f"[OCR] Error en vehiculo #{tid}: {e}")
        finally:
            self._ocr_running.discard(tid)

        plate_str = plate if plate else '???'
        self.plates[tid] = plate_str
        ts = time.strftime("%H:%M:%S")

        if speed > self.speed_limit_kmh:
            print(f"[INFRACCION] {ts} | Vehiculo #{tid} | Placa: {plate_str} | Velocidad: {speed:.1f} km/h")
        else:
            print(f"[REGISTRO]   {ts} | Vehiculo #{tid} | Placa: {plate_str}")

    # ------------------------------------------------------------------
    # Callback del raton
    # ------------------------------------------------------------------

    def _mouse_callback(self, event, x, y, flags, param):
        """Fija las lineas de medicion con dos clics. Usa X o Y segun el modo."""
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        # Coordenada relevante segun orientacion
        pos = x if self.line_mode == self.MODE_VERTICAL else y
        axis_label = "X" if self.line_mode == self.MODE_VERTICAL else "Y"

        if self.click_n == 0:
            self.line1_pos = pos
            self.click_n   = 1
            print(f"[INFO] Linea 1 fijada en {axis_label}={pos}px")

        elif self.click_n == 1 and pos != self.line1_pos:
            self.line2_pos = pos
            self.click_n   = 2
            sep = abs(self.line2_pos - self.line1_pos)
            print(f"[INFO] Linea 2 fijada en {axis_label}={pos}px  |  Separacion: {sep}px")
            print(f"[INFO] Distancia real: {self.real_distance_m} m")
            self._init_speed_calculator()

    # ------------------------------------------------------------------
    # Inicializacion del SpeedCalculator
    # ------------------------------------------------------------------

    def _init_speed_calculator(self):
        """Crea o reinicia el SpeedCalculator con los parametros actuales."""
        if self.line1_pos is None or self.line2_pos is None:
            return

        if self.line_mode == self.MODE_VERTICAL:
            axis      = 'x'
            direction = 'right'   # Vehiculos van de izquierda a derecha
        else:
            axis      = 'y'
            direction = 'down'    # Vehiculos van de arriba hacia abajo

        self.speed_calc = SpeedCalculator(
            line1_pos=self.line1_pos,
            line2_pos=self.line2_pos,
            distance_meters=self.real_distance_m,
            direction=direction,
            axis=axis,
        )
        self.speeds = {}
        lo = min(self.line1_pos, self.line2_pos)
        hi = max(self.line1_pos, self.line2_pos)
        print(f"[INFO] SpeedCalculator | modo={self.line_mode} | dist={self.real_distance_m}m | L1={lo}px | L2={hi}px")

    def _reset_lines(self):
        """Borra las lineas, velocidades y el calculador."""
        self.line1_pos  = None
        self.line2_pos  = None
        self.click_n    = 0
        self.speed_calc = None
        self.speeds     = {}
        print("[INFO] Lineas y velocidades reseteadas.")

    # ------------------------------------------------------------------
    # Renderizado del HUD
    # ------------------------------------------------------------------

    def _draw_lines(self, frame):
        """Dibuja las lineas de medicion segun el modo actual."""
        h, w = frame.shape[:2]

        if self.line_mode == self.MODE_VERTICAL:
            # Lineas VERTICALES (de arriba a abajo)
            if self.line1_pos is not None:
                cv2.line(frame, (self.line1_pos, 0), (self.line1_pos, h), (255, 130, 0), 2)
                cv2.putText(frame, "L1", (self.line1_pos + 5, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 130, 0), 2)
            if self.line2_pos is not None:
                cv2.line(frame, (self.line2_pos, 0), (self.line2_pos, h), (0, 60, 255), 2)
                cv2.putText(frame, f"L2 | {self.real_distance_m:.1f}m", (self.line2_pos + 5, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 60, 255), 2)
        else:
            # Lineas HORIZONTALES (de izquierda a derecha)
            if self.line1_pos is not None:
                cv2.line(frame, (0, self.line1_pos), (w, self.line1_pos), (255, 130, 0), 2)
                cv2.putText(frame, "L1", (8, self.line1_pos - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 130, 0), 2)
            if self.line2_pos is not None:
                cv2.line(frame, (0, self.line2_pos), (w, self.line2_pos), (0, 60, 255), 2)
                cv2.putText(frame, f"L2 | {self.real_distance_m:.1f}m", (8, self.line2_pos + 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 60, 255), 2)

    def _draw_hud(self, frame):
        """Dibuja el panel de informacion en la esquina superior derecha."""
        h, w = frame.shape[:2]

        # Fondo semitransparente
        panel_x = w - 310
        overlay = frame.copy()
        cv2.rectangle(overlay, (panel_x - 6, 0), (w, 175), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

        # Colores de estado
        ready_color = (0, 220, 0) if self.click_n == 2 else (0, 165, 255)
        det_color   = (0, 220, 0) if self.detected_count > 0 else (0, 60, 255)
        diag_color  = (0, 255, 255) if self.diag_mode else (130, 130, 130)
        mode_color  = (180, 255, 180)

        click_states = {
            0: "CLIC: fijar Linea 1",
            1: "CLIC: fijar Linea 2",
            2: "Midiendo velocidad...",
        }

        row = 22
        step = 22

        cv2.putText(frame, click_states[self.click_n],
                    (panel_x, row), cv2.FONT_HERSHEY_SIMPLEX, 0.55, ready_color, 1)
        row += step

        cv2.putText(frame, f"FPS: {self.fps_display:.1f}   conf: {YOLO_CONF}",
                    (panel_x, row), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (200, 200, 200), 1)
        row += step

        cv2.putText(frame, f"Vehiculos en frame: {self.detected_count}",
                    (panel_x, row), cv2.FONT_HERSHEY_SIMPLEX, 0.52, det_color, 1)
        row += step

        # Limite de velocidad con indicador de teclas
        cv2.putText(frame, f"Limite: {self.speed_limit_kmh:.0f} km/h  [, / .]",
                    (panel_x, row), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 200, 255), 1)
        row += step

        # Distancia real
        cv2.putText(frame, f"Dist real: {self.real_distance_m:.1f} m  [- / +]",
                    (panel_x, row), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (200, 200, 200), 1)
        row += step

        # Modo de lineas
        mode_label = "| |  VERTICALES" if self.line_mode == self.MODE_VERTICAL else "== HORIZONTALES"
        cv2.putText(frame, f"Lineas: {mode_label}  [L]",
                    (panel_x, row), cv2.FONT_HERSHEY_SIMPLEX, 0.50, mode_color, 1)
        row += step

        # Modo diagnostico
        cv2.putText(frame, f"Diagnostico: {'ON' if self.diag_mode else 'OFF'}  [D]",
                    (panel_x, row), cv2.FONT_HERSHEY_SIMPLEX, 0.50, diag_color, 1)
        row += step

        cv2.putText(frame, "[R] Reset   [S] Screenshot   [Q] Salir",
                    (panel_x, row), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (150, 150, 150), 1)

    def _draw_speed_labels(self, frame, tracked_vehicles):
        """Dibuja la velocidad calculada encima de cada vehiculo."""
        for vehicle in tracked_vehicles:
            tid    = vehicle['track_id']
            cx, cy = vehicle['centroid']
            speed  = self.speeds.get(tid)

            if speed is not None:
                over   = speed > self.speed_limit_kmh
                color  = (0, 0, 255) if over else (0, 220, 0)
                label  = f"{speed:.1f} km/h {'[!]' if over else ''}"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
                cv2.rectangle(frame, (cx - 4, cy - th - 12), (cx + tw + 4, cy - 2), (0, 0, 0), -1)
                cv2.putText(frame, label, (cx, cy - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

    def _draw_infraction_alert(self, frame):
        """
        Muestra un borde rojo parpadeante en todo el frame durante 3 segundos
        despues de detectar una infraccion.
        """
        if time.time() < self.alert_until:
            h, w = frame.shape[:2]
            thickness = 8
            # Parpadeo: visible cada 0.3s
            if int(time.time() * 4) % 2 == 0:
                cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 0, 255), thickness)
                # Banner superior
                cv2.rectangle(frame, (0, 0), (w, 50), (0, 0, 180), -1)
                cv2.putText(frame, "!! INFRACCION DE VELOCIDAD !!",
                            (w // 2 - 200, 35), cv2.FONT_HERSHEY_SIMPLEX,
                            0.9, (255, 255, 255), 2)

    def _draw_infraction_log(self, frame):
        """
        Muestra en la esquina inferior izquierda el historial de las ultimas
        5 infracciones detectadas en esta sesion.
        """
        if not self.infraction_log:
            return

        h, w = frame.shape[:2]
        log   = self.infraction_log[-5:]   # Mostrar solo las 5 mas recientes
        n     = len(log)
        box_h = n * 22 + 14
        y0    = h - box_h - 10

        # Fondo
        overlay = frame.copy()
        cv2.rectangle(overlay, (5, y0), (280, h - 10), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.60, frame, 0.40, 0, frame)

        cv2.putText(frame, "INFRACCIONES DETECTADAS",
                    (10, y0 + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 100, 255), 1)

        for i, inf in enumerate(reversed(log)):
            txt = f"  #{inf['id']:>3}  {inf['speed']:>6.1f} km/h   {inf['time']}"
            cv2.putText(frame, txt,
                        (10, y0 + 14 + (i + 1) * 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 60, 255), 1)

    # ------------------------------------------------------------------
    # Bucle principal
    # ------------------------------------------------------------------

    def run(self):
        print("\n" + "=" * 65)
        print("  SPEED RADAR - TEST INTERACTIVO")
        print("=" * 65)
        print(f"  Camara         : indice {self.camera_index}")
        print(f"  Limite inicial : {self.speed_limit_kmh} km/h  (ajustar con , / .)")
        print(f"  Distancia init : {self.real_distance_m} m   (ajustar con - / +)")
        print(f"  YOLO conf      : {YOLO_CONF}")
        print("-" * 65)
        print("  CONTROLES:")
        print("    Clic x2  -> Fijar lineas de medicion")
        print("    L        -> Cambiar orientacion lineas (VERTICAL / HORIZONTAL)")
        print("    + / -    -> Distancia real entre lineas")
        print("    . / ,    -> Limite de velocidad")
        print("    D        -> Modo diagnostico (ver todos los objetos)")
        print("    R        -> Resetear")
        print("    S        -> Screenshot")
        print("    Q        -> Salir")
        print("=" * 65 + "\n")

        # Abrir camara
        cap = cv2.VideoCapture(self.camera_index, CAMERA_BACKEND)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not cap.isOpened():
            print(f"[ERROR] No se pudo abrir la camara {self.camera_index}.")
            return

        fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"[INFO] Camara abierta: {fw}x{fh}")

        # Cargar YOLO
        print(f"[INFO] Cargando YOLOv8n (conf={YOLO_CONF})...")
        self.tracker = VehicleTracker(model_path='yolov8n.pt', conf=YOLO_CONF)
        print("[OK] YOLOv8n listo.")

        # Cargar OCR de placas
        # El modelo esta en src/ (donde se ejecuto el entrenamiento)
        ocr_model_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), '..', 'char_cnn.pth'
        )
        self.plate_ocr = PlateOCR(model_path=ocr_model_path)
        print("[OK] PlateOCR listo.\n")

        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, min(fw, 1280), min(fh, 720))
        cv2.setMouseCallback(WINDOW_NAME, self._mouse_callback)

        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                time.sleep(0.05)
                continue

            current_time = time.time()

            # Calcular FPS
            self.fps_counter += 1
            elapsed = current_time - self.fps_timer
            if elapsed >= 1.0:
                self.fps_display = self.fps_counter / elapsed
                self.fps_counter = 0
                self.fps_timer   = current_time

            # Deteccion
            if self.diag_mode:
                self.detected_count, annotated_frame = self.tracker.detect_all(frame)
                tracked_vehicles = []
            else:
                tracked_vehicles, annotated_frame = self.tracker.process_frame(frame)
                self.detected_count = len(tracked_vehicles)

            # Calculo de velocidad y disparo de OCR
            if self.speed_calc is not None and self.click_n == 2:
                new_speeds = self.speed_calc.update(tracked_vehicles, current_time)
                self.speeds.update(new_speeds)

                for tid, spd in new_speeds.items():
                    ts = time.strftime("%H:%M:%S")

                    if spd > self.speed_limit_kmh:
                        self.alert_until = current_time + 3.0
                        self.infraction_log.append({
                            'id':    tid,
                            'speed': spd,
                            'time':  ts,
                            'plate': '...',   # Se actualiza cuando termina el OCR
                        })

                    # Lanzar OCR en hilo separado si no hay uno ya corriendo para este ID
                    if self.plate_ocr is not None and tid not in self._ocr_running:
                        # Buscar el bbox del vehiculo en la lista actual
                        vehicle_data = next(
                            (v for v in tracked_vehicles if v['track_id'] == tid), None
                        )
                        if vehicle_data is not None:
                            self._ocr_running.add(tid)
                            frame_copy = frame.copy()
                            t = threading.Thread(
                                target=self._run_ocr_async,
                                args=(tid, frame_copy, vehicle_data['bbox'], spd),
                                daemon=True,
                            )
                            t.start()

            # Dibujar
            self._draw_lines(annotated_frame)
            self._draw_speed_labels(annotated_frame, tracked_vehicles)
            self._draw_hud(annotated_frame)
            self._draw_infraction_alert(annotated_frame)   # Borde rojo parpadeante
            self._draw_infraction_log(annotated_frame)     # Historial de infracciones

            cv2.imshow(WINDOW_NAME, annotated_frame)

            # --- Teclado ---
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q') or key == 27:       # Q o Esc
                break

            elif key == ord('l'):                   # L: cambiar orientacion de lineas
                if self.line_mode == self.MODE_VERTICAL:
                    self.line_mode = self.MODE_HORIZONTAL
                else:
                    self.line_mode = self.MODE_VERTICAL
                print(f"[INFO] Orientacion de lineas: {self.line_mode.upper()}")
                self._reset_lines()

            elif key == ord('d'):                   # D: diagnostico
                self.diag_mode = not self.diag_mode
                print(f"[INFO] Diagnostico: {'ON' if self.diag_mode else 'OFF'}")

            elif key == ord('r'):                   # R: resetear
                self._reset_lines()

            elif key == ord('+') or key == ord('='):    # +: mas distancia
                self.real_distance_m = round(self.real_distance_m + REAL_DISTANCE_STEP, 1)
                print(f"[INFO] Distancia real: {self.real_distance_m} m")
                if self.click_n == 2:
                    self._init_speed_calculator()

            elif key == ord('-'):                   # -: menos distancia
                self.real_distance_m = max(0.5, round(self.real_distance_m - REAL_DISTANCE_STEP, 1))
                print(f"[INFO] Distancia real: {self.real_distance_m} m")
                if self.click_n == 2:
                    self._init_speed_calculator()

            elif key == ord('.'):                   # .: mas limite de velocidad
                self.speed_limit_kmh = round(self.speed_limit_kmh + SPEED_LIMIT_STEP, 0)
                print(f"[INFO] Limite de velocidad: {self.speed_limit_kmh:.0f} km/h")

            elif key == ord(','):                   # ,: menos limite de velocidad
                self.speed_limit_kmh = max(5.0, round(self.speed_limit_kmh - SPEED_LIMIT_STEP, 0))
                print(f"[INFO] Limite de velocidad: {self.speed_limit_kmh:.0f} km/h")

            elif key == ord('s'):                   # S: screenshot
                fname = f"screenshot_{self.screenshot_n:03d}.jpg"
                cv2.imwrite(fname, annotated_frame)
                print(f"[INFO] Screenshot guardado: {fname}")
                self.screenshot_n += 1

        cap.release()
        cv2.destroyAllWindows()
        print("\n[INFO] Test finalizado.")


if __name__ == "__main__":
    test = InteractiveSpeedTest(
        camera_index=CAMERA_INDEX,
        real_distance_m=REAL_DISTANCE_M,
        speed_limit_kmh=SPEED_LIMIT_KMH,
    )
    test.run()
