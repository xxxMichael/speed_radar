"""
Script de prueba del sistema de deteccion de vehiculos y calculo de velocidad.

Soporta dos modos de entrada:
  - Camara en vivo (por defecto)
  - Archivo de video (con --video <ruta>)

Controles:
    Clic x2   -> Fijar las dos lineas de medicion (zona de calibracion)
    L         -> Cambiar orientacion de lineas: VERTICAL / HORIZONTAL
    + / -     -> Aumentar/disminuir distancia real entre lineas (metros)
    . / ,     -> Aumentar/disminuir limite de velocidad (km/h)
    D         -> Activar/desactivar modo diagnostico (muestra todos los objetos)
    R         -> Resetear lineas, velocidades y contadores
    S         -> Guardar screenshot del frame actual
    ESPACIO   -> Pausa/continuar (solo en modo video)
    ->        -> Avanzar 1 frame (solo en modo video, mientras esta en pausa)
    Q         -> Salir
"""

import cv2
import time
import threading
import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from detection.vehicle_tracker import VehicleTracker
from detection.speed_calculator import SpeedCalculator
from ocr.plate_ocr import PlateOCR
from fuzzy.fine_system import FineSystem
from notifications.email_sender import send_infraction_email


# =============================================================================
# CONFIGURACION INICIAL (ajustable en la interfaz)
# =============================================================================
CAMERA_INDEX       = 1          # DroidCam USB (cambiar a 0 si es webcam integrada)
CAMERA_BACKEND     = cv2.CAP_DSHOW

SPEED_LIMIT_KMH    = 30.0       # Limite inicial (ajustable con . y ,)
SPEED_LIMIT_STEP   = 5.0        # Cuanto cambia el limite con cada pulsacion

REAL_DISTANCE_M    = 5.0        # Distancia real inicial entre lineas (ajustable con + y -)
REAL_DISTANCE_STEP = 0.5

YOLO_CONF          = 0.30       # Umbral de confianza YOLO

WINDOW_NAME        = "Speed Radar - Deteccion (Q=salir)"
# =============================================================================


class InteractiveSpeedTest:
    """
    Herramienta interactiva de deteccion y medicion de velocidad.

    Soporta dos fuentes de video:
      - Camara en vivo (DroidCam, webcam, etc.)
      - Archivo de video (.mp4, .avi, etc.)

    Soporta lineas de medicion VERTICALES (movimiento horizontal de vehiculos)
    y HORIZONTALES (movimiento vertical de vehiculos), intercambiables con 'L'.
    El limite de velocidad y la distancia real se ajustan desde la interfaz.

    Calcula velocidad por TRAYECTORIA MULTI-FRAME: mide el desplazamiento del
    centroide del vehiculo en los ultimos N frames. Las lineas definen la zona
    de calibracion (pixeles -> metros) y la zona de reporte.
    """

    # Modos de orientacion de lineas
    MODE_VERTICAL   = 'vertical'    # Lineas | |  →  miden movimiento izquierda/derecha
    MODE_HORIZONTAL = 'horizontal'  # Lineas ═ ═  →  miden movimiento arriba/abajo

    def __init__(self, camera_index: int, real_distance_m: float,
                 speed_limit_kmh: float, video_path: str = None):
        self.camera_index    = camera_index
        self.real_distance_m = real_distance_m
        self.speed_limit_kmh = speed_limit_kmh
        self.video_path      = video_path  # None = camara en vivo

        # Orientacion de las lineas de medicion
        self.line_mode = self.MODE_VERTICAL   # Por defecto: lineas verticales

        # Posicion de las dos lineas (en pixeles)
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

        # FPS de la fuente (se detecta al abrir)
        self.source_fps = 30.0

        self.screenshot_n = 0

        # --- Control de reproduccion (modo video) ---
        self.paused     = False
        self.frame_num  = 0

        # --- Infracciones ---
        self.infraction_log: list[dict] = []
        self.alert_until: float = 0.0

        # --- OCR de placas ---
        self.plate_ocr: PlateOCR | None = None   # Se inicializa en run()
        # Placa final (ganadora por votacion): {track_id: plate_str}
        self.plates: dict[int, str] = {}
        # Votos acumulados por vehiculo: {track_id: {plate_str: count}}
        self._plate_votes: dict[int, dict[str, int]] = {}
        # Set de track_ids cuyo OCR ya esta corriendo en hilo (evita duplicados)
        self._ocr_running: set[int] = set()
        # Ultimo tiempo de escaneo por vehiculo (para reintentos dinamicos)
        self._last_ocr_time: dict[int, float] = {}

        # --- Sistema Difuso de Multas ---
        self.fine_system = FineSystem()

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

        # Registrar voto y obtener placa ganadora por votacion
        plate_str = self._vote_plate(tid, plate if plate else '')
        self.plates[tid] = plate_str
        ts = time.strftime("%H:%M:%S")

        # Actualizar en el log de infracciones de la interfaz
        for inf in self.infraction_log:
            if inf['id'] == tid and inf['plate'] == '...':
                inf['plate'] = plate_str

        if speed > self.speed_limit_kmh:
            # Calcular la multa usando el sistema difuso Mamdani
            fine_amount = self.fine_system.calculate_fine(speed, self.speed_limit_kmh)
            
            print(f"[INFRACCION] {ts} | Vehiculo #{tid} | Placa: {plate_str} | Velocidad: {speed:.1f} km/h | Multa Difusa: ${fine_amount:.2f} USD")
            
            # Recortar la imagen del vehículo para enviarla como evidencia
            try:
                h, w = frame_copy.shape[:2]
                x1, y1, x2, y2 = bbox
                # Asegurar límites dentro del frame
                x1, y1 = max(0, int(x1)), max(0, int(y1))
                x2, y2 = min(w, int(x2)), min(h, int(y2))
                
                if x2 > x1 and y2 > y1:
                    vehicle_crop = frame_copy[y1:y2, x1:x2]
                else:
                    vehicle_crop = None
            except Exception as e:
                vehicle_crop = None
                print(f"[WARNING] No se pudo recortar la evidencia para vehiculo #{tid}: {e}")
            
            # Disparar correo asíncrono
            send_infraction_email(
                vehicle_id=tid,
                plate=plate_str,
                speed=speed,
                speed_limit=self.speed_limit_kmh,
                fine_amount=fine_amount,
                frame_crop=vehicle_crop
            )
        else:
            print(f"[REGISTRO]   {ts} | Vehiculo #{tid} | Placa: {plate_str}")

    def _vote_plate(self, tid: int, plate: str) -> str:
        """
        Acumula votos de lecturas de placa y retorna la placa ganadora.
        Prefiere placas con formato ecuatoriano valido (7 chars con guion).
        """
        import re
        if tid not in self._plate_votes:
            self._plate_votes[tid] = {}

        if plate and plate != '???':
            votes = self._plate_votes[tid]
            votes[plate] = votes.get(plate, 0) + 1

        # Ordenar por votos, priorizando placas con formato AAA-NNNN
        pattern = re.compile(r'^[A-Z]{3}-\d{4}$')
        all_votes = self._plate_votes.get(tid, {})
        if not all_votes:
            return '???'

        # Primero buscar entre las que cumplen el formato
        valid = {p: v for p, v in all_votes.items() if pattern.match(p)}
        if valid:
            return max(valid, key=valid.get)

        # Si ninguna cumple el formato, devolver la mas votada con al menos 4 chars
        long_plates = {p: v for p, v in all_votes.items() if len(p) >= 4}
        if long_plates:
            return max(long_plates, key=long_plates.get)

        return max(all_votes, key=all_votes.get)

    def _run_ocr_continuous_async(self, tid: int, frame_copy: 'np.ndarray', bbox: list):
        """
        Ejecuta el OCR de forma continua/dinamica para un vehiculo en pantalla.
        Acumula votos y actualiza la placa ganadora.
        """
        try:
            plate, _ = self.plate_ocr.read_plate(frame_copy, bbox)
        except Exception as e:
            plate = ''
        finally:
            self._ocr_running.discard(tid)

        # Registrar voto y obtener ganadora
        winner = self._vote_plate(tid, plate)
        old    = self.plates.get(tid, '???')

        if winner != old:
            self.plates[tid] = winner
            ts = time.strftime("%H:%M:%S")
            total_votes = sum(self._plate_votes.get(tid, {}).values())
            print(f"[OCR] {ts} | Vehiculo #{tid} | Placa: {winner}  (lectura #{total_votes})")

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
            fps=self.source_fps,
        )
        self.speeds = {}
        lo = min(self.line1_pos, self.line2_pos)
        hi = max(self.line1_pos, self.line2_pos)
        m_per_px = self.real_distance_m / max(1, hi - lo)
        print(f"[INFO] SpeedCalculator (trayectoria) | modo={self.line_mode} | dist={self.real_distance_m}m | L1={lo}px | L2={hi}px | {m_per_px:.4f} m/px")

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

        # Dibujar zona de medicion semitransparente si ambas lineas estan fijadas
        if self.line1_pos is not None and self.line2_pos is not None:
            lo = min(self.line1_pos, self.line2_pos)
            hi = max(self.line1_pos, self.line2_pos)
            overlay = frame.copy()
            if self.line_mode == self.MODE_VERTICAL:
                cv2.rectangle(overlay, (lo, 0), (hi, h), (0, 180, 0), -1)
            else:
                cv2.rectangle(overlay, (0, lo), (w, hi), (0, 180, 0), -1)
            cv2.addWeighted(overlay, 0.12, frame, 0.88, 0, frame)

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
        panel_h = 200 if self.video_path else 175
        overlay = frame.copy()
        cv2.rectangle(overlay, (panel_x - 6, 0), (w, panel_h), (0, 0, 0), -1)
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

        # Fuente de video
        if self.video_path:
            src_label = f"Video: {os.path.basename(self.video_path)}"
            pause_label = " [PAUSA]" if self.paused else ""
            cv2.putText(frame, f"{src_label}{pause_label}",
                        (panel_x, row), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 255), 1)
            row += step

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

        controls = "[R] Reset   [S] Screenshot   [Q] Salir"
        if self.video_path:
            controls = "[SPC] Pausa  " + controls
        cv2.putText(frame, controls,
                    (panel_x, row), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (150, 150, 150), 1)

        # Alerta visual si falta configurar las lineas de calibracion
        if self.click_n < 2:
            warn_msg = "AVISO: Haga 2 clics en la pantalla para fijar las lineas de calibracion"
            cv2.rectangle(frame, (10, h - 35), (w - 10, h - 10), (0, 0, 0), -1)
            cv2.putText(frame, warn_msg, (20, h - 17),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 140, 255), 1)

    def _draw_speed_labels(self, frame, tracked_vehicles):
        """Dibuja la velocidad calculada y la placa encima de cada vehiculo."""
        for vehicle in tracked_vehicles:
            tid    = vehicle['track_id']
            cx, cy = vehicle['centroid']
            speed  = self.speeds.get(tid)
            plate  = self.plates.get(tid)

            # Usar la velocidad en tiempo real de la trayectoria si aun no esta reportada
            if speed is None and self.speed_calc is not None:
                speed = self.speed_calc.get_speed(tid)

            # Armar la etiqueta
            label_parts = []
            if plate:
                label_parts.append(f"Placa: {plate}")
            else:
                label_parts.append("Placa: Buscando...")

            if speed is not None:
                label_parts.append(f"{speed:.1f} km/h")

            label = " | ".join(label_parts)
            over = speed is not None and speed > self.speed_limit_kmh
            color = (0, 0, 255) if over else (0, 220, 0)
            
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(frame, (cx - 4, cy - th - 12), (cx + tw + 4, cy - 2), (0, 0, 0), -1)
            cv2.putText(frame, label, (cx, cy - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)

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
    # Abrir fuente de video
    # ------------------------------------------------------------------

    def _open_source(self):
        """
        Abre la fuente de video (camara o archivo) y retorna el VideoCapture.

        Returns:
            tuple: (cap, fw, fh, fps) o None si fallo.
        """
        if self.video_path:
            # --- Modo archivo de video ---
            if not os.path.exists(self.video_path):
                print(f"[ERROR] Archivo de video no encontrado: {self.video_path}")
                return None
            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                print(f"[ERROR] No se pudo abrir el video: {self.video_path}")
                return None
            fw  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            fh  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            print(f"[INFO] Video abierto: {self.video_path}")
            print(f"[INFO] Resolucion: {fw}x{fh} | FPS: {fps:.1f} | Frames: {total}")
        else:
            # --- Modo camara en vivo ---
            cap = cv2.VideoCapture(self.camera_index, CAMERA_BACKEND)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if not cap.isOpened():
                print(f"[ERROR] No se pudo abrir la camara {self.camera_index}.")
                return None
            fw  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            fh  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            print(f"[INFO] Camara abierta: {fw}x{fh}")

        self.source_fps = fps
        return cap, fw, fh, fps

    # ------------------------------------------------------------------
    # Bucle principal
    # ------------------------------------------------------------------

    def run(self):
        print("\n" + "=" * 65)
        print("  SPEED RADAR - TEST INTERACTIVO")
        print("=" * 65)

        if self.video_path:
            print(f"  Fuente         : VIDEO -> {self.video_path}")
        else:
            print(f"  Fuente         : CAMARA indice {self.camera_index}")

        print(f"  Limite inicial : {self.speed_limit_kmh} km/h  (ajustar con , / .)")
        print(f"  Distancia init : {self.real_distance_m} m   (ajustar con - / +)")
        print(f"  YOLO conf      : {YOLO_CONF}")
        print(f"  Algoritmo vel. : Trayectoria multi-frame")
        print("-" * 65)
        print("  CONTROLES:")
        print("    Clic x2  -> Fijar lineas de medicion (zona de calibracion)")
        print("    L        -> Cambiar orientacion lineas (VERTICAL / HORIZONTAL)")
        print("    + / -    -> Distancia real entre lineas")
        print("    . / ,    -> Limite de velocidad")
        print("    D        -> Modo diagnostico (ver todos los objetos)")
        print("    R        -> Resetear")
        print("    S        -> Screenshot")
        if self.video_path:
            print("    ESPACIO  -> Pausa/continuar")
            print("    ->       -> Avanzar 1 frame (en pausa)")
        print("    Q        -> Salir")
        print("=" * 65)
        print("\n[IMPORTANTE] Para calcular y ver las velocidades en consola y video:")
        print("  1. Haga DOS CLICS en la ventana del video para fijar las lineas L1 y L2.")
        print("  2. Asegurese de que la orientacion (Vertical/Horizontal) con tecla 'L' coincida con la trayectoria.")
        print("  3. Ajuste la distancia de separacion real en metros con las teclas '+' y '-'.\n")

        # Abrir fuente
        result = self._open_source()
        if result is None:
            return
        cap, fw, fh, fps = result

        # Configurar lineas de calibracion por defecto si no se han establecido
        if self.line1_pos is None and self.line2_pos is None:
            if self.line_mode == self.MODE_VERTICAL:
                self.line1_pos = int(fw * 0.20)
                self.line2_pos = int(fw * 0.80)
            else:
                self.line1_pos = int(fh * 0.20)
                self.line2_pos = int(fh * 0.80)
            self.click_n = 2
            self._init_speed_calculator()
            print(f"[INFO] Lineas de calibracion por defecto activadas: L1={self.line1_pos}px, L2={self.line2_pos}px. SpeedCalculator activo.")

        # Cargar YOLO
        print(f"[INFO] Cargando YOLOv8n (conf={YOLO_CONF})...")
        tracker_yaml = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 'custom_bytetrack.yaml'
        )
        self.tracker = VehicleTracker(model_path='yolov8n.pt', tracker=tracker_yaml, conf=YOLO_CONF)
        print("[OK] YOLOv8n listo.")

        # Cargar OCR de placas
        # Priorizar la CNN sintética (FE-Schrift), con fallback al modelo original EMNIST
        ocr_model_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), '..', 'char_cnn_synthetic.pth'
        )
        if not os.path.exists(ocr_model_path):
            ocr_model_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), '..', 'char_cnn.pth'
            )
        self.plate_ocr = PlateOCR(model_path=ocr_model_path)
        print("[OK] PlateOCR listo.\n")

        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, min(fw, 1280), min(fh, 720))
        cv2.setMouseCallback(WINDOW_NAME, self._mouse_callback)

        # Delay entre frames para sincronizar con FPS del video
        frame_delay_ms = int(1000 / fps) if self.video_path else 1

        while True:
            # --- Control de pausa (solo modo video) ---
            if self.paused and self.video_path:
                key = cv2.waitKey(50) & 0xFF
                if key == ord(' '):  # Espacio: continuar
                    self.paused = False
                    print("[INFO] Reproduccion reanudada.")
                elif key == 83 or key == ord('n'):  # Flecha derecha o N: avanzar 1 frame
                    pass  # Dejar que el frame se lea abajo
                elif key == ord('q') or key == 27:
                    break
                elif key == ord('r'):
                    self._reset_lines()
                    continue
                elif key == ord('s'):
                    fname = f"screenshot_{self.screenshot_n:03d}.jpg"
                    cv2.imwrite(fname, self._last_frame if hasattr(self, '_last_frame') else frame)
                    print(f"[INFO] Screenshot guardado: {fname}")
                    self.screenshot_n += 1
                    continue
                else:
                    continue  # Queda en pausa sin leer frame

            ret, frame = cap.read()
            if not ret or frame is None:
                if self.video_path:
                    # Loop: reiniciar desde el inicio
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    self.frame_num = 0
                    # Resetear tracker y velocidades para el nuevo loop
                    if self.speed_calc:
                        self.speed_calc.reset()
                    self.speeds = {}
                    print("[INFO] Video reiniciado (loop).")
                    continue
                else:
                    time.sleep(0.05)
                    continue

            self.frame_num += 1

            # Timestamp: usar frame_num / fps para video, time.time() para camara
            if self.video_path:
                current_time = self.frame_num / self.source_fps
            else:
                current_time = time.time()

            # Calcular FPS de procesamiento
            wall_time = time.time()
            self.fps_counter += 1
            elapsed = wall_time - self.fps_timer
            if elapsed >= 1.0:
                self.fps_display = self.fps_counter / elapsed
                self.fps_counter = 0
                self.fps_timer   = wall_time

            # Deteccion
            if self.diag_mode:
                self.detected_count, annotated_frame = self.tracker.detect_all(frame)
                tracked_vehicles = []
            else:
                tracked_vehicles, annotated_frame = self.tracker.process_frame(frame)
                self.detected_count = len(tracked_vehicles)

            # Disparar OCR dinamico continuo para vehiculos en pantalla en todo momento
            if self.plate_ocr is not None and not self.diag_mode:
                import re
                _plate_pattern = re.compile(r'^[A-Z]{3}-\d{4}$')
                for vehicle in tracked_vehicles:
                    tid = vehicle['track_id']
                    current_plate = self.plates.get(tid, '???')
                    # Si ya tenemos una placa valida (formato AAA-NNNN), no reintentar
                    if _plate_pattern.match(current_plate):
                        continue
                    # Reintentar si no tenemos placa o si la placa es parcial/invalida
                    if time.time() - self._last_ocr_time.get(tid, 0) > 2.0:
                        if tid not in self._ocr_running:
                            self._ocr_running.add(tid)
                            self._last_ocr_time[tid] = time.time()
                            frame_copy = frame.copy()
                            t = threading.Thread(
                                target=self._run_ocr_continuous_async,
                                args=(tid, frame_copy, vehicle['bbox']),
                                daemon=True,
                            )
                            t.start()

            # Calculo de velocidad por trayectoria y disparo de OCR
            if self.speed_calc is not None and self.click_n == 2:
                new_speeds = self.speed_calc.update(tracked_vehicles, current_time)
                self.speeds.update(new_speeds)

                for tid, spd in new_speeds.items():
                    ts = time.strftime("%H:%M:%S")

                    # Determinar estado frente al límite (Infracción, Cerca del límite, Normal)
                    diff = self.speed_limit_kmh - spd
                    if spd > self.speed_limit_kmh:
                        status_str = f"\033[91m[INFRACCIÓN - ¡Exceso de velocidad! (+{abs(diff):.1f} km/h)]\033[0m"
                        self.alert_until = time.time() + 3.0
                        self.infraction_log.append({
                            'id':    tid,
                            'speed': spd,
                            'time':  ts,
                            'plate': '...',   # Se actualiza cuando termina el OCR
                        })
                    elif spd >= self.speed_limit_kmh - 5.0:
                        status_str = f"\033[93m[ADVERTENCIA - Cerca del límite (-{diff:.1f} km/h)]\033[0m"
                    else:
                        status_str = f"\033[92m[NORMAL - Velocidad segura (-{diff:.1f} km/h)]\033[0m"

                    print(f"[{ts}] Radar -> Vehículo #{tid:<2d} | Velocidad: {spd:>5.1f} km/h (Límite: {self.speed_limit_kmh:.0f} km/h) | {status_str}")

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

            # Guardar ultimo frame para screenshots en pausa
            self._last_frame = annotated_frame

            # Dibujar
            self._draw_lines(annotated_frame)
            self._draw_speed_labels(annotated_frame, tracked_vehicles)
            self._draw_hud(annotated_frame)
            self._draw_infraction_alert(annotated_frame)   # Borde rojo parpadeante
            self._draw_infraction_log(annotated_frame)     # Historial de infracciones

            cv2.imshow(WINDOW_NAME, annotated_frame)

            # --- Teclado ---
            key = cv2.waitKey(frame_delay_ms) & 0xFF

            if key == ord('q') or key == 27:       # Q o Esc
                break

            elif key == ord(' ') and self.video_path:  # Espacio: pausa
                self.paused = True
                print("[INFO] Reproduccion en pausa. [ESPACIO]=continuar, [->]=avanzar frame")

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


def parse_args():
    """Parsea argumentos de linea de comandos."""
    parser = argparse.ArgumentParser(
        description="Speed Radar - Test Interactivo de Deteccion de Velocidad"
    )
    parser.add_argument(
        '--video', '-v',
        type=str,
        default=None,
        help='Ruta a un archivo de video (.mp4, .avi). Si no se especifica, usa la camara en vivo.'
    )
    parser.add_argument(
        '--camera', '-c',
        type=int,
        default=CAMERA_INDEX,
        help=f'Indice de la camara (default: {CAMERA_INDEX}). Ignorado si se usa --video.'
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    test = InteractiveSpeedTest(
        camera_index=args.camera,
        real_distance_m=REAL_DISTANCE_M,
        speed_limit_kmh=SPEED_LIMIT_KMH,
        video_path=args.video,
    )
    test.run()
