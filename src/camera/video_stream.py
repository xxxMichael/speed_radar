import cv2
import threading
import time


class IPWebcamStream:
    """
    Captura de video en tiempo real sin lag desde una camara IP o fuente local.

    Estrategia anti-lag (doble nivel):
    1. Nivel protocolo: soporte RTSP (H.264) ademas de MJPEG HTTP.
       RTSP tiene mucha menor latencia en red WiFi local.
    2. Nivel buffer: usa grab()+retrieve() para DRENAR el buffer interno
       de ffmpeg sin decodificar cada frame intermedio.
       grab() es rapido (no decodifica). Solo retrieve() decodifica el ultimo.

    El resultado: el hilo principal SIEMPRE recibe el frame mas fresco,
    sin saltos ni acumulacion de frames viejos.
    """

    def __init__(
        self,
        src: str | int = 0,
        reconnect_delay: float = 2.0,
        drain_frames: int = 4,
        target_width: int = None,
    ):
        """
        Args:
            src (str | int): URL del stream (RTSP o HTTP MJPEG) o indice de camara local.
                             Ejemplos:
                               RTSP (recomendado): 'rtsp://192.168.100.59:8080/h264_ulaw.sdp'
                               MJPEG HTTP:        'http://192.168.100.59:8080/video'
            reconnect_delay (float): Segundos de espera antes de reconectar si el stream cae.
            drain_frames (int): Cuantos frames drenar del buffer en cada iteracion.
                                Valores 3-6 eliminan el lag sin perder continuidad.
            target_width (int): Si se especifica, escala el frame a este ancho (mantiene aspecto).
                                Util para reducir carga de CPU en el hilo principal.
                                Ej: 640 para reducir a 640px de ancho.
        """
        self.src             = src
        self.reconnect_delay = reconnect_delay
        self.drain_frames    = drain_frames
        self.target_width    = target_width

        # Frame mas reciente compartido entre hilos
        self.frame = None
        self.ret   = False
        self._lock = threading.Lock()

        self.stopped = False
        self._cap    = None

        self._connect()

        self._thread = threading.Thread(target=self._update, daemon=True)
        self._thread.start()

    # ------------------------------------------------------------------
    # Conexion
    # ------------------------------------------------------------------

    def _connect(self) -> bool:
        """Abre la fuente de video con configuracion optimizada para tiempo real."""
        if self._cap is not None:
            self._cap.release()

        self._cap = cv2.VideoCapture(self.src, cv2.CAP_FFMPEG)

        # Reducir buffer interno de ffmpeg al minimo (1 frame)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if self._cap.isOpened():
            # Mostrar resolucion y FPS del stream recibido
            w   = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h   = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = self._cap.get(cv2.CAP_PROP_FPS)
            print(f"[INFO] Stream abierto: {self.src}")
            print(f"[INFO] Resolucion fuente: {w}x{h} @ {fps:.0f} FPS")
            return True
        else:
            print(f"[ERROR] No se pudo abrir: {self.src}")
            return False

    # ------------------------------------------------------------------
    # Hilo de captura
    # ------------------------------------------------------------------

    def _update(self):
        """
        Bucle de captura en hilo secundario.

        Tecnica de drenado de buffer:
        - Se llama a grab() varias veces seguidas (rapido, sin decodificar)
          para saltar los frames viejos que ffmpeg acumulo.
        - Solo el ultimo grab() se decodifica con retrieve().
        - Esto elimina los saltos bruscos de "escena" causados por el buffer.
        """
        while not self.stopped:
            if not self._cap or not self._cap.isOpened():
                print(f"[WARN] Stream caido. Reconectando en {self.reconnect_delay}s...")
                time.sleep(self.reconnect_delay)
                self._connect()
                continue

            # --- DRENADO DE BUFFER ---
            # Hacer grab() N veces sin decodificar para saltar frames acumulados
            grabbed = False
            for _ in range(self.drain_frames):
                grabbed = self._cap.grab()
                if not grabbed:
                    break

            if not grabbed:
                # Si ningun grab tuvo exito, el stream puede haber caido
                time.sleep(0.05)
                continue

            # Decodificar SOLO el ultimo frame grabado (el mas reciente)
            ret, frame = self._cap.retrieve()

            if not ret or frame is None:
                time.sleep(0.05)
                continue

            # Redimensionar si se especifico target_width (reduce carga de CPU)
            if self.target_width is not None and frame.shape[1] != self.target_width:
                ratio  = self.target_width / frame.shape[1]
                height = int(frame.shape[0] * ratio)
                frame  = cv2.resize(frame, (self.target_width, height), interpolation=cv2.INTER_LINEAR)

            # Sobreescribir con el frame mas reciente
            with self._lock:
                self.ret   = ret
                self.frame = frame

        if self._cap is not None:
            self._cap.release()
        print("[INFO] Hilo de captura detenido y recursos liberados.")

    # ------------------------------------------------------------------
    # API publica
    # ------------------------------------------------------------------

    def read(self):
        """
        Retorna el frame mas reciente disponible.

        Returns:
            tuple[bool, numpy.ndarray | None]: (exito, frame) — siempre el mas fresco.
        """
        with self._lock:
            if self.frame is None:
                return False, None
            return self.ret, self.frame.copy()

    def is_opened(self) -> bool:
        """True si el stream esta activo."""
        return self._cap is not None and self._cap.isOpened()

    def stop(self):
        """Detiene el hilo y libera recursos."""
        self.stopped = True
        self._thread.join(timeout=3.0)
        print("[INFO] Stream de video detenido.")


# ------------------------------------------------------------------
# Script de prueba con diagnostico de latencia
# ------------------------------------------------------------------
if __name__ == "__main__":
    # IP Webcam expone dos URLs utiles:
    #   RTSP (menor latencia, H.264): rtsp://IP:8080/h264_ulaw.sdp
    #   MJPEG HTTP (mayor latencia):  http://IP:8080/video
    #
    # PRUEBA PRIMERO EL RTSP — si funciona, usalo en produccion.
    IP = "192.168.100.59"

    RTSP_URL  = f"rtsp://{IP}:8080/h264_ulaw.sdp"
    MJPEG_URL = f"http://{IP}:8080/video"

    # Cambiar a MJPEG_URL si RTSP no funciona
    STREAM_URL = RTSP_URL

    print(f"[INFO] Probando stream: {STREAM_URL}")
    print("[INFO] Presiona 'q' para salir.")
    print("[INFO] FPS mostrado = fluidez real del stream en tu PC.")
    print("-" * 60)

    camara = IPWebcamStream(
        src=STREAM_URL,
        drain_frames=4,
        target_width=640,   # Redimensionar a 640px para mejor rendimiento
    )
    time.sleep(1.5)

    # Medir FPS real para diagnosticar latencia
    fps_counter = 0
    fps_display = 0.0
    fps_timer   = time.time()

    while True:
        ret, frame = camara.read()

        if not ret or frame is None:
            print("[WARN] Sin frames, esperando...")
            time.sleep(0.1)
            continue

        # Calcular FPS cada segundo
        fps_counter += 1
        elapsed = time.time() - fps_timer
        if elapsed >= 1.0:
            fps_display = fps_counter / elapsed
            fps_counter = 0
            fps_timer   = time.time()

        # Mostrar FPS y resolucion en el frame
        h, w = frame.shape[:2]
        cv2.putText(frame, f"FPS: {fps_display:.1f}  |  {w}x{h}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("Test Stream (q para salir)", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    camara.stop()
    cv2.destroyAllWindows()
