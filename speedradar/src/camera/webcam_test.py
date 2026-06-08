"""
Modulo de prueba y comparacion de fuentes de video.

Detecta automaticamente todas las camaras disponibles (webcam USB, DroidCam,
camara integrada) y permite probar cada una midiendo FPS real y latencia.

Uso:
    python webcam_test.py              -> Prueba camara 0 (por defecto)
    python webcam_test.py --index 1    -> Prueba camara con indice 1
    python webcam_test.py --scan       -> Escanea todos los indices disponibles
    python webcam_test.py --compare    -> Abre todas las camaras y compara lado a lado

Controles durante la prueba:
    q       -> Salir
    s       -> Guardar captura de pantalla
    +/-     -> Aumentar/disminuir resolucion objetivo
"""

import cv2
import time
import argparse
import sys


# Configuracion de resoluciones de prueba
RESOLUTIONS = [
    (320,  240),   # QVGA
    (640,  480),   # VGA
    (1280, 720),   # HD
    (1920, 1080),  # Full HD
]


def scan_cameras(max_index: int = 8) -> list[int]:
    """
    Escanea los primeros N indices de camara y retorna los que estan disponibles.

    Args:
        max_index (int): Cuantos indices probar (0..max_index-1).

    Returns:
        list[int]: Lista de indices donde se encontro una camara activa.
    """
    print(f"[INFO] Escaneando camaras en indices 0..{max_index - 1}...")
    found = []

    for i in range(max_index):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)  # CAP_DSHOW = DirectShow en Windows
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                print(f"  [ENCONTRADA] Indice {i} -> {w}x{h} @ {fps:.0f} FPS")
                found.append(i)
            cap.release()
        else:
            cap.release()

    if not found:
        print("[WARN] No se encontro ninguna camara conectada.")
    else:
        print(f"[INFO] Total de camaras encontradas: {len(found)} -> Indices: {found}")

    return found


def measure_fps(cap: cv2.VideoCapture, duration: float = 3.0) -> float:
    """
    Mide los FPS reales de una captura durante N segundos.

    Args:
        cap: VideoCapture ya abierto.
        duration: Cuantos segundos medir.

    Returns:
        float: FPS promedio medido.
    """
    count = 0
    start = time.time()

    while time.time() - start < duration:
        ret, frame = cap.read()
        if ret:
            count += 1

    elapsed = time.time() - start
    return count / elapsed if elapsed > 0 else 0.0


def test_camera(index: int = 0, target_width: int = 640) -> None:
    """
    Abre una camara por indice y muestra el stream con metricas en tiempo real.

    Teclas:
        q -> Salir
        s -> Guardar screenshot

    Args:
        index (int): Indice de la camara (0=primera, 1=segunda, etc.)
        target_width (int): Ancho al que redimensionar los frames para mostrar.
    """
    print(f"\n[INFO] Abriendo camara con indice: {index}")
    print("[INFO] Controles: [q] Salir | [s] Screenshot")
    print("-" * 60)

    # CAP_DSHOW en Windows da latencia mucho menor que el backend por defecto
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Buffer minimo

    if not cap.isOpened():
        print(f"[ERROR] No se pudo abrir la camara {index}.")
        return

    # Info de la fuente
    w_src = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_src = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_src = cap.get(cv2.CAP_PROP_FPS)
    print(f"[INFO] Resolucion fuente: {w_src}x{h_src} @ {fps_src:.0f} FPS (reportado por driver)")

    # Medir FPS real antes de mostrar
    print("[INFO] Midiendo FPS real (3 segundos)...")
    fps_real = measure_fps(cap, duration=3.0)
    print(f"[INFO] FPS real medido: {fps_real:.1f}")

    if fps_real >= 25:
        print("[OK] Camara APTA para uso en tiempo real (>= 25 FPS)")
    elif fps_real >= 15:
        print("[WARN] Camara MARGINAL para tiempo real (15-24 FPS)")
    else:
        print("[ERROR] Camara NO APTA para tiempo real (< 15 FPS)")

    print("-" * 60)

    # Bucle de visualizacion con metricas
    fps_counter = 0
    fps_display = 0.0
    fps_timer   = time.time()
    frame_count = 0
    screenshot_n = 0

    while True:
        ret, frame = cap.read()

        if not ret or frame is None:
            print("[WARN] Frame perdido...")
            continue

        frame_count += 1
        fps_counter += 1

        # Calcular FPS cada segundo
        elapsed = time.time() - fps_timer
        if elapsed >= 1.0:
            fps_display = fps_counter / elapsed
            fps_counter = 0
            fps_timer   = time.time()

        # Redimensionar para visualizacion
        if frame.shape[1] != target_width:
            ratio  = target_width / frame.shape[1]
            height = int(frame.shape[0] * ratio)
            display = cv2.resize(frame, (target_width, height), interpolation=cv2.INTER_LINEAR)
        else:
            display = frame.copy()

        h_d, w_d = display.shape[:2]

        # --- HUD de metricas ---
        # Fondo semitransparente para el texto
        overlay = display.copy()
        cv2.rectangle(overlay, (0, 0), (w_d, 70), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, display, 0.5, 0, display)

        # Indicador de calidad por color
        color = (0, 255, 0) if fps_display >= 25 else (0, 165, 255) if fps_display >= 15 else (0, 0, 255)

        cv2.putText(display, f"FPS: {fps_display:5.1f}  |  Camara #{index}  |  {w_src}x{h_src} fuente",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
        cv2.putText(display, f"Frames totales: {frame_count}  |  [q] Salir  [s] Screenshot",
                    (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

        cv2.imshow(f"Test Camara #{index} (webcam_test.py)", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            filename = f"screenshot_cam{index}_{screenshot_n:03d}.jpg"
            cv2.imwrite(filename, frame)
            print(f"[INFO] Screenshot guardado: {filename}")
            screenshot_n += 1

    cap.release()
    cv2.destroyAllWindows()
    print(f"[INFO] Test finalizado. FPS promedio final: {fps_display:.1f}")


def compare_cameras(indices: list[int]) -> None:
    """
    Abre multiples camaras en paralelo y las muestra lado a lado para comparar
    latencia y calidad entre webcam USB, celular por WiFi, etc.

    Args:
        indices (list[int]): Lista de indices de camaras a comparar.
    """
    if not indices:
        print("[ERROR] No hay indices de camara para comparar.")
        return

    print(f"\n[INFO] Comparando camaras: {indices}")
    print("[INFO] Presiona 'q' para salir.")

    caps = {}
    for idx in indices:
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if cap.isOpened():
            caps[idx] = cap
            print(f"  [OK] Camara {idx} abierta.")
        else:
            print(f"  [ERROR] Camara {idx} no disponible, se omite.")

    if not caps:
        print("[ERROR] No se pudo abrir ninguna camara.")
        return

    fps_counters = {idx: 0 for idx in caps}
    fps_displays = {idx: 0.0 for idx in caps}
    fps_timer    = time.time()
    PANEL_WIDTH  = 480  # Ancho de cada panel en la comparacion

    while True:
        panels = []

        elapsed = time.time() - fps_timer
        if elapsed >= 1.0:
            for idx in caps:
                fps_displays[idx] = fps_counters[idx] / elapsed
                fps_counters[idx] = 0
            fps_timer = time.time()

        for idx, cap in caps.items():
            ret, frame = cap.read()
            if not ret or frame is None:
                # Panel negro si la camara no entrega frame
                panel = __import__('numpy').zeros((270, PANEL_WIDTH, 3), dtype='uint8')
                cv2.putText(panel, f"Cam #{idx} - SIN SEÑAL", (10, 135),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            else:
                fps_counters[idx] += 1
                ratio  = PANEL_WIDTH / frame.shape[1]
                height = int(frame.shape[0] * ratio)
                panel  = cv2.resize(frame, (PANEL_WIDTH, height))

                # HUD
                fps_val = fps_displays[idx]
                color   = (0, 255, 0) if fps_val >= 25 else (0, 165, 255) if fps_val >= 15 else (0, 0, 255)
                cv2.rectangle(panel, (0, 0), (PANEL_WIDTH, 40), (0, 0, 0), -1)
                cv2.putText(panel, f"Cam #{idx}  FPS: {fps_val:.1f}", (8, 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            panels.append(panel)

        # Asegurar que todos los paneles tengan la misma altura
        max_h = max(p.shape[0] for p in panels)
        import numpy as np
        padded = []
        for p in panels:
            if p.shape[0] < max_h:
                pad = np.zeros((max_h - p.shape[0], p.shape[1], 3), dtype='uint8')
                p   = np.vstack([p, pad])
            padded.append(p)

        combined = np.hstack(padded)
        cv2.imshow("Comparacion de camaras (q para salir)", combined)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    for cap in caps.values():
        cap.release()
    cv2.destroyAllWindows()


# ------------------------------------------------------------------
# Main: argumentos de linea de comandos
# ------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Herramienta de prueba y comparacion de fuentes de video."
    )
    parser.add_argument(
        "--index", type=int, default=0,
        help="Indice de camara a probar (default: 0)"
    )
    parser.add_argument(
        "--scan", action="store_true",
        help="Escanear todos los indices y mostrar camaras disponibles"
    )
    parser.add_argument(
        "--compare", action="store_true",
        help="Abrir todas las camaras disponibles y comparar lado a lado"
    )
    parser.add_argument(
        "--width", type=int, default=640,
        help="Ancho de visualizacion en pixeles (default: 640)"
    )
    args = parser.parse_args()

    if args.scan:
        found = scan_cameras(max_index=8)
        if found:
            print(f"\nPara probar una camara especifica ejecuta:")
            print(f"  python webcam_test.py --index <N>")
            print(f"\nPara comparar todas las camaras:")
            print(f"  python webcam_test.py --compare")

    elif args.compare:
        found = scan_cameras(max_index=8)
        if found:
            compare_cameras(found)

    else:
        test_camera(index=args.index, target_width=args.width)
