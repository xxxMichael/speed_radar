import os
import cv2
import glob
import time
import numpy as np
from plate_detector import PlateDetector
from plate_ocr import PlateOCR

def test_transcription():
    # Rutas
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'imagenes'))
    out_dir = os.path.join(base_dir, 'test_results')
    os.makedirs(out_dir, exist_ok=True)
    
    # Cargar modelos
    print("[TEST] Cargando Detector YOLO y Modelo CNN Sintético...")
    detector = PlateDetector()
    ocr = PlateOCR()
    
    # Buscar imagenes
    image_paths = glob.glob(os.path.join(base_dir, '*.jpg')) + glob.glob(os.path.join(base_dir, '*.png'))
    image_paths = [p for p in image_paths if 'test_results' not in p]
    
    if not image_paths:
        print(f"[TEST] No se encontraron imagenes en {base_dir}")
        return
        
    print(f"[TEST] Se encontraron {len(image_paths)} imagenes para evaluar.")
    
    for img_path in image_paths:
        img_name = os.path.basename(img_path)
        print(f"\n--- Analizando: {img_name} ---")
        img = cv2.imread(img_path)
        if img is None:
            print("[TEST] Error al cargar la imagen.")
            continue
            
        h, w = img.shape[:2]
        
        # En la realidad, el sistema recibe el crop del vehiculo
        # Asumiremos que la imagen entera es el "crop del vehiculo"
        vehicle_crop = img.copy()
        
        # 1. Detectar placa
        start_t = time.time()
        plate_crop, bbox = detector.find_plate(vehicle_crop)
        det_time = time.time() - start_t
        
        if plate_crop is None:
            print(f"[WARN] No se detecto ninguna placa en {img_name}")
            continue
            
        print(f"[*] Placa detectada en {det_time:.2f}s. Resolucion de placa: {plate_crop.shape[1]}x{plate_crop.shape[0]}")
        
        # 2. Transcribir placa
        start_t = time.time()
        # PlateOCR pide el frame entero y el bbox de la placa, o si le damos la placa y usamos read_plate...
        # Wait, read_plate expects (frame_copy, vehicle_bbox).
        # It's better to just call segment_characters and the CNN directly, but PlateOCR does exactly that.
        # Let's pass the vehicle_crop and a dummy full-image bbox so it searches the whole thing.
        plate_str, debug_img = ocr.read_plate(vehicle_crop, [0, 0, w, h])
        ocr_time = time.time() - start_t
        
        print(f"[*] Transcripcion completada en {ocr_time:.2f}s.")
        print(f"    --> Resultado: '{plate_str}'")
        
        # 3. Guardar resultados visuales
        # Dibujar bbox de la placa en la imagen original
        vis_img = detector.draw_detection(vehicle_crop, bbox)
        
        # Añadir texto con la transcripcion arriba
        cv2.putText(vis_img, f"OCR: {plate_str}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Combinar vis_img y debug_img horizontalmente (haciendo resize de debug_img al alto de vis_img)
        if debug_img is not None:
            dh, dw = debug_img.shape[:2]
            target_dh = vis_img.shape[0]
            if dh > 0:
                scale = target_dh / dh
                debug_resized = cv2.resize(debug_img, (int(dw * scale), target_dh))
                final_img = np.hstack([vis_img, debug_resized])
            else:
                final_img = vis_img
        else:
            final_img = vis_img
            
        out_path = os.path.join(out_dir, f"result_{img_name}")
        cv2.imwrite(out_path, final_img)
        print(f"[TEST] Resultado guardado en: {out_path}")

if __name__ == "__main__":
    test_transcription()
