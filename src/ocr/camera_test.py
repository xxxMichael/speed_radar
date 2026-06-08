import cv2
import time
import os
import sys

# Agregar src al path para poder importar módulos
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ocr.plate_detector import PlateDetector
from ocr.plate_ocr import PlateOCR
from notifications.email_sender import send_ocr_test_email

def run_camera_ocr_test():
    print("[INFO] Inicializando cámara web...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("[ERROR] No se pudo abrir la cámara.")
        return

    print("[INFO] Cargando modelos de IA...")
    detector = PlateDetector()
    ocr = PlateOCR()
    
    print("[INFO] Modelos cargados. Presiona 'q' para salir.")
    
    # Diccionario para no spamear correos de la misma placa repetidamente (cooldown de 15 seg)
    sent_emails = {}
    COOLDOWN_SECONDS = 15
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Detectar placa en todo el frame
        plate_crop, bbox = detector.find_plate(frame)
        
        display_frame = frame.copy()
        
        if plate_crop is not None and bbox is not None:
            # Dibujar caja de la placa encontrada (YOLO)
            display_frame = detector.draw_detection(display_frame, bbox)
            
            # Segmentar e inferir directamente del recorte de la placa para no repetir YOLO
            char_imgs, debug_img, _stages = ocr.segmenter.segment_characters(plate_crop)
            
            plate_str = ""
            avg_char_conf = 0.0
            if char_imgs:
                plate_chars = []
                confs = []
                for char_img in char_imgs:
                    char, conf = ocr._classify_char(char_img)
                    if conf >= ocr.char_conf_threshold:
                        plate_chars.append(char)
                        confs.append(conf)
                
                if confs:
                    avg_char_conf = sum(confs) / len(confs)
                
                # Importar función de normalización localmente si es necesario
                from ocr.plate_ocr import normalize_plate
                plate_str = normalize_plate(''.join(plate_chars).upper())
            
            if plate_str and len(plate_str) >= 4:
                det_method = getattr(detector, 'last_detection_method', 'Desconocido')
                det_conf = getattr(detector, 'last_detection_conf', 0.0)
                
                # Mostrar en pantalla
                text_pos = (bbox[0], max(20, bbox[1] - 10)) if bbox else (20, 40)
                cv2.putText(display_frame, f"PLACA: {plate_str}", text_pos,
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                
                # Revisar si se debe enviar el correo
                current_time = time.time()
                last_sent = sent_emails.get(plate_str, 0)
                
                if (current_time - last_sent) > COOLDOWN_SECONDS:
                    print(f"\n[*] ¡Nueva placa detectada! {plate_str}")
                    print(f"    - Método de Detección: {det_method}")
                    if det_method == "YOLO":
                        print(f"    - Confianza YOLO: {det_conf:.1%}")
                    print(f"    - Confianza Promedio OCR (Caracteres): {avg_char_conf:.1%}")
                    print(f"    Enviando correo de prueba...")
                    
                    sent_emails[plate_str] = current_time
                    
                    # Preparar imágenes para el correo (recorte amplio para contexto)
                    if bbox:
                        h, w = frame.shape[:2]
                        x1, y1, x2, y2 = bbox
                        pad_w = int((x2 - x1) * 0.5)
                        pad_h = int((y2 - y1) * 0.5)
                        cx1 = max(0, int(x1 - pad_w))
                        cy1 = max(0, int(y1 - pad_h))
                        cx2 = min(w, int(x2 + pad_w))
                        cy2 = min(h, int(y2 + pad_h))
                        context_crop = frame[cy1:cy2, cx1:cx2]
                    else:
                        context_crop = frame.copy()
                    
                    send_ocr_test_email(
                        plate=plate_str, 
                        vehicle_crop=context_crop, 
                        plate_crop=debug_img,
                        det_method=det_method,
                        det_conf=det_conf,
                        avg_char_conf=avg_char_conf
                    )
            
            # Mostrar la imagen de debug (los recortes verdes) en una ventana aparte
            if debug_img is not None and debug_img.shape[0] > 0 and debug_img.shape[1] > 0:
                cv2.imshow("Debug OCR (Caracteres)", debug_img)
                
        # Mostrar cámara en vivo
        cv2.imshow("Test OCR en Tiempo Real", display_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Programa finalizado.")

if __name__ == "__main__":
    run_camera_ocr_test()
