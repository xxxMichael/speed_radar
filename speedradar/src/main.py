import cv2
import time
import torch
import numpy as np

# Importar los módulos del proyecto
from camera.video_stream import IPWebcamStream
from detection.vehicle_tracker import VehicleTracker
from detection.speed_calculator import SpeedCalculator
from ocr.plate_segmentation import PlateSegmenter
from ocr.cnn_model import CharClassifierCNN
from fuzzy.fine_system import FineSystem

def main():
    # --- Configuración Inicial ---
    # URL de la IP Webcam (Celular) - Modificar según sea necesario
    stream_url = "http://192.168.100.59:8080/video" 
    
    limite_velocidad_via = 50.0  # Límite en km/h
    distancia_lineas_metros = 10.0 # Distancia real entre las dos líneas virtuales
    line1_y = 200 # Coordenada Y de la línea 1 en la imagen
    line2_y = 400 # Coordenada Y de la línea 2 en la imagen
    
    # --- Inicialización de Módulos ---
    print("[INFO] Iniciando Stream de Video...")
    stream = IPWebcamStream(src=stream_url)   # El hilo inicia automaticamente
    time.sleep(2.0)  # Dar tiempo a que el stream inicie
    
    print("[INFO] Cargando modelos y sistemas...")
    tracker = VehicleTracker(model_path='yolov8n.pt')
    speed_calc = SpeedCalculator(line1_y, line2_y, distancia_lineas_metros, direction='down')
    
    plate_segmenter = PlateSegmenter(target_size=(28, 28))
    
    # Cargar CNN OCR
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ocr_model = CharClassifierCNN(num_classes=47).to(device)
    # ocr_model.load_state_dict(torch.load('src/ocr/char_cnn.pth')) # Descomentar cuando esté entrenado
    ocr_model.eval()
    
    fine_system = FineSystem()
    
    print("[INFO] Sistema listo. Presione 'q' para salir.")
    
    # Para mapear la salida de EMNIST (47 clases) a caracteres reales
    # Este mapping dependerá de cómo torchvision carga el split 'balanced'
    
    while True:
        ret, frame = stream.read()
        if not ret or frame is None:
            continue
            
        current_time = time.time()
        
        # 1. Detección y Tracking
        tracked_vehicles, annotated_frame = tracker.process_frame(frame)
        
        # 2. Cálculo de Velocidad
        new_speeds = speed_calc.update(tracked_vehicles, current_time)
        
        # Dibujar líneas virtuales
        cv2.line(annotated_frame, (0, line1_y), (frame.shape[1], line1_y), (255, 0, 0), 2)
        cv2.line(annotated_frame, (0, line2_y), (frame.shape[1], line2_y), (0, 0, 255), 2)
        
        # 3. Procesar eventos (vehículos con velocidad calculada)
        for vehicle in tracked_vehicles:
            track_id = vehicle['track_id']
            speed = speed_calc.get_speed(track_id)
            
            if speed is not None:
                # Dibujar velocidad en pantalla
                cx, cy = vehicle['centroid']
                text = f"ID: {track_id} - {speed:.1f} km/h"
                color = (0, 255, 0)
                
                # Gatillo de Infracción
                if speed > limite_velocidad_via:
                    color = (0, 0, 255) # Rojo para infracción
                    text += " [INFRACCION]"
                    
                    # --- Fase de OCR y Multa (Solo si hay infracción) ---
                    # Nota: Aquí deberíamos detectar la placa primero (ej. con otro modelo YOLO)
                    # Por simplicidad en este esqueleto, asumiremos que recortamos la zona central del vehículo
                    # x1, y1, x2, y2 = vehicle['bbox']
                    # plate_crop = frame[y1:y2, x1:x2] # Reemplazar con detección real de placa
                    
                    # Supongamos que segmentamos (si tuviéramos un crop real)
                    # char_crops, _ = plate_segmenter.segment_characters(plate_crop)
                    
                    # Predicción CNN...
                    # Multa
                    multa = fine_system.calculate_fine(speed, limite_velocidad_via)
                    
                    # Mostrar alerta
                    cv2.putText(annotated_frame, f"MULTA: ${multa:.2f}", (cx, cy - 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                
                cv2.putText(annotated_frame, text, (cx, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
        # Mostrar resultado
        cv2.imshow("Speed Radar Real-Time", annotated_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    # Limpieza
    stream.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
