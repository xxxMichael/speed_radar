import cv2
import numpy as np

class PlateSegmenter:
    """
    Clase para segmentar caracteres individuales de una imagen recortada de una placa
    vehicular utilizando técnicas clásicas de Visión Computacional (OpenCV).
    """
    def __init__(self, target_size=(28, 28)):
        """
        Inicializa el segmentador.
        
        Args:
            target_size (tuple): Tamaño de salida para cada carácter segmentado (ancho, alto).
                                 28x28 es el estándar para EMNIST.
        """
        self.target_size = target_size

    def segment_characters(self, plate_img):
        """
        Segmenta los caracteres de la imagen de una placa.
        
        Args:
            plate_img (numpy.ndarray): Imagen BGR de la placa recortada.
            
        Returns:
            list: Lista de imágenes (numpy.ndarray) de los caracteres segmentados, 
                  en escala de grises y redimensionados a target_size, ordenados de izquierda a derecha.
            numpy.ndarray: Imagen de depuración con los contornos dibujados.
        """
        if plate_img is None or plate_img.size == 0:
            return [], None
            
        # 1. Convertir a escala de grises
        gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
        
        # 2. Mejora del contraste y suavizado
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # 3. Binarización (Umbralización Adaptativa o de Otsu)
        # Las placas suelen tener texto oscuro sobre fondo claro (o viceversa). 
        # Usaremos Otsu inverso para que las letras sean blancas (255) y el fondo negro (0).
        _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Opcional: Operaciones morfológicas para unir partes de caracteres rotos
        # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        # binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        # 4. Encontrar contornos
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        char_crops = []
        debug_img = plate_img.copy()
        
        # Filtrar contornos basados en el tamaño y la relación de aspecto
        # Altura y anchura típicas de un carácter de placa
        h_img, w_img = binary.shape
        min_h, max_h = int(h_img * 0.3), int(h_img * 0.95)
        
        valid_contours = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / float(h)
            
            # Filtro heurístico: los caracteres suelen ser más altos que anchos (aspect_ratio < 1)
            # Y no deben ser ni muy pequeños ni ocupar toda la imagen
            if min_h < h < max_h and 0.2 < aspect_ratio < 1.0:
                valid_contours.append((x, y, w, h))
                cv2.rectangle(debug_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
        # 5. Ordenar contornos de izquierda a derecha
        valid_contours = sorted(valid_contours, key=lambda b: b[0])
        
        # 6. Recortar y redimensionar cada carácter
        for x, y, w, h in valid_contours:
            # Añadir un pequeño padding al recorte
            pad = 2
            x_start = max(0, x - pad)
            y_start = max(0, y - pad)
            x_end = min(w_img, x + w + pad)
            y_end = min(h_img, y + h + pad)
            
            char_crop = binary[y_start:y_end, x_start:x_end]
            
            # Redimensionar a 28x28
            # Para mantener la relación de aspecto, es mejor hacer pad
            char_resized = self._resize_with_pad(char_crop, self.target_size)
            char_crops.append(char_resized)
            
        return char_crops, debug_img

    def _resize_with_pad(self, img, size):
        """
        Redimensiona una imagen a un tamaño específico manteniendo la relación de aspecto
        y rellenando con negro (0).
        """
        h, w = img.shape[:2]
        target_w, target_h = size
        
        # Calcular factor de escala
        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # Crear lienzo negro del tamaño objetivo
        canvas = np.zeros((target_h, target_w), dtype=np.uint8)
        
        # Calcular posición para centrar
        x_offset = (target_w - new_w) // 2
        y_offset = (target_h - new_h) // 2
        
        # Pegar la imagen redimensionada en el centro del lienzo
        canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
        
        return canvas
