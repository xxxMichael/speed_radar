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

        # 1. Redimensionar si la placa es muy pequeña (mejora el OCR)
        ph, pw = plate_img.shape[:2]
        if pw < 200:
            scale  = 200.0 / pw
            plate_img = cv2.resize(plate_img, (200, int(ph * scale)), interpolation=cv2.INTER_CUBIC)

        # 2. Convertir a escala de grises
        gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)

        # 3. CLAHE: mejora local del contraste (critico para placas con iluminacion desigual)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
        gray  = clahe.apply(gray)

        # 4. Suavizado ligero
        blur = cv2.GaussianBlur(gray, (3, 3), 0)

        # 5. Binarizacion dual: probamos Otsu normal e inverso y elegimos el que da mas caracteres.
        # Las placas ecuatorianas son blancas con texto negro → BINARY_INV da letras blancas.
        _, binary_inv = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        _, binary_nor = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY     + cv2.THRESH_OTSU)

        # Operacion morfologica: cierre pequeno para unir partes de letras rotas
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        binary_inv = cv2.morphologyEx(binary_inv, cv2.MORPH_CLOSE, kernel)
        binary_nor = cv2.morphologyEx(binary_nor, cv2.MORPH_CLOSE, kernel)

        def _extract_contours(binary):
            """Retorna lista de (x, y, w, h) de caracteres validos, filtrando bordes e interiores."""
            h_img, w_img = binary.shape
            # Altura minima: 20% de la imagen; maxima: 95%
            min_h = max(5, int(h_img * 0.20))
            max_h = int(h_img * 0.97)
            contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            valid = []
            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)
                # Evitar contorno del borde exterior de la placa o marcos grandes
                if w > w_img * 0.85 or h > h_img * 0.85:
                    continue
                if not (min_h < h < max_h):
                    continue
                if w < 3:
                    continue
                aspect = w / float(h)
                # Rango de aspecto ampliado: cubre letras anchas (B, P, D) y el guion
                if 0.15 < aspect < 2.5:
                    valid.append((x, y, w, h))
            
            # Eliminar contornos internos (anidados/agujeros como el de la 'O' o 'B')
            # Si un contorno esta completamente contenido dentro de otro mas grande, lo descartamos
            filtered_valid = []
            for box in valid:
                x, y, w, h = box
                is_nested = False
                for other in valid:
                    if other == box:
                        continue
                    ox, oy, ow, oh = other
                    if ox <= x and oy <= y and (ox + ow) >= (x + w) and (oy + oh) >= (y + h):
                        if ow > w or oh > h:
                            is_nested = True
                            break
                if not is_nested:
                    filtered_valid.append(box)
                    
            return filtered_valid

        valid_inv = _extract_contours(binary_inv)
        valid_nor = _extract_contours(binary_nor)

        # Usar la binarizacion que produjo mas caracteres plausibles
        if len(valid_inv) >= len(valid_nor):
            binary        = binary_inv
            valid_contours = valid_inv
        else:
            binary        = binary_nor
            valid_contours = valid_nor

        # Ordenar de izquierda a derecha
        valid_contours = sorted(valid_contours, key=lambda b: b[0])

        # Debug: dibujar bboxes sobre la imagen de placa
        h_img, w_img = binary.shape
        debug_img = plate_img.copy()
        for x, y, w, h in valid_contours:
            cv2.rectangle(debug_img, (x, y), (x + w, y + h), (0, 255, 0), 1)

        # 6. Recortar y redimensionar cada caracter
        char_crops = []
        for x, y, w, h in valid_contours:
            pad    = 2
            x_start = max(0, x - pad)
            y_start = max(0, y - pad)
            x_end   = min(w_img, x + w + pad)
            y_end   = min(h_img, y + h + pad)
            char_crop    = binary[y_start:y_end, x_start:x_end]
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
