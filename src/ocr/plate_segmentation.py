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

        # 1. Estandarizar el tamaño de la placa a una altura fija para consistencia de parámetros
        ph, pw = plate_img.shape[:2]
        target_h = 100
        scale = target_h / ph
        plate_img = cv2.resize(plate_img, (int(pw * scale), target_h), interpolation=cv2.INTER_CUBIC)

        # 2. Convertir a escala de grises
        gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)

        # 3. CLAHE: mejora local del contraste (critico para placas con iluminacion desigual)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        gray  = clahe.apply(gray)

        # 4. Suavizado ligero (Bilateral filter reduce ruido pero preserva bordes)
        blur = cv2.bilateralFilter(gray, 11, 17, 17)

        # 5. Binarizacion adaptativa: ideal para fondos con contraste pobre (placas naranja/amarillo) o luces.
        # Bloque de 19, constante C de 10 funciona bien para texto.
        binary_inv = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 19, 10)
        binary_nor = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 19, 10)

        # Operacion morfologica: cierre pequeno para unir partes de letras rotas
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        binary_inv = cv2.morphologyEx(binary_inv, cv2.MORPH_CLOSE, kernel)
        binary_nor = cv2.morphologyEx(binary_nor, cv2.MORPH_CLOSE, kernel)

        def _extract_contours(binary):
            """Retorna lista de (x, y, w, h) de caracteres validos, filtrando bordes e interiores."""
            h_img, w_img = binary.shape
            # Los caracteres reales de la placa ocupan la mayor parte del espacio vertical.
            # Incrementamos min_h al 30% (antes 40%) para evitar perder letras si YOLO recorta con mucho margen,
            # pero manteniendolo suficiente para descartar palabras pequeñas como "ECUADOR".
            min_h = max(10, int(h_img * 0.30))
            max_h = int(h_img * 0.95)
            contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            valid = []
            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)
                
                # Evitar contorno del borde exterior o líneas horizontales/verticales del marco
                if w > w_img * 0.85 or h > h_img * 0.95:
                    continue
                
                # Filtrar si el contorno está pegado al borde superior o inferior (suele ser el borde de la placa)
                if y < 2 or (y + h) > h_img - 2:
                    # Si toca el borde y es muy ancho, probablemente sea el marco
                    if w > w_img * 0.5:
                        continue
                        
                if not (min_h <= h <= max_h):
                    continue
                if w < 3:
                    continue
                aspect = w / float(h)
                # Rango de aspecto más relajado para letras con perspectiva (0.15 a 1.5)
                # Un carácter no suele ser 3 veces más ancho que alto.
                if 0.15 < aspect < 1.5:
                    valid.append((x, y, w, h))
            
            # Eliminar contornos internos (agujeros como el de la 'O' o 'B')
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
            
            # Asegurar que el caracter sea letra blanca sobre fondo negro (para la CNN)
            # Analizamos los bordes del recorte. Si la mayoría es blanco, invertimos.
            borders = np.concatenate([char_crop[0,:], char_crop[-1,:], char_crop[:,0], char_crop[:,-1]])
            if np.mean(borders) > 127:
                char_crop = cv2.bitwise_not(char_crop)
                
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
