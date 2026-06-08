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

    def segment_characters(self, plate_img: np.ndarray) -> tuple[list[np.ndarray], np.ndarray, dict]:
        """
        Toma el recorte de la placa a color, lo binariza, aisla los contornos
        de cada caracter, los ordena de izquierda a derecha, y los recorta.
        
        Args:
            plate_img: Imagen BGR de la placa recortada.
            
        Returns:
            - Lista de imagenes de los caracteres (28x28, escala de grises, fondo negro).
            - Imagen de debug con bounding boxes dibujados sobre la placa original.
            - Diccionario con las imagenes de cada fase (gris, blur, binarizacion).
        """
        if plate_img is None or plate_img.size == 0:
            return [], None, {}

        # 1. Estandarizar el tamaño de la placa a una altura fija para consistencia de parámetros
        ph, pw = plate_img.shape[:2]
        target_h = 100
        scale = target_h / ph
        plate_img = cv2.resize(plate_img, (int(pw * scale), target_h), interpolation=cv2.INTER_CUBIC)

        # 2. Convertir a escala de grises
        gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)

        # 3. CLAHE: mejora local del contraste (critico para placas con iluminacion desigual).
        # clipLimit moderado (2.0): un valor alto amplifica el ruido del sensor del celular
        # y lo convierte en "motas" que ensucian la binarizacion ("filtros raros").
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray  = clahe.apply(gray)

        # 4. Suavizado que preserva bordes. Bilateral mas ligero (d=7) = menos ruido y mas rapido.
        blur = cv2.bilateralFilter(gray, 7, 50, 50)

        # 5. Tres estrategias de binarizacion; mas abajo se elige la mas plausible.
        #    - Otsu invertido: limpio cuando la placa esta bien recortada y con buen contraste.
        #    - Adaptativo invertido: robusto a iluminacion desigual (sombras, reflejos).
        #    - Adaptativo normal: para placas con texto claro sobre fondo oscuro.
        _, otsu_inv = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        adap_inv = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 6)
        adap_nor = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 6)

        # Limpieza: apertura (quita motas/ruido = los "filtros raros") + cierre (une trazos partidos).
        open_k  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        close_k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 5))

        def _clean(b):
            b = cv2.morphologyEx(b, cv2.MORPH_OPEN,  open_k)   # elimina motas pequenas
            b = cv2.morphologyEx(b, cv2.MORPH_CLOSE, close_k)  # une caracteres partidos por tornillos
            return b

        otsu_inv = _clean(otsu_inv)
        adap_inv = _clean(adap_inv)
        adap_nor = _clean(adap_nor)

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
                if y <= 2 or (y + h) >= h_img - 2:
                    # Si toca el borde y es muy ancho, probablemente sea el marco
                    if w > w_img * 0.5:
                        continue
                        
                # Filtrar si está tocando los bordes laterales (evita la franja negra lateral).
                # Usamos un porcentaje del ancho en lugar de pixeles absolutos para placas lejanas.
                margin_x = max(3, int(w_img * 0.04))
                if x <= margin_x or (x + w) >= w_img - margin_x:
                    continue
                        
                if not (min_h <= h <= max_h):
                    continue
                if w < 3:
                    continue
                aspect = w / float(h)
                # Rango de aspecto más relajado para letras con perspectiva.
                # Subimos el mínimo a 0.18 para evitar líneas de ruido que se leen como 'I'.
                if 0.18 < aspect < 1.5:
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

        variants     = [otsu_inv, adap_inv, adap_nor]
        contour_sets = [_extract_contours(v) for v in variants]

        # Elegir la binarizacion cuyo numero de caracteres sea MAS PLAUSIBLE para una placa.
        # Antes se elegia "la que tenga mas contornos", lo que premiaba a la binarizacion
        # mas ruidosa (mas falsos caracteres). Una placa tipica tiene 6-7 caracteres, asi que
        # puntuamos por cercania a 7 y penalizamos el exceso (ruido).
        def _plausibility(boxes):
            n = len(boxes)
            if n == 0:
                return -100
            return -abs(n - 7)

        best_i = max(range(len(variants)), key=lambda i: _plausibility(contour_sets[i]))
        binary         = variants[best_i]
        valid_contours = contour_sets[best_i]

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

        stages_dict = {
            "Placa Original": plate_img,
            "Gris + CLAHE": gray,
            "Filtro Bilateral": blur,
            "Binarización (mejor variante)": binary,
            "Segmentación Final": debug_img
        }

        return char_crops, debug_img, stages_dict

    def _resize_with_pad(self, img, size):
        """
        Redimensiona una imagen a un tamaño específico manteniendo la relación de aspecto
        y rellenando con negro (0).
        """
        h, w = img.shape[:2]
        target_w, target_h = size
        
        # Calcular factor de escala para dejar un margen (ej: max dimension 20)
        # EMNIST y el modelo sintético usan caracteres centrados dejando ~4px de margen por lado
        inner_w, inner_h = target_w - 8, target_h - 8
        
        scale = min(inner_w / float(w), inner_h / float(h))
        new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
        
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # Crear lienzo negro del tamaño objetivo
        canvas = np.zeros((target_h, target_w), dtype=np.uint8)
        
        # Calcular posición para centrar
        x_offset = (target_w - new_w) // 2
        y_offset = (target_h - new_h) // 2
        
        # Pegar la imagen redimensionada en el centro del lienzo
        canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
        
        return canvas
