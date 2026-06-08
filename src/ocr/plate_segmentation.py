import cv2
import numpy as np


class PlateSegmenter:
    """
    Clase para segmentar caracteres individuales de una imagen recortada de una placa
    vehicular utilizando técnicas clásicas de Visión Computacional (OpenCV).

    Pipeline mejorado para placas ecuatorianas (fondo verde / blanco, texto negro):
      1. Estandarizar altura a 100 px
      2. Convertir a gris
      3. CLAHE (contraste local)
      4. Filtro Gaussiano ligero para suprimir micro-ruido antes del bilateral
      5. Filtro Bilateral (preserva bordes)
      6. Sharpening con kernel unsharp-mask para realzar trazos
      7. Binarización: Otsu global + Adaptativa Gaussiana → OR para capturar lo mejor de ambas
      8. Morfología: apertura (elimina ruido fino) + cierre (une trazos partidos)
      9. Extracción de contornos con filtros geométricos y eliminación de anidados
    """

    def __init__(self, target_size=(28, 28)):
        """
        Args:
            target_size (tuple): Tamaño de salida para cada carácter segmentado (ancho, alto).
                                 28x28 es el estándar para EMNIST.
        """
        self.target_size = target_size

    # ------------------------------------------------------------------
    # API pública
    # ------------------------------------------------------------------

    def segment_characters(self, plate_img: np.ndarray) -> tuple[list[np.ndarray], np.ndarray, dict]:
        """
        Toma el recorte de la placa a color, lo binariza, aisla los contornos
        de cada caracter, los ordena de izquierda a derecha, y los recorta.

        Args:
            plate_img: Imagen BGR de la placa recortada.

        Returns:
            - Lista de imágenes de los caracteres (28x28, escala de grises, fondo negro).
            - Imagen de debug con bounding boxes dibujados sobre la placa original.
            - Diccionario con las imágenes de cada fase (gris, blur, binarización, etc.).
        """
        if plate_img is None or plate_img.size == 0:
            return [], None, {}

        # ── 1. Estandarizar tamaño (altura fija = 100 px) ──────────────────
        ph, pw = plate_img.shape[:2]
        target_h = 100
        scale = target_h / ph
        plate_img = cv2.resize(plate_img, (int(pw * scale), target_h),
                               interpolation=cv2.INTER_CUBIC)

        # ── 2. Convertir a escala de grises ─────────────────────────────────
        gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)

        # ── 3. CLAHE — mejora local del contraste ───────────────────────────
        # clipLimit bajo (2.0) para no sobreenfatizar el ruido de fondo verde/blanco
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
        gray = clahe.apply(gray)

        # ── 4. Gaussian ligero — suprime micro-ruido previo al bilateral ─────
        gauss = cv2.GaussianBlur(gray, (3, 3), 0)

        # ── 5. Filtro Bilateral — reduce ruido pero preserva bordes de letra ─
        # d=9, sigmaColor=30, sigmaSpace=30: más suave que antes para no mezclar borde
        blur = cv2.bilateralFilter(gauss, 9, 30, 30)

        # ── 6. Unsharp Mask — realza trazos de los caracteres ────────────────
        # sharp = original + alpha*(original − blur)
        # Usamos sigma alto (5) para que el blur base del unsharp no confunda ruido fino
        blur_for_sharp = cv2.GaussianBlur(blur, (0, 0), sigmaX=5)
        sharp = cv2.addWeighted(blur, 1.8, blur_for_sharp, -0.8, 0)

        # ── 7. Binarización dual (Otsu global + Adaptativa) ──────────────────
        # Otsu: excelente cuando el histograma tiene dos modos claros (letra oscura / fondo claro)
        _, bin_otsu_inv = cv2.threshold(sharp, 0, 255,
                                        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        _, bin_otsu_nor = cv2.threshold(sharp, 0, 255,
                                        cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Adaptativa: captura letras en zonas con iluminación desigual.
        # Bloque más grande (21) y C más alto (9) → menos ruido de fondo
        bin_ada_inv = cv2.adaptiveThreshold(sharp, 255,
                                            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                            cv2.THRESH_BINARY_INV, 21, 9)
        bin_ada_nor = cv2.adaptiveThreshold(sharp, 255,
                                            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                            cv2.THRESH_BINARY, 21, 9)

        # Combinar: AND entre Otsu y Adaptativa → conserva solo lo que ambas acuerdan
        # Esto elimina los artefactos de fondo que la adaptativa introduce
        bin_inv = cv2.bitwise_and(bin_otsu_inv, bin_ada_inv)
        bin_nor = cv2.bitwise_and(bin_otsu_nor, bin_ada_nor)

        # ── 8. Morfología: apertura → elimina ruido, cierre → une trazos ────
        # Apertura pequeña (2,2) para no erosionar dígitos finos como '1' o 'I'
        kernel_open  = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        # Cierre un poco más alto que ancho (2,4) para unir trazos verticales partidos
        kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 4))

        bin_inv = cv2.morphologyEx(bin_inv, cv2.MORPH_OPEN,  kernel_open)
        bin_inv = cv2.morphologyEx(bin_inv, cv2.MORPH_CLOSE, kernel_close)
        bin_nor = cv2.morphologyEx(bin_nor, cv2.MORPH_OPEN,  kernel_open)
        bin_nor = cv2.morphologyEx(bin_nor, cv2.MORPH_CLOSE, kernel_close)

        # ── 9. Seleccionar la binarización con más contornos válidos ─────────
        valid_inv = self._extract_contours(bin_inv)
        valid_nor = self._extract_contours(bin_nor)

        if len(valid_inv) >= len(valid_nor):
            binary         = bin_inv
            valid_contours = valid_inv
        else:
            binary         = bin_nor
            valid_contours = valid_nor

        # Ordenar de izquierda a derecha
        valid_contours = sorted(valid_contours, key=lambda b: b[0])

        # Debug: dibujar bboxes sobre la imagen de placa a color
        debug_img = plate_img.copy()
        for x, y, w, h in valid_contours:
            cv2.rectangle(debug_img, (x, y), (x + w, y + h), (0, 255, 0), 1)

        # ── 10. Recortar y redimensionar cada carácter ────────────────────────
        h_img, w_img = binary.shape
        char_crops = []
        for x, y, w, h in valid_contours:
            pad     = 2
            x_start = max(0, x - pad)
            y_start = max(0, y - pad)
            x_end   = min(w_img, x + w + pad)
            y_end   = min(h_img, y + h + pad)
            char_crop = binary[y_start:y_end, x_start:x_end]

            # Garantizar letra blanca sobre fondo negro (convención de la CNN)
            borders = np.concatenate([char_crop[0, :], char_crop[-1, :],
                                      char_crop[:, 0], char_crop[:, -1]])
            if np.mean(borders) > 127:
                char_crop = cv2.bitwise_not(char_crop)

            char_resized = self._resize_with_pad(char_crop, self.target_size)
            char_crops.append(char_resized)

        stages_dict = {
            "Placa Original (Color)": plate_img,
            "Gris Original":           gray,
            "Gris + Gaussian":         gauss,
            "Filtro Bilateral":        blur,
            "Sharpening":              sharp,
            "Otsu Inv":                bin_otsu_inv,
            "Adaptativo Inv":          bin_ada_inv,
            "Binarización Final":      binary,
            "Segmentación Final":      debug_img,
        }

        return char_crops, debug_img, stages_dict

    # ------------------------------------------------------------------
    # Métodos privados
    # ------------------------------------------------------------------

    def _extract_contours(self, binary: np.ndarray) -> list[tuple]:
        """
        Extrae contornos válidos de la imagen binaria.
        Filtra el marco de la placa, ruido pequeño, y contornos anidados (huecos de 'O', 'B').

        Returns:
            Lista de (x, y, w, h) válidos.
        """
        h_img, w_img = binary.shape

        # Rango de altura válida: 30%–90% de la altura de la placa.
        # 30% captura letras con margen generoso; 90% descarta el marco completo.
        min_h = max(10, int(h_img * 0.30))
        max_h = int(h_img * 0.90)

        contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        valid = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)

            # Descartar contornos que abarcan casi todo el ancho (marco)
            if w > w_img * 0.85 or h > h_img * 0.95:
                continue

            # Descartar si toca bordes superior/inferior Y es muy ancho (borde de placa)
            if y <= 2 or (y + h) >= h_img - 2:
                if w > w_img * 0.5:
                    continue

            # Descartar si toca los bordes laterales (franja negra lateral del recorte)
            margin_x = max(3, int(w_img * 0.04))
            if x <= margin_x or (x + w) >= w_img - margin_x:
                continue

            # Filtro de altura
            if not (min_h <= h <= max_h):
                continue

            # Ancho mínimo razonable
            if w < 4:
                continue

            # Relación de aspecto: letras/dígitos de placa van de ~0.2 ('I') a ~1.2 ('W')
            aspect = w / float(h)
            if not (0.18 <= aspect <= 1.3):
                continue

            valid.append((x, y, w, h))

        # Eliminar contornos internos (huecos como el de 'O', 'B', 'D')
        filtered = []
        for box in valid:
            x, y, w, h = box
            is_nested = False
            for other in valid:
                if other == box:
                    continue
                ox, oy, ow, oh = other
                # El otro contiene completamente a este y es mayor
                if ox <= x and oy <= y and (ox + ow) >= (x + w) and (oy + oh) >= (y + h):
                    if ow > w or oh > h:
                        is_nested = True
                        break
            if not is_nested:
                filtered.append(box)

        return filtered

    def _resize_with_pad(self, img: np.ndarray, size: tuple) -> np.ndarray:
        """
        Redimensiona una imagen a 'size' manteniendo la relación de aspecto
        y rellenando con negro (0).  Deja ~4 px de margen por lado (estándar EMNIST).
        """
        h, w = img.shape[:2]
        target_w, target_h = size

        # Área interna con margen de 4 px por lado
        inner_w, inner_h = target_w - 8, target_h - 8
        scale   = min(inner_w / float(w), inner_h / float(h))
        new_w   = max(1, int(w * scale))
        new_h   = max(1, int(h * scale))

        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # Lienzo negro
        canvas   = np.zeros((target_h, target_w), dtype=np.uint8)
        x_offset = (target_w - new_w) // 2
        y_offset = (target_h - new_h) // 2
        canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized

        return canvas
