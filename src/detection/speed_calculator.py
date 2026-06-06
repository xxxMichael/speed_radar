import time


class SpeedCalculator:
    """
    Calcula la velocidad de vehiculos midiendo el tiempo que tardan en cruzar
    dos lineas virtuales paralelas entre si.

    Soporta dos modos segun la direccion del movimiento del vehiculo:
    - axis='y' (por defecto): lineas HORIZONTALES, vehiculos se mueven arriba/abajo.
                              Mide usando la coordenada Y del centroide.
    - axis='x':               lineas VERTICALES,   vehiculos se mueven izquierda/derecha.
                              Mide usando la coordenada X del centroide.
    """

    def __init__(
        self,
        line1_pos: int,
        line2_pos: int,
        distance_meters: float,
        direction: str = 'down',
        axis: str = 'y',
    ):
        """
        Inicializa el calculador de velocidad.

        Args:
            line1_pos (int): Posicion de la primera linea en pixeles.
                             Y-pixel si axis='y', X-pixel si axis='x'.
            line2_pos (int): Posicion de la segunda linea en pixeles.
            distance_meters (float): Distancia real en metros entre las dos lineas.
            direction (str): Direccion del movimiento del vehiculo.
                             'down'  -> eje Y, de arriba hacia abajo (Y crece).
                             'up'    -> eje Y, de abajo hacia arriba (Y decrece).
                             'right' -> eje X, de izquierda a derecha (X crece).
                             'left'  -> eje X, de derecha a izquierda (X decrece).
            axis (str): 'y' para lineas horizontales (movimiento vertical),
                        'x' para lineas verticales (movimiento horizontal).
        """
        # Garantizar que line1 sea siempre la de menor valor (mas arriba o mas a la izq.)
        self.line1_pos      = min(line1_pos, line2_pos)
        self.line2_pos      = max(line1_pos, line2_pos)
        self.distance_meters = distance_meters
        self.direction      = direction
        self.axis           = axis  # 'x' o 'y'

        # Historial de cada vehiculo trackeado
        # {track_id: {'history': [pos1, pos2, ...], 't1': float, 't2': float, 'speed': float}}
        self.vehicle_data: dict[int, dict] = {}

    # ------------------------------------------------------------------
    # Propiedades de compatibilidad (para que main.py no se rompa)
    # ------------------------------------------------------------------

    @property
    def line1_y(self):
        return self.line1_pos

    @property
    def line2_y(self):
        return self.line2_pos

    # ------------------------------------------------------------------
    # API publica
    # ------------------------------------------------------------------

    def update(self, tracked_vehicles: list, current_time: float = None) -> dict:
        """
        Actualiza el estado de los vehiculos y calcula velocidad cuando cruzan ambas lineas.

        Args:
            tracked_vehicles (list): Salida de VehicleTracker.process_frame().
            current_time (float): Timestamp actual. Si None, usa time.time().

        Returns:
            dict: {track_id: speed_kmh} de los vehiculos que obtuvieron velocidad en este frame.
        """
        if current_time is None:
            current_time = time.time()

        speeds_calculated_now: dict[int, float] = {}

        for vehicle in tracked_vehicles:
            track_id  = vehicle['track_id']
            cx, cy    = vehicle['centroid']
            pos       = cx if self.axis == 'x' else cy   # Coordenada relevante segun eje

            if track_id not in self.vehicle_data:
                self.vehicle_data[track_id] = {
                    'history': [],
                    't1': None,
                    't2': None,
                    'speed': None,
                }

            data = self.vehicle_data[track_id]
            data['history'].append(pos)

            # Mantener solo los ultimos 5 valores
            if len(data['history']) > 5:
                data['history'].pop(0)

            if len(data['history']) >= 2:
                prev_pos = data['history'][-2]
                curr_pos = data['history'][-1]

                # Cruce de Linea 1
                if data['t1'] is None and self._crossed(prev_pos, curr_pos, self.line1_pos):
                    data['t1'] = current_time

                # Cruce de Linea 2 (solo si ya cruzo la linea 1)
                if data['t1'] is not None and data['t2'] is None and self._crossed(prev_pos, curr_pos, self.line2_pos):
                    data['t2'] = current_time

            # Calcular velocidad si se tienen ambos timestamps
            if data['t1'] is not None and data['t2'] is not None and data['speed'] is None:
                time_diff = abs(data['t2'] - data['t1'])
                if time_diff > 0:
                    speed_mps         = self.distance_meters / time_diff
                    speed_kmh         = speed_mps * 3.6
                    data['speed']     = speed_kmh
                    speeds_calculated_now[track_id] = speed_kmh

        return speeds_calculated_now

    def get_speed(self, track_id: int) -> float | None:
        """Retorna la velocidad calculada para un track_id, o None si aun no se tiene."""
        data = self.vehicle_data.get(track_id)
        return data['speed'] if data else None

    def reset(self):
        """Limpia todos los datos de vehiculos trackeados."""
        self.vehicle_data.clear()

    # ------------------------------------------------------------------
    # Metodo privado de cruce de linea
    # ------------------------------------------------------------------

    def _crossed(self, pos_prev: int, pos_curr: int, line: int) -> bool:
        """
        Verifica si el vehiculo cruzo la linea entre el frame anterior y el actual.
        Funciona para ambos ejes (x e y) y ambas direcciones.
        """
        if self.direction in ('down', 'right'):
            return pos_prev <= line < pos_curr
        else:  # 'up' o 'left'
            return pos_prev >= line > pos_curr
