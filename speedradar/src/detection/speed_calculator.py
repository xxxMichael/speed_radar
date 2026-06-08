import time
import math


class SpeedCalculator:
    """
    Calcula la velocidad de vehiculos usando desplazamiento de trayectoria
    multi-frame (trajectory-based).

    En lugar de requerir que un vehiculo cruce dos lineas exactas (lo que falla
    frecuentemente por perdida de tracking), este algoritmo:

    1. Usa las dos lineas para definir una ZONA DE MEDICION y para calibrar
       la relacion pixeles -> metros.
    2. Calcula velocidad continuamente a partir del desplazamiento del centroide
       en los ultimos N frames.
    3. Reporta la velocidad cuando el vehiculo esta DENTRO de la zona de medicion
       y se han acumulado suficientes observaciones (MIN_SAMPLES).

    Soporta dos modos segun la direccion del movimiento del vehiculo:
    - axis='y': lineas HORIZONTALES, vehiculos se mueven arriba/abajo.
    - axis='x': lineas VERTICALES,   vehiculos se mueven izquierda/derecha.
    """

    # Configuracion de la ventana de observacion
    MIN_SAMPLES      = 5     # Frames minimos para calcular velocidad (~0.17s a 30fps)
    MAX_HISTORY      = 15    # Maximo de muestras en la ventana deslizante
    MIN_DISPLACEMENT = 8     # Pixeles minimos de desplazamiento para considerar que hay movimiento
    MAX_SPEED_KMH    = 300   # Velocidad maxima razonable (filtro anti-glitch)

    def __init__(
        self,
        line1_pos: int,
        line2_pos: int,
        distance_meters: float,
        direction: str = 'down',
        axis: str = 'y',
        fps: float = 30.0,
    ):
        """
        Inicializa el calculador de velocidad por trayectoria.

        Args:
            line1_pos (int): Posicion de la primera linea en pixeles.
            line2_pos (int): Posicion de la segunda linea en pixeles.
            distance_meters (float): Distancia real en metros entre las dos lineas.
            direction (str): Direccion esperada del movimiento (para compatibilidad).
            axis (str): 'y' para lineas horizontales, 'x' para lineas verticales.
            fps (float): Frames por segundo de la fuente de video (para timestamps
                         precisos cuando se usa archivo de video).
        """
        # Garantizar que line1 sea siempre la de menor valor
        self.line1_pos       = min(line1_pos, line2_pos)
        self.line2_pos       = max(line1_pos, line2_pos)
        self.distance_meters = distance_meters
        self.direction       = direction
        self.axis            = axis
        self.fps             = fps

        # Calibracion: metros por pixel
        pixel_separation = abs(self.line2_pos - self.line1_pos)
        if pixel_separation > 0:
            self.meters_per_pixel = distance_meters / pixel_separation
        else:
            self.meters_per_pixel = 0.01  # fallback

        # Historial de cada vehiculo trackeado
        # {track_id: {
        #     'positions': [(pos, timestamp), ...],  # ventana deslizante
        #     'speed': float | None,                 # ultima velocidad calculada
        #     'reported': bool,                      # ya se reporto?
        #     'in_zone': bool,                       # esta dentro de la zona?
        # }}
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
        Actualiza el estado de los vehiculos y calcula velocidades por trayectoria.

        Args:
            tracked_vehicles (list): Salida de VehicleTracker.process_frame().
            current_time (float): Timestamp actual. Si None, usa time.time().

        Returns:
            dict: {track_id: speed_kmh} de los vehiculos que se reportan en este frame.
        """
        if current_time is None:
            current_time = time.time()

        speeds_reported_now: dict[int, float] = {}

        for vehicle in tracked_vehicles:
            track_id = vehicle['track_id']
            cx, cy   = vehicle['centroid']
            pos      = cx if self.axis == 'x' else cy

            # Inicializar si es nuevo
            if track_id not in self.vehicle_data:
                self.vehicle_data[track_id] = {
                    'positions': [],
                    'speed': None,
                    'reported': False,
                    'in_zone': False,
                }

            data = self.vehicle_data[track_id]

            # Acumular posicion + timestamp
            data['positions'].append((pos, current_time))

            # Mantener ventana deslizante
            if len(data['positions']) > self.MAX_HISTORY:
                data['positions'].pop(0)

            # Verificar si el vehiculo esta dentro de la zona de medicion
            in_zone = self.line1_pos <= pos <= self.line2_pos
            data['in_zone'] = in_zone

            # Calcular velocidad si tenemos suficientes muestras
            if len(data['positions']) >= self.MIN_SAMPLES:
                speed_kmh = self._calculate_speed(data['positions'])

                if speed_kmh is not None:
                    data['speed'] = speed_kmh

                    # Reportar si esta en zona y no se ha reportado aun
                    if in_zone and not data['reported'] and speed_kmh > 0:
                        data['reported'] = True
                        speeds_reported_now[track_id] = speed_kmh

        return speeds_reported_now

    def get_speed(self, track_id: int) -> float | None:
        """Retorna la velocidad calculada para un track_id, o None si aun no se tiene."""
        data = self.vehicle_data.get(track_id)
        return data['speed'] if data else None

    def is_in_zone(self, track_id: int) -> bool:
        """Retorna True si el vehiculo esta actualmente dentro de la zona de medicion."""
        data = self.vehicle_data.get(track_id)
        return data['in_zone'] if data else False

    def reset(self):
        """Limpia todos los datos de vehiculos trackeados."""
        self.vehicle_data.clear()

    # ------------------------------------------------------------------
    # Metodos privados
    # ------------------------------------------------------------------

    def _calculate_speed(self, positions: list[tuple[int, float]]) -> float | None:
        """
        Calcula velocidad en km/h a partir del desplazamiento del centroide
        en la ventana de observacion.

        Usa regresion lineal simple (posicion vs tiempo) para obtener una
        estimacion robusta de la velocidad, filtrada contra ruido y jitter.

        Args:
            positions: Lista de (posicion_px, timestamp) ordenada cronologicamente.

        Returns:
            float | None: Velocidad en km/h, o None si no es calculable.
        """
        if len(positions) < self.MIN_SAMPLES:
            return None

        # Extraer posiciones y tiempos
        pos_values = [p[0] for p in positions]
        t_values   = [p[1] for p in positions]

        # Rango de tiempo
        dt = t_values[-1] - t_values[0]
        if dt <= 0:
            return None

        # Desplazamiento total (absoluto)
        total_displacement = abs(pos_values[-1] - pos_values[0])

        # Filtrar vehiculos estacionarios / ruido
        if total_displacement < self.MIN_DISPLACEMENT:
            return None

        # Regresion lineal: pos = slope * t + intercept
        # slope = velocidad en pixeles/segundo
        n = len(positions)
        t0 = t_values[0]
        sum_t  = sum(t - t0 for t in t_values)
        sum_p  = sum(pos_values)
        sum_tt = sum((t - t0) ** 2 for t in t_values)
        sum_tp = sum((t - t0) * p for t, p in zip(t_values, pos_values))

        denominator = n * sum_tt - sum_t ** 2
        if abs(denominator) < 1e-9:
            return None

        slope = (n * sum_tp - sum_t * sum_p) / denominator  # px/s

        # Velocidad absoluta en m/s -> km/h
        speed_px_per_s = abs(slope)
        speed_m_per_s  = speed_px_per_s * self.meters_per_pixel
        speed_kmh      = speed_m_per_s * 3.6

        # Filtro anti-glitch
        if speed_kmh > self.MAX_SPEED_KMH:
            return None

        return round(speed_kmh, 1)
