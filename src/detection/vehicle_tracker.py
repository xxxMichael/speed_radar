import cv2
import torch
from ultralytics import YOLO


class VehicleTracker:
    """
    Detecta y hace tracking de vehiculos usando YOLOv8 + ByteTrack.

    Clases COCO detectadas por defecto:
        2  -> car
        3  -> motorcycle
        5  -> bus
        7  -> truck
    """

    def __init__(
        self,
        model_path: str = 'yolov8n.pt',
        tracker: str = 'bytetrack.yaml',
        conf: float = 0.15,
        iou: float = 0.45,
    ):
        """
        Args:
            model_path (str): Ruta al modelo YOLOv8 (se descarga automaticamente si no existe).
            tracker (str): Configuracion de tracker ('bytetrack.yaml' o 'botsort.yaml').
            conf (float): Umbral de confianza minima para deteccion (0.0 - 1.0).
                          Bajar este valor detecta vehiculos en condiciones dificiles
                          (angulo inusual, distancia, poca luz) a costa de mas falsos positivos.
            iou (float): Umbral de IoU para Non-Maximum Suppression.
        """
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"[INFO] Cargando modelo YOLO: {model_path} en dispositivo: {self.device}")
        self.model          = YOLO(model_path)
        self.tracker_config = tracker
        self.conf           = conf
        self.iou            = iou

        # Clases COCO consideradas "vehiculos"
        self.vehicle_classes = [2, 3, 5, 7]

    def process_frame(self, frame) -> tuple[list[dict], any]:
        """
        Detecta y trackea vehiculos en un frame.

        Args:
            frame (numpy.ndarray): Frame BGR de la camara.

        Returns:
            list[dict]: Vehiculos trackeados. Cada elemento contiene:
                        {'track_id': int, 'bbox': [x1,y1,x2,y2],
                         'class_id': int, 'class_name': str,
                         'confidence': float, 'centroid': (cx, cy)}
            numpy.ndarray: Frame con anotaciones dibujadas por YOLO.
        """
        results = self.model.track(
            frame,
            persist=True,
            tracker=self.tracker_config,
            classes=self.vehicle_classes,
            conf=self.conf,
            iou=self.iou,
            verbose=False,
            device=self.device,
        )

        tracked_vehicles = []
        annotated_frame  = results[0].plot()

        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes       = results[0].boxes.xyxy.cpu().numpy()
            track_ids   = results[0].boxes.id.int().cpu().numpy()
            class_ids   = results[0].boxes.cls.int().cpu().numpy()
            confidences = results[0].boxes.conf.cpu().numpy()
            names       = results[0].names

            for box, track_id, class_id, conf in zip(boxes, track_ids, class_ids, confidences):
                x1, y1, x2, y2 = map(int, box)
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2

                tracked_vehicles.append({
                    'track_id':   int(track_id),
                    'bbox':       [x1, y1, x2, y2],
                    'class_id':   int(class_id),
                    'class_name': names[int(class_id)],
                    'confidence': float(conf),
                    'centroid':   (cx, cy),
                })

        return tracked_vehicles, annotated_frame

    def detect_all(self, frame) -> tuple[int, any]:
        """
        Modo diagnostico: detecta TODOS los objetos (sin filtro de clases) para
        verificar que YOLO esta funcionando y ver que ve en la escena.

        Returns:
            int: Numero de objetos detectados.
            numpy.ndarray: Frame anotado con todas las detecciones.
        """
        results = self.model.predict(frame, conf=self.conf, verbose=False, device=self.device)
        return len(results[0].boxes), results[0].plot()
