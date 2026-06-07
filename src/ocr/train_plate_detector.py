"""
Script para fine-tuning de YOLOv8 en deteccion de placas vehiculares.

Usa el dataset ALPR (Automatic License Plate Recognition) en formato YOLO.
Al final del entrenamiento, copia el mejor modelo a models/plate_detector_yolov8n.pt
para que el sistema lo use automaticamente.

Estructura esperada del dataset:
    datasets/plate_detector/
        dataset.yaml
        train/
            images/  (*.jpg / *.png)
            labels/  (*.txt en formato YOLO)
        valid/
            images/
            labels/
        test/           (opcional)
            images/
            labels/

Uso:
    .\.venv\Scripts\python.exe src/ocr/train_plate_detector.py
    .\.venv\Scripts\python.exe src/ocr/train_plate_detector.py --data datasets/mi_dataset --epochs 50

"""
import os
import sys
import shutil
import argparse

# Verificar ultralytics
try:
    from ultralytics import YOLO
except ImportError:
    print("[ERROR] ultralytics no instalado. Ejecuta: pip install ultralytics")
    sys.exit(1)


# =============================================================================
# CONFIGURACION
# =============================================================================
DEFAULT_DATASET_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', '..', 'datasets', 'plate_detector'
)
MODEL_SAVE_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', '..', 'models', 'plate_detector_yolov8n.pt'
)
BASE_MODEL      = 'yolov8n.pt'   # Modelo base de Ultralytics (se descarga si no existe)
EPOCHS_DEFAULT  = 50
IMG_SIZE        = 640
BATCH_SIZE      = 16             # Ajustar segun VRAM (RTX 3060: 16-32 es seguro)
# =============================================================================


def create_dataset_yaml(dataset_dir: str) -> str:
    """
    Crea el archivo dataset.yaml para YOLOv8 si no existe.
    Asume que el dataset tiene exactamente 1 clase: 'license_plate'.
    """
    yaml_path = os.path.join(dataset_dir, 'dataset.yaml')

    if os.path.exists(yaml_path):
        print(f"[INFO] dataset.yaml ya existe en: {yaml_path}")
        return yaml_path

    # Detectar carpetas de train y valid
    train_imgs = os.path.join(dataset_dir, 'train', 'images')
    valid_imgs = os.path.join(dataset_dir, 'valid', 'images')

    if not os.path.isdir(train_imgs):
        raise FileNotFoundError(
            f"No se encontro la carpeta de imagenes de entrenamiento: {train_imgs}\n"
            f"Asegurate de colocar el dataset en: {dataset_dir}"
        )

    # Contar imagenes
    n_train = len([f for f in os.listdir(train_imgs) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    n_valid = len([f for f in os.listdir(valid_imgs) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]) if os.path.isdir(valid_imgs) else 0

    print(f"[INFO] Imagenes encontradas: {n_train} train, {n_valid} valid")

    yaml_content = f"""# Dataset de deteccion de placas vehiculares (ALPR)
# Generado automaticamente por train_plate_detector.py

path: {os.path.abspath(dataset_dir)}
train: train/images
val: valid/images
test: test/images

# Clases
nc: 1
names:
  0: license_plate
"""
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)

    print(f"[OK] dataset.yaml creado: {yaml_path}")
    return yaml_path


def train(dataset_dir: str, epochs: int):
    """Ejecuta el fine-tuning de YOLOv8 para deteccion de placas."""

    print("\n" + "=" * 65)
    print("  ENTRENAMIENTO - DETECTOR DE PLACAS VEHICULARES (YOLOv8)")
    print("=" * 65)
    print(f"  Dataset   : {dataset_dir}")
    print(f"  Epocas    : {epochs}")
    print(f"  Batch     : {BATCH_SIZE}")
    print(f"  Img size  : {IMG_SIZE}")
    print(f"  Modelo    : {BASE_MODEL}")
    print("=" * 65 + "\n")

    # Crear dataset.yaml si no existe
    yaml_path = create_dataset_yaml(dataset_dir)

    # Cargar modelo base pre-entrenado en COCO
    print(f"[INFO] Cargando modelo base: {BASE_MODEL}")
    model = YOLO(BASE_MODEL)

    # Fine-tuning
    print(f"[INFO] Iniciando fine-tuning en: {yaml_path}")
    results = model.train(
        data=yaml_path,
        epochs=epochs,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        project='runs/plate_detector',
        name='train',
        exist_ok=True,
        device=0,           # GPU 0 (CUDA). Cambiar a 'cpu' si no hay GPU.
        patience=15,        # Early stopping: detiene si no mejora en 15 epocas
        save=True,
        save_period=10,
        val=True,
        plots=True,
        verbose=True,
    )

    # Copiar mejor modelo al directorio de modelos del proyecto
    best_weights = os.path.join('runs', 'plate_detector', 'train', 'weights', 'best.pt')

    if not os.path.exists(best_weights):
        print(f"[ERROR] No se encontro el mejor modelo en: {best_weights}")
        return

    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    shutil.copy2(best_weights, MODEL_SAVE_PATH)
    print(f"\n[OK] Mejor modelo copiado a: {MODEL_SAVE_PATH}")
    print("[OK] El sistema usara automaticamente este modelo en la proxima ejecucion.")

    # Mostrar metricas finales
    print("\n" + "=" * 65)
    print("  METRICAS FINALES")
    print("=" * 65)
    if hasattr(results, 'results_dict'):
        metrics = results.results_dict
        print(f"  mAP50     : {metrics.get('metrics/mAP50(B)', 'N/A'):.4f}")
        print(f"  mAP50-95  : {metrics.get('metrics/mAP50-95(B)', 'N/A'):.4f}")
        print(f"  Precision : {metrics.get('metrics/precision(B)', 'N/A'):.4f}")
        print(f"  Recall    : {metrics.get('metrics/recall(B)', 'N/A'):.4f}")
    print("=" * 65)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Fine-tuning de YOLOv8 para deteccion de placas vehiculares (ALPR dataset)'
    )
    parser.add_argument(
        '--data', type=str, default=DEFAULT_DATASET_DIR,
        help=f'Directorio raiz del dataset ALPR (default: {DEFAULT_DATASET_DIR})'
    )
    parser.add_argument(
        '--epochs', type=int, default=EPOCHS_DEFAULT,
        help=f'Numero de epocas de entrenamiento (default: {EPOCHS_DEFAULT})'
    )
    args = parser.parse_args()

    if not os.path.isdir(args.data):
        print(f"\n[ERROR] Dataset no encontrado en: {args.data}")
        print("\nEstructura esperada:")
        print(f"  {args.data}/")
        print("    train/")
        print("      images/  (archivos .jpg / .png)")
        print("      labels/  (archivos .txt en formato YOLO)")
        print("    valid/")
        print("      images/")
        print("      labels/")
        print("\nColoca el dataset ALPR en esa estructura y vuelve a ejecutar.")
        sys.exit(1)

    train(dataset_dir=args.data, epochs=args.epochs)
