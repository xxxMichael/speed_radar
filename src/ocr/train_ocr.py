import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from cnn_model import CharClassifierCNN


# ── Función de transpuesta a nivel de módulo (requerido en Windows por multiprocessing spawn) ──
def transpose_image(x: torch.Tensor) -> torch.Tensor:
    """
    Corrige la rotación y espejo originales de las imágenes EMNIST.
    EMNIST almacena las imágenes transpuestas; esta función las endereза.
    Debe estar definida a nivel de módulo (no como lambda) para que pickle de
    multiprocessing en Windows pueda serializarla.
    """
    return x.transpose(1, 2)


def train_model(
    epochs: int = 30,
    batch_size: int = 128,
    learning_rate: float = 0.001,
    save_path: str = 'char_cnn.pth'
) -> None:
    """
    Entrena el modelo CharClassifierCNN con configuración optimizada para >94%.

    Mejoras clave sobre la versión anterior:
    - Normalización exacta de EMNIST balanced (mean=0.1751, std=0.3277).
    - Data Augmentation avanzada: RandomPerspective + RandomErasing.
    - Optimizador AdamW con weight decay para regularización.
    - Scheduler CosineAnnealingLR para descenso suave y final preciso del LR.
    - Label Smoothing (0.1) en CrossEntropyLoss para generalización.
    - 30 épocas (cubriendo un ciclo coseno completo).
    - Guarda el mejor modelo (best checkpoint) según exactitud de validación.

    Args:
        epochs (int): Número total de épocas de entrenamiento.
        batch_size (int): Tamaño del lote.
        learning_rate (float): Tasa de aprendizaje inicial para AdamW.
        save_path (str): Ruta de destino para guardar los pesos del modelo.
    """

    # ── 1. Configuración del dispositivo ────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Dispositivo seleccionado para entrenamiento: {device}")

    # ── 2. Normalización exacta de EMNIST Balanced ──────────────────────────
    # Valores calculados sobre el conjunto completo de EMNIST balanced.
    # Usar la normalización correcta estabiliza los gradientes y mejora la convergencia.
    EMNIST_MEAN = (0.1751,)
    EMNIST_STD  = (0.3277,)

    # ── 3. Transformaciones ──────────────────────────────────────────────────
    # Entrenamiento: augmentation agresivo para simular condiciones reales de placa.
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(transpose_image),           # Corregir orientación EMNIST

        # Augmentation 1: deformación afín (rotación, traslación, escala, cizallamiento)
        transforms.RandomAffine(
            degrees=12,
            translate=(0.12, 0.12),
            scale=(0.85, 1.15),
            shear=8
        ),

        # Augmentation 2: perspectiva aleatoria (simula ángulo de cámara)
        transforms.RandomPerspective(distortion_scale=0.2, p=0.5),

        # Normalización con estadísticas reales del dataset
        transforms.Normalize(EMNIST_MEAN, EMNIST_STD),

        # Augmentation 3: borrado aleatorio de regiones (simula oclusiones en placa)
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.15), ratio=(0.3, 3.3)),
    ])

    # Validación: solo corrección de orientación y normalización (sin augmentation)
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(transpose_image),
        transforms.Normalize(EMNIST_MEAN, EMNIST_STD),
    ])

    # ── 4. Carga del Dataset EMNIST ─────────────────────────────────────────
    print("[INFO] Descargando/Cargando dataset EMNIST (split='balanced')...")
    train_dataset = datasets.EMNIST(
        root='./data', split='balanced', train=True,
        download=True, transform=train_transform
    )
    test_dataset = datasets.EMNIST(
        root='./data', split='balanced', train=False,
        download=True, transform=test_transform
    )

    # num_workers=2 en Windows (spawn). Ajusta a 0 si aparece error de pickling.
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    # ── 5. Modelo, Pérdida, Optimizador y Scheduler ─────────────────────────
    model     = CharClassifierCNN(num_classes=47).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # Suavizado de etiquetas

    # AdamW: Adam con weight decay desacoplado → mejor regularización
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    # CosineAnnealingLR: baja el LR suavemente siguiendo una curva cosenoidal
    # T_max = total de epocas -> un ciclo completo coseno durante todo el entrenamiento
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    # -- 6. Entrenamiento con guardado del mejor checkpoint ------------------
    best_val_accuracy = 0.0
    print(f"[INFO] Iniciando entrenamiento por {epochs} epocas...")
    print(f"[INFO] Arquitectura: 4 bloques CNN | FC: 1024->512->47 | Scheduler: CosineAnnealingLR")
    print("-" * 70)

    for epoch in range(epochs):
        # -- Fase de entrenamiento --
        model.train()
        running_loss = 0.0
        correct = 0
        total   = 0

        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)

            optimizer.zero_grad()
            outputs = model(images)
            loss    = criterion(outputs, labels)
            loss.backward()

            # Gradient Clipping: evita explosión de gradientes en redes profundas
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            # Estadísticas de entrenamiento
            running_loss += loss.item()
            _, predicted  = torch.max(outputs.data, 1)
            total         += labels.size(0)
            correct       += (predicted == labels).sum().item()

            if (i + 1) % 100 == 0:
                current_lr = scheduler.get_last_lr()[0]
                print(
                    f"Época [{epoch+1:>2}/{epochs}], "
                    f"Paso [{i+1:>4}/{len(train_loader)}], "
                    f"Pérdida: {loss.item():.4f}, "
                    f"Exactitud: {100 * correct / total:.2f}%, "
                    f"LR: {current_lr:.6f}"
                )

        # ── Fase de validación ──
        model.eval()
        val_correct = 0
        val_total   = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                outputs        = model(images)
                _, predicted   = torch.max(outputs.data, 1)
                val_total      += labels.size(0)
                val_correct    += (predicted == labels).sum().item()

        val_accuracy = 100 * val_correct / val_total
        print(f"[INFO] Fin Época {epoch+1:>2}. Exactitud en Validación: {val_accuracy:.2f}%", end="")

        # Guardar el mejor modelo encontrado durante todo el entrenamiento
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), save_path)
            print(f"  [BEST] Mejor modelo guardado ({best_val_accuracy:.2f}%)")
        else:
            print()

        # Actualizar el learning rate (CosineAnnealing)
        scheduler.step()

    print("-" * 70)
    print(f"[SUCCESS] Entrenamiento completado.")
    print(f"[SUCCESS] Mejor exactitud en validación: {best_val_accuracy:.2f}%")
    print(f"[SUCCESS] Modelo guardado en: {save_path}")


if __name__ == "__main__":
    train_model(epochs=30, batch_size=128, learning_rate=0.001)
