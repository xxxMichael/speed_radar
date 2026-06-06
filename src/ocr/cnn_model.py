import torch
import torch.nn as nn
import torch.nn.functional as F


class CharClassifierCNN(nn.Module):
    """
    Arquitectura CNN profunda para clasificación de caracteres individuales (28x28 px, escala de grises).
    Diseñada para alcanzar >94% de exactitud en EMNIST Balanced (47 clases).

    Mejoras respecto a la versión anterior:
    - 4 bloques convolucionales en lugar de 3 (mayor profundidad).
    - Más filtros por capa: 64 → 128 → 256 → 256.
    - Capas FC más grandes: 1024 → 512 → num_classes.
    - Dropout calibrado para regularización sin perder capacidad.
    """

    def __init__(self, num_classes: int = 47):
        """
        Inicializa las capas de la CNN mejorada.

        Args:
            num_classes (int): Número de clases de salida. EMNIST Balanced = 47.
        """
        super(CharClassifierCNN, self).__init__()

        # ── Bloque Convolucional 1 ──────────────────────────────────────────
        # Entrada: [B, 1, 28, 28]  →  Salida: [B, 64, 28, 28]
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(64)

        # ── Bloque Convolucional 2 ──────────────────────────────────────────
        # Entrada: [B, 64, 28, 28]  →  Salida después de Pool: [B, 128, 14, 14]
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(128)

        # ── Bloque Convolucional 3 ──────────────────────────────────────────
        # Entrada: [B, 128, 14, 14]  →  Salida después de Pool: [B, 256, 7, 7]
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3   = nn.BatchNorm2d(256)

        # ── Bloque Convolucional 4 ──────────────────────────────────────────
        # Entrada: [B, 256, 7, 7]  →  Salida: [B, 256, 7, 7]  (sin pool extra)
        self.conv4 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn4   = nn.BatchNorm2d(256)

        # ── Pooling y Regularización ────────────────────────────────────────
        self.pool      = nn.MaxPool2d(kernel_size=2, stride=2)
        self.drop_conv = nn.Dropout2d(0.15)   # Dropout convolucional suave
        self.drop_fc   = nn.Dropout(0.5)       # Dropout fuerte en capas densas

        # ── Capas Completamente Conectadas ──────────────────────────────────
        # Después de Conv1 → Conv2 → Pool(14x14) → Conv3 → Pool(7x7) → Conv4
        # Feature map final: 256 canales * 7 * 7 = 12544
        self.fc1 = nn.Linear(256 * 7 * 7, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Paso hacia adelante de la red.

        Args:
            x (torch.Tensor): Tensor de entrada [B, 1, 28, 28].

        Returns:
            torch.Tensor: Logits de salida [B, num_classes].
        """
        # Bloque 1: Conv → BN → ReLU
        x = F.relu(self.bn1(self.conv1(x)))

        # Bloque 2: Conv → BN → ReLU → Pool → Dropout
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)           # 28x28 → 14x14
        x = self.drop_conv(x)

        # Bloque 3: Conv → BN → ReLU → Pool → Dropout
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)           # 14x14 → 7x7
        x = self.drop_conv(x)

        # Bloque 4: Conv → BN → ReLU  (sin pool adicional)
        x = F.relu(self.bn4(self.conv4(x)))

        # Aplanar [B, 256, 7, 7] → [B, 12544]
        x = x.view(x.size(0), -1)

        # FC 1: 12544 → 1024
        x = F.relu(self.fc1(x))
        x = self.drop_fc(x)

        # FC 2: 1024 → 512
        x = F.relu(self.fc2(x))
        x = self.drop_fc(x)

        # FC 3 (cabeza de clasificación): 512 → num_classes
        x = self.fc3(x)
        return x


if __name__ == "__main__":
    print("[INFO] Validando arquitectura mejorada de CharClassifierCNN...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Dispositivo: {device}")

    model = CharClassifierCNN(num_classes=47).to(device)
    print(model)

    # Contar parámetros entrenables
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[INFO] Parámetros entrenables: {total_params:,}")

    dummy_input = torch.randn(4, 1, 28, 28).to(device)
    with torch.no_grad():
        output = model(dummy_input)

    print(f"[SUCCESS] Output shape (debe ser [4, 47]): {output.shape}")
