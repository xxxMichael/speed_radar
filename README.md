# 🚔 Radar de Velocidad Inteligente con OCR y Lógica Difusa

Este proyecto es un **sistema de videovigilancia y control vial basado en Inteligencia Artificial** diseñado para estimar la velocidad de vehículos en tiempo real, detectar infracciones según los límites permitidos, capturar y reconocer las placas vehiculares mediante OCR, y evaluar la gravedad de la multa a través de un sistema de inferencia difusa.

---

## 📋 Tabla de Contenidos
1. [Características Principales](#-características-principales)
2. [Estructura del Proyecto](#-estructura-del-proyecto)
3. [Instalación y Configuración](#-instalación-y-configuración)
4. [🛠️ Script de Prueba: Reconocimiento de Placas (`plate_ocr_test.py`)](#️-script-de-prueba-reconocimiento-de-placas-plate_ocr_testpy)
5. [🚗 Script de Prueba: Radar de Infracciones (`detection_test.py`)](#-script-de-prueba-radar-de-infracciones-detection_testpy)
6. [🧠 Módulos Adicionales](#-módulos-adicionales)

---

## ✨ Características Principales

- **Detección y Tracking**: Utiliza **YOLOv8** y **ByteTrack** para rastrear múltiples vehículos simultáneamente sin perder su ID.
- **Medición de Velocidad**: Emplea líneas virtuales configurables (horizontales o verticales). Calcula la velocidad midiendo el tiempo de tránsito entre ambas líneas según una distancia real calibrada.
- **Procesamiento Concurrente**: El módulo de OCR de placas corre en un **hilo separado en segundo plano** para evitar caídas de FPS en el procesamiento de video en tiempo real.
- **OCR con Red Neuronal Convolucional (CNN)**: Pipeline propio que extrae la mitad inferior del vehículo (zona probable de la placa), segmenta caracteres con OpenCV heurístico y los clasifica usando una CNN en PyTorch entrenada con EMNIST.
- **Sistema de Multas Difuso (Fuzzy Logic)**: Implementa un controlador difuso Mamdani mediante `scikit-fuzzy` que calcula el monto de la multa ($0 - $500) cruzando el *exceso de velocidad* y el *límite de velocidad de la vía* (ej. es más grave exceder el límite en zona urbana que en carretera).

---

## 📁 Estructura del Proyecto

```text
speed_radar/
│
├── .venv/                     # Entorno virtual de Python
├── yolov8n.pt                 # Modelo preentrenado de YOLOv8
├── requirements.txt           # Dependencias del sistema
├── README.md                  # Documentación del proyecto (Este archivo)
│
└── src/
    ├── main.py                # Script principal de integración
    ├── char_cnn.pth           # Pesos entrenados del clasificador OCR
    │
    ├── camera/                # Captura y transmisión de video
    │   ├── video_stream.py    # Hilos para DroidCam (IP Webcam) y Webcams locales
    │   └── webcam_test.py     # Script para verificar transmisión e índices de cámara
    │
    ├── detection/             # Detección de vehículos y radar
    │   ├── vehicle_tracker.py # Tracking de vehículos (YOLOv8 + Bytetrack)
    │   ├── speed_calculator.py# Algoritmo de velocidad con líneas virtuales
    │   └── detection_test.py  # CALIBRACIÓN Y PRUEBA DE INFRACCIONES
    │
    ├── ocr/                   # Pipeline de lectura de placas
    │   ├── cnn_model.py       # Arquitectura CNN para clasificación de caracteres
    │   ├── plate_segmentation.py # Segmentación de letras/números usando OpenCV
    │   ├── plate_ocr.py       # Pipeline de extracción de placa y clasificación
    │   ├── plate_ocr_test.py  # PRUEBA EXCLUSIVA DE OCR DE PLACA
    │   └── train_ocr.py       # Entrenamiento de la CNN en EMNIST/Custom Datasets
    │
    └── fuzzy/                 # Inferencia difusa
        └── fine_system.py     # Sistema difuso Mamdani para cálculo de multas
```

---

## 🚀 Levantamiento del Proyecto con `venv`

Para levantar el proyecto y aislar sus dependencias de otras instalaciones de Python en tu sistema, sigue estos pasos:

### 1. Crear el Entorno Virtual (`venv`)
Abre tu consola de comandos (PowerShell o CMD) en la carpeta raíz del proyecto (`c:\Dev\7-semestre\AI\ProyectoFinal\speed_radar`) y ejecuta:
```powershell
python -m venv .venv
```
*Esto creará una carpeta llamada `.venv` en el directorio de tu proyecto con una instalación limpia de Python.*

### 2. Activar el Entorno Virtual
Debes activar el entorno virtual antes de instalar las dependencias o ejecutar los scripts.

* **En Windows (PowerShell):**
  > [!NOTE]
  > Si PowerShell te muestra un error de políticas de ejecución al activar el script, ejecuta primero:
  > `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process`
  ```powershell
  .\.venv\Scripts\Activate.ps1
  ```
* **En Windows (CMD):**
  ```cmd
  .\.venv\Scripts\activate.bat
  ```

Una vez activado, verás que el prefijo de tu terminal cambia para mostrar `(.venv)`.

### 3. Instalar Dependencias Generales
Con el entorno virtual activo, instala las dependencias base ejecutando:
```powershell
pip install -r requirements.txt
```

---

## ⚡ Configuración y Aceleración por GPU (CUDA)

Para que YOLOv8 y la Red Neuronal del OCR procesen a máxima velocidad en tiempo real, se recomienda utilizar una tarjeta gráfica NVIDIA compatible con CUDA.

### 1. Verificar compatibilidad en tu sistema
Ejecuta en tu terminal el siguiente comando para ver la versión de CUDA instalada en los drivers de tu GPU NVIDIA:
```powershell
nvidia-smi
```
Verás la versión de CUDA soportada (ej. `CUDA Version: 12.1`, `12.4`, etc.) en la esquina superior derecha del reporte generado.

### 2. Instalar PyTorch con Soporte CUDA
Por defecto, al instalar `requirements.txt` se descarga la versión de PyTorch para CPU. Para reemplazarla por la versión compatible con GPU CUDA, ejecuta el comando adecuado según tu versión de drivers:

* **Para CUDA 12.1 / 12.4 (Recomendado):**
  ```powershell
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 --force-reinstall
  ```
* **Para CUDA 11.8 (GPUs más antiguas):**
  ```powershell
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 --force-reinstall
  ```

El parámetro `--force-reinstall` asegura que se elimine la versión de CPU instalada anteriormente y se configure la de GPU.

### 3. Verificar que CUDA está Activo
Para confirmar que el entorno virtual reconoce correctamente tu tarjeta de video NVIDIA para aceleración por GPU, ejecuta:
```powershell
python -c "import torch; print('CUDA Disponible:', torch.cuda.is_available()); print('Dispositivo GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'Ninguno')"
```
Debe retornar:
```text
CUDA Disponible: True
Dispositivo GPU: NVIDIA GeForce ... (tu tarjeta gráfica)
```

Tanto YOLOv8 (`ultralytics`) como el pipeline OCR (`PlateOCR`) detectan automáticamente la disponibilidad de CUDA y la utilizarán de forma predeterminada sin necesidad de modificar el código fuente.

---

## 📧 Sistema de Notificaciones por Correo

El sistema cuenta con un servicio automático y asíncrono para enviar notificaciones de infracción de tránsito por correo electrónico. Cuando un vehículo supera el límite de velocidad:
1. El OCR lee la placa.
2. El **Sistema Difuso Mamdani** calcula el monto de la multa ($).
3. Se recorta un frame del vehículo infractor como evidencia visual.
4. Se despacha un correo HTML en segundo plano con el reporte de infracción adjuntando la evidencia.

### Configuración de Credenciales (`.env`)
Para activar el envío de correos, crea o edita el archivo `.env` en la raíz del proyecto con las siguientes variables:
```env
# Configuración del servidor de correo saliente (SMTP)
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587

# Cuenta emisora (Remitente)
SMTP_USER=tu_correo@gmail.com
SMTP_PASSWORD=tu_contrasena_de_aplicacion  # Contraseña de aplicación de Google

# Cuenta receptora (Destinatario)
EMAIL_RECIPIENT=correo_destinatario@gmail.com
```

> [!IMPORTANT]
> Si utilizas Gmail, debes habilitar la **Verificación en Dos Pasos** en tu cuenta de Google y generar una **Contraseña de Aplicación** (App Passwords) en la sección de Seguridad. No utilices la contraseña habitual de tu correo, ya que Google bloqueará la conexión directa.

---

## 💻 Ejecución de los Scripts del Proyecto

Una vez que tengas el entorno virtual listo y configurado con o sin CUDA, puedes lanzar los scripts de prueba utilizando el intérprete de Python del entorno virtual:

### Opción A: Con el entorno virtual activado
```powershell
# Probar el OCR de placas por cámara
python src/ocr/plate_ocr_test.py

# Probar el radar de velocidad e infracciones completo
python src/detection/detection_test.py
```

### Opción B: Ejecución directa sin activar previamente (Recomendado para scripts automatizados)
Puedes llamar directamente al ejecutable de Python dentro del entorno virtual desde la raíz del proyecto:
```powershell
# Ejecutar prueba de OCR de placas
.\.venv\Scripts\python.exe src/ocr/plate_ocr_test.py

# Ejecutar radar de velocidad completo
.\.venv\Scripts\python.exe src/detection/detection_test.py
```


---

## 🛠️ Script de Prueba: Reconocimiento de Placas (`plate_ocr_test.py`)

Este script sirve exclusivamente para probar la **cámara y el pipeline del OCR de placas** de forma aislada, sin activar YOLO ni medir velocidad. Es sumamente útil para calibrar la iluminación, la distancia de enfoque y la calidad de la segmentación de caracteres.

### Cómo ejecutarlo
Desde la terminal con el entorno virtual activo:

```powershell
python src/ocr/plate_ocr_test.py
```

### Modos de Operación e Interfaz
1. **Recuadro de Enfoque (ROI)**: Aparece un recuadro verde con esquinas marcadas en el centro de la pantalla. Coloca la placa dentro de este recuadro.
2. **Modo Manual**: Apunta a la placa y presiona la tecla `ESPACIO` para capturar y realizar una predicción inmediata. El resultado se mostrará en un banner inferior y en consola.
3. **Modo Continuo**: Presiona `C` para activar la detección automática cada 1 segundo.
4. **Ventana de Debug de Segmentación**: Al presionar `D`, se abrirá una ventana secundaria llamada `Debug - Segmentacion de Caracteres`. En ella podrás ver la placa binarizada y los contornos verdes que el algoritmo detecta como caracteres válidos. **Si los contornos coinciden correctamente con las letras y números, la predicción de la CNN será exitosa.**

### Controles de Teclado del Script de Placas

| Tecla | Acción |
| :---: | :--- |
| `ESPACIO` | Capturar y reconocer la placa actual dentro del recuadro |
| **`C`** | Activar / desactivar modo continuo (lectura cada segundo) |
| **`D`** | Mostrar / ocultar ventana de debug de segmentación de caracteres |
| **`+` / `-`** | Agrandar o achicar el recuadro verde de captura |
| **`S`** | Guardar una captura de pantalla del frame actual |
| **`Q` / `ESC`**| Salir del script y ver un resumen de todas las placas leídas |

---

## 🚗 Script de Prueba: Radar de Infracciones (`detection_test.py`)

Este es el script de **prueba integral**. Ejecuta el tracker YOLOv8 en paralelo y permite calibrar las líneas de radar interactivamente. Cuando un vehículo cruza ambas líneas, calcula su velocidad en tiempo real y, **en un hilo asíncrono para no trabar el video**, recorta la placa, ejecuta el OCR, determina el valor de la multa por medio del sistema de control de inferencia difuso y despacha el correo de alerta de manera simultánea.

### Cómo ejecutarlo
Desde la terminal con el entorno virtual activo:

```powershell
python src/detection/detection_test.py
```

### Funcionamiento y Calibración en Pantalla
1. **Fijar las Líneas de Medición**: Al abrir el video, realiza **doble clic izquierdo** en dos puntos de la pantalla para definir la línea de inicio y la línea de fin. Las velocidades solo se medirán si las líneas están fijadas.
2. **Orientación de las Líneas (`L`)**: Puedes cambiar el sentido de las líneas:
   - **Horizontal (Por defecto)**: Mide el tránsito vertical (vehículos bajando o subiendo).
   - **Vertical**: Mide el tránsito horizontal (vehículos moviéndose de izquierda a derecha o viceversa).
3. **Calibración de Distancia (`+` / `-`)**: Ajusta la distancia física real en metros entre ambas líneas virtuales directamente desde el teclado para que el cálculo en `km/h` sea preciso.
4. **Límite de Velocidad (`.` / `,`)**: Sube o baja el límite permitido para forzar alertas de infracción en el radar.
5. **Modo Diagnóstico (`D`)**: Muestra todas las detecciones de YOLO, no solo vehículos (muy útil para depurar si YOLO no detecta algún objeto).

### Salida de Datos en Consola (Consola de Infracciones)
Cuando el vehículo cruza la segunda línea y es analizado por el OCR y el sistema difuso, se imprimen los registros:

```bash
# Caso 1: Vehículo supera el límite de velocidad (Genera multa difusa y dispara correo electrónico)
[INFRACCION] 13:45:12 | Vehiculo #4 | Placa: ABC123 | Velocidad: 45.8 km/h | Multa Difusa: $185.34 USD
[SMTP] Iniciando envío de correo de alerta para Vehículo #4...
[SMTP] Correo enviado exitosamente para el vehiculo #4 (Placa: ABC123, Multa: $185.34)

# Caso 2: Vehículo transita a velocidad permitida (Solo registra de forma informativa, sin multa ni correo)
[REGISTRO]   13:45:18 | Vehiculo #5 | Placa: XYZ789
```

### Controles de Teclado del Radar de Infracciones

| Tecla / Acción | Acción |
| :---: | :--- |
| **Doble Clic** | Fija las posiciones de las líneas virtuales en la pantalla |
| **`L`** | Alterna el sentido de las líneas virtuales (Vertical u Horizontal) |
| **`+` / `-`** | Aumenta / disminuye la distancia física real en metros entre las líneas virtuales |
| **`.` / `,`** | Aumenta / disminuye el límite de velocidad de la vía |
| **`D`** | Activa / desactiva modo diagnóstico (muestra todas las de YOLO) |
| **`R`** | Resetea la calibración de las líneas y limpia el historial de velocidades |
| **`S`** | Guarda una captura de pantalla del frame actual |
| **`Q`** | Salir del script |

---

## 🧠 Módulos Adicionales

### 1. Clasificador CNN (`src/ocr/cnn_model.py`)
Es la red neuronal del OCR basada en PyTorch. Consta de múltiples capas de convolución, normalización y dropout para clasificar imágenes de caracteres de 28x28 en una de las 47 clases del set de datos EMNIST Balanced (letras mayúsculas, minúsculas difíciles unificadas y números).

### 2. Inferencia Difusa (`src/fuzzy/fine_system.py`)
Controlador difuso basado en lógica Mamdani. 
- **Entradas**: `exceso_velocidad` ($0 - 100 \text{ km/h}$) y `limite_via` ($30 - 120 \text{ km/h}$).
- **Reglas**:
  - Si el exceso de velocidad es *bajo* y la vía es de *carretera*, la severidad es *leve*.
  - Si el exceso de velocidad es *bajo* y la vía es *urbana*, la severidad es *moderada*.
  - Si el exceso de velocidad es *medio* y la vía es *urbana*, la severidad es *grave*.
  - Si el exceso de velocidad es *alto*, la severidad siempre es *grave*.
- **Salida**: Retorna el monto sugerido de la multa en dólares ($0 - $500).
