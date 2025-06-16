# AITknika
IA realizada por el centro de estudio Almi  ---   | SCRIPT | "pruebas_12_5_2025.py"  ---   | ANDROID | Realeses --> "Pose Landmarker.zip"

# Instalación de Drivers y Configuración de Entorno

Este documento proporciona una guía detallada para la instalación de drivers y la configuración del entorno necesario para trabajar con NVIDIA, CUDA, cuDNN, TensorRT, TensorFlow y MediaPipe.

## Instalación de Drivers NVIDIA

1. **Actualizar el sistema e instalar los drivers:**

    ```bash
    sudo -s
    sudo apt install nvidia-driver-535 -y
    reboot
    ```

2. **Verificar la instalación:**

    ```bash
    nvidia-smi
    ```

## Configuración de Python

1. **Instalar Python y crear un entorno virtual:**

    ```bash
    apt install python3.10-venv
    python3 -m venv ~/env
    source ~/env/bin/activate
    ```

## Instalación de CUDA

1. **Añadir el repositorio de CUDA:**

    ```bash
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
    sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
    wget https://developer.download.nvidia.com/compute/cuda/12.8.1/local_installers/cuda-repo-ubuntu2004-12-8-local_12.8.1-570.124.06-1_amd64.deb
    sudo dpkg -i cuda-repo-ubuntu2004-12-8-local_12.8.1-570.124.06-1_amd64.deb
    sudo cp /var/cuda-repo-ubuntu2004-12-8-local/cuda-*-keyring.gpg /usr/share/keyrings/
    sudo apt-get update
    ```

2. **Instalar CUDA Toolkit:**

    ```bash
    sudo apt-get -y install cuda-toolkit-12-8
    apt install nvidia-cuda-toolkit
    ```

## Instalación de cuDNN

1. **Añadir el repositorio de cuDNN:**

    ```bash
    wget https://developer.download.nvidia.com/compute/cudnn/9.8.0/local_installers/cudnn-local-repo-ubuntu2004-9.8.0_1.0-1_amd64.deb
    sudo dpkg -i cudnn-local-repo-ubuntu2004-9.8.0_1.0-1_amd64.deb
    sudo cp /var/cudnn-local-repo-ubuntu2004-9.8.0/cudnn-*-keyring.gpg /usr/share/keyrings/
    sudo apt-get update
    ```

2. **Instalar cuDNN:**

    ```bash
    sudo apt-get -y install cudnn
    ```

## Instalación de TensorRT

1. **Instalar TensorRT:**

    ```bash
    python3 -m pip install --upgrade pip
    python3 -m pip install wheel
    python3 -m pip install --upgrade tensorrt
    python3 -m pip install tensorrt-cu11 tensorrt-lean-cu11 tensorrt-dispatch-cu11
    python3 -m pip install --upgrade tensorrt-lean
    python3 -m pip install --upgrade tensorrt-dispatch
    ```

2. **Descargar e instalar TensorRT desde el sitio oficial:**

    ```bash
    wget https://developer.nvidia.com/tensorrt/download/10x
    sudo dpkg -i "archivo"
    ```

    Para más detalles, consulta la [documentación oficial de TensorRT](https://docs.nvidia.com/deeplearning/tensorrt/latest/installing-tensorrt/installing.html).

## Instalación de TensorFlow

1. **Instalar TensorFlow con soporte para CUDA:**

    ```bash
    pip install tensorflow[and-cuda]
    ```

## Instalación de MediaPipe

1. **Clonar el repositorio de MediaPipe:**

    ```bash
    apt install git
    git clone https://github.com/google-ai-edge/mediapipe.git
    ```

2. **Instalar las dependencias necesarias:**

    ```bash
    pip install absl-py attrs flatbuffers jax jaxlib matplotlib "numpy<2" opencv-contrib-python "protobuf>=4.25.3,<5" "sounddevice>=0.4.4" sentencepiece
    pip install mediapipe --extra-index-url https://google-coral.github.io/py-repo/
    pip install --upgrade setuptools
    pip install seqeval
    pip install mediapipe-model-maker
    ```

## Solución de Problemas con TensorFlow

Si hay algún tipo de error con TensorFlow, estos comandos pueden ayudar a solucionarlo:

```bash
pip uninstall tensorflow mediapipe-model-maker tensorflow-text tfx-bsl
pip install mediapipe-model-maker tensorflow-text tfx-bsl
pip install --upgrade mediapipe-model-maker
python -c "import tensorflow as tf; print('TensorFlow version:', tf.__version__)"
pip install tensorflow-gpu

