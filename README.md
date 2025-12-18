# Face Emotion Recognition (CNN)

A simple **face emotion recognition** project with:

- **Training** notebooks (TensorFlow/Keras and PyTorch)
- A **real-time webcam demo** using OpenCV + a PyTorch CNN (`realtimedetection.py`)

Dataset (Kaggle): https://www.kaggle.com/datasets/jonathanoheix/face-expression-recognition-dataset

## Features

- 7-class emotion classification: `angry`, `disgust`, `fear`, `happy`, `neutral`, `sad`, `surprise`
- Face detection via **OpenCV Haar Cascade**
- CNN inference with **PyTorch** (`emotiondetector.pth`)
- GPU acceleration for training/inference when CUDA is available

## Project Structure

- `realtimedetection.py` — webcam + face detection + emotion inference (PyTorch)
- `emotiondetector.pth` — trained PyTorch model weights
- `testgpu.py` — quick CUDA check for PyTorch
- `trainmodels.ipynb` — **PyTorch** training notebook
- `trainmodel.ipynb` — **Keras/TensorFlow** training notebook
- `requirements.txt` — common Python dependencies (note: PyTorch install is handled separately)
- `images/` — dataset folder (ignored by git)

## Dataset Setup

This repo expects the dataset extracted into the following structure:

```
images/
  train/
    angry/ disgust/ fear/ happy/ neutral/ sad/ surprise/
  test/
    angry/ disgust/ fear/ happy/ neutral/ sad/ surprise/
```

If your extracted Kaggle dataset ends up nested (for example `images/images/train`), move/copy the inner folder contents so that `images/train` and `images/test` exist.

### Download from Kaggle (optional)

1. Install the Kaggle CLI:

```bash
pip install kaggle
```

2. Configure credentials (choose one):

- Put `kaggle.json` in `%USERPROFILE%\.kaggle\kaggle.json`
- Or set env vars: `KAGGLE_USERNAME` and `KAGGLE_KEY`

3. Download + unzip:

```bash
kaggle datasets download -d jonathanoheix/face-expression-recognition-dataset -p ./images --unzip
```

## Installation

Create and activate a virtual environment (recommended), then install deps.

### Windows (PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### PyTorch (required for `realtimedetection.py`)

PyTorch is not listed in `requirements.txt` because the correct wheel depends on **CPU vs CUDA**.

- **CPU-only**:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

- **NVIDIA GPU (CUDA)**: pick the CUDA build that matches your system (example shown for CUDA 12.1):

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

Also install Pillow (used for image preprocessing) if you don’t already have it:

```bash
pip install pillow
```

## GPU Training / CUDA Setup (Required Things)

This project can train on GPU in **two ways**, depending on the notebook you use:

- **PyTorch GPU**: `trainmodels.ipynb`
- **TensorFlow/Keras GPU**: `trainmodel.ipynb`

### A) PyTorch GPU (recommended here, matches the real-time demo)

1. **Install NVIDIA driver** (Windows) and reboot.
2. Install **CUDA-enabled PyTorch** (see commands above).
3. Verify PyTorch sees your GPU:

```bash
python testgpu.py
```

You should see:
- `CUDA available: True`
- `GPU name: ...`

4. Run training:
- Open `trainmodels.ipynb` in Jupyter/VS Code and run all cells.
- The notebook uses:

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EmotionCNN().to(device)
```

So it will automatically use GPU if available.

**Common PyTorch GPU gotchas**
- If `torch.cuda.is_available()` is `False`, you likely installed the CPU wheel.
- If you have multiple GPUs, you can select one with:
  - Windows PowerShell: `setx CUDA_VISIBLE_DEVICES 0`
  - Or inside Python via `torch.cuda.set_device(0)`

### B) TensorFlow/Keras GPU (Windows notes)

TensorFlow GPU support depends on your OS/version.

- **Windows native GPU**: historically, the last official native Windows GPU build was TensorFlow 2.10.
- **Most reliable option on Windows** (2025): use **WSL2** (Ubuntu) for TensorFlow GPU.

If you want TensorFlow GPU on Windows:
1. Prefer **WSL2 + NVIDIA CUDA for WSL**.
2. Inside WSL, install TensorFlow and verify:

```python
import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))
```

If you don’t need TensorFlow specifically, use the PyTorch notebook instead.

## Train the Model

### PyTorch
- Notebook: `trainmodels.ipynb`
- Output weights: `emotiondetector.pth`

### Keras/TensorFlow
- Notebook: `trainmodel.ipynb`
- Note: this notebook is a separate training pipeline from the PyTorch demo.

## Downloads (Google Drive)

If you prefer downloading the trained model and watching the demo without running training locally:

- **Trained PyTorch weights (`emotiondetector.pth`)**: [link](https://drive.google.com/file/d/1uFBWMcoYQR8ASM16NAC1YzQv2K-PRmxj/view?usp=drive_link) 
- **Demo video**: <PASTE_GOOGLE_DRIVE_LINK_HERE>
- **Dataset**: [link](https://drive.google.com/file/d/1ABVPPBUzR_RwG09kGCjhY8t5DrE4RDSD/view?usp=sharing)

## Real-time Webcam Demo

Make sure `emotiondetector.pth` exists in the project root, then:

```bash
python realtimedetection.py
```

Controls:
- Press `q` to quit.
