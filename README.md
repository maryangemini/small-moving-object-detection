# Small Moving Object Detection from a Moving Camera

Real-time Computer Vision pipeline for detecting and tracking small moving objects in video captured by a moving camera.

The pipeline combines classical camera motion compensation with a lightweight neural network and object tracking.

## Pipeline

```text
Video
  ↓
Frame preprocessing
  ↓
Feature detection (GFTT)
  ↓
Pyramidal Lucas–Kanade optical flow
  ↓
Forward-backward validation
  ↓
RANSAC camera motion estimation
  ↓
Frame alignment
  ↓
Tiny U-Net
  ↓
Motion segmentation
  ↓
Bounding boxes
  ↓
SORT tracking
```

## Technologies

- Python
- OpenCV
- NumPy
- SciPy
- PyTorch
- CUDA
- Git

## Environment

Primary tested environment:

- Windows 11
- Python 3.12
- NVIDIA GeForce RTX 3050 Laptop GPU
- CUDA-enabled PyTorch

The project also contains optional Core ML support for Apple Silicon.

## Installation

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install the project dependencies:

```powershell
pip install -r requirements.txt
```

Install a CUDA-enabled PyTorch build separately for your NVIDIA/CUDA environment.

Verify CUDA:

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

## Run Inference

Run pipelined inference and render detections:

```powershell
python .\src\inference.py `
  --video .\data\test.mov `
  --checkpoint .\models\best_unet.pth `
  --pipelined `
  --output .\outputs\result.mp4
```

Predictions and tracking results can also be exported:

```powershell
python .\src\inference.py `
  --video .\data\test.mov `
  --checkpoint .\models\best_unet.pth `
  --pipelined `
  --output .\outputs\result.mp4 `
  --predictions-json .\outputs\predictions.json `
  --tracks-json .\outputs\tracks.json
```

## Benchmark

Example benchmark configuration:

- NVIDIA GeForce RTX 3050 Laptop GPU
- PyTorch CUDA backend
- Processing resolution: 640×360
- 300 input frames
- Pipelined CPU/GPU execution

Results:

| Stage | Mean time | Throughput |
|---|---:|---:|
| Classical motion compensation | 9.91 ms | 100.9 FPS |
| Neural network | 10.45 ms | 95.7 FPS |
| Neural network + postprocessing | 11.11 ms | 90.0 FPS |
| Estimated sequential pipeline | 21.02 ms | ~47.6 FPS |
| Pipelined pipeline | 11.11 ms | 90.0 FPS |

Pipelining improved throughput by approximately **1.89×** by overlapping CPU-based motion compensation with GPU inference.

These results are hardware-, video-, and configuration-dependent and should not be interpreted as universal performance numbers.

Run the benchmark with:

```powershell
python .\src\inference.py `
  --video .\data\test.mov `
  --checkpoint .\models\best_unet.pth `
  --benchmark-pipelined `
  --warmup-frames 5 `
  --max-frames 300
```

## Project Structure

```text
.
├── src/
│   ├── inference.py
│   └── presentation.py
├── requirements.txt
├── README.md
└── .gitignore
```

Model checkpoints, datasets, test videos, and generated outputs are not tracked in Git.

## Notes

The real-time inference path produces causal predictions.

`presentation.py` contains optional offline visualization and smoothing utilities intended for demonstrations. Offline smoothing may use information from future frames and therefore should not be used for real-time evaluation metrics.

Pipelined inference improves throughput by overlapping CPU and GPU work, but it adds approximately one frame of additional pipeline latency.

## Current Status

The project is being reorganized from an experimental Computer Vision prototype into a reproducible portfolio project.