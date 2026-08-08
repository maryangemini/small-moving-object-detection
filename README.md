\# Small Moving Object Detection from a Moving Camera



Computer Vision pipeline for detecting small moving objects in video captured by a moving camera.



\## Pipeline



Video

→ Frame preprocessing

→ Feature detection

→ Pyramidal Lucas–Kanade optical flow

→ Forward-backward validation

→ RANSAC camera motion estimation

→ Frame alignment

→ Tiny U-Net

→ Motion segmentation

→ Bounding boxes

→ SORT tracking



\## Technologies



\- Python

\- OpenCV

\- NumPy

\- SciPy

\- PyTorch

\- CUDA



\## Environment



Primary development environment:



\- Windows

\- NVIDIA GPU / CUDA

\- PyTorch



\## Project Structure



```text

src/

&#x20;   inference.py

&#x20;   presentation.py



assets/

notebooks/

outputs/

requirements.txt

