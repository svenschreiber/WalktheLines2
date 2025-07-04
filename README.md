# Walk the Lines 2

[[`Paper`]()] [[`Video`](https://fiona.uni-hamburg.de/00b1ea9d/kelm-walk-the-lines-2-icpr2021.mp4)] [[`BibTex`](#citing-walk-the-lines-2)]

![tracer_gif](assets/tracer.gif?raw=true)

This is the official Python code for Walk the Lines 2 (WtL2), a contour tracking algorithm for highly detailed segmentation of RGB objects. WtL2 forms precise, closed contours, even for complex shapes (e.g. infrared ships), which can be binarized into high-IoU masks. 

## Installation
To use the code provided in this repository, you first need to install the required packages. Follow the steps below to create a Conda environment with all necessary dependencies:
```bash
git clone https://github.com/svenschreiber/WalktheLines2
cd WalktheLines2

conda create -n wtl2 python=3.10
conda activate wtl2

pip install -e .
```

## Usage
This is a minimal example to use the WtL2 algorithm. A checkpoint file for the TracerNet model is included in this repository, and can be found in the [`checkpoints/`](checkpoints) directory.
```python
import cv2
from wtl2 import WtL2

# Load model checkpoint
checkpoint_path = "tracer_net.pth"
wtl = WtL2(checkpoint_path, device="cuda")

# Read RGB image and 16-bit soft contour map
img = cv2.imread("rgb.jpg")[...,::-1]
cont = cv2.imread("soft_contour_map.png", -1)

# Apply Walk the Lines 2
result = wtl.tracer_walk(img, cont)
cv2.imwrite("output.png", result)
```
For additional information about the usage, see the provided [`example.ipynb`](examples/example.ipynb) notebook.

### Binarization

## Citing Walk the Lines 2
If you use Walk the Lines 2 in your research, please use the following BibTeX entry.
```
Coming soon
```