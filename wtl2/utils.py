import numpy as np
import scipy.ndimage as ndi
import torch
from skimage.color import label2rgb
import matplotlib.pyplot as plt

def in_notebook():
    try:
        from IPython import get_ipython
        return 'IPKernelApp' in get_ipython().config
    except:
        return False
if in_notebook(): from IPython import display

def checkerboard(height, width, block_size):
    num_blocks_x = height // block_size 
    num_blocks_y = width // block_size
    checker_pattern = (-1) ** (np.add.outer(np.arange(num_blocks_x), np.arange(num_blocks_y)))
    result = np.kron(checker_pattern, np.ones((block_size, block_size)))
    return result[:height, :width]

# Adapted from https://gist.github.com/bmabey/4dd36d9938b83742a88b6f68ac1901a6
def bwmorph_endpoints(image):
    image = image.astype(np.int32)
    k = np.array([[1,1,1],[1,0,1],[1,1,1]])
    neighborhood_count = ndi.convolve(image,k, mode='constant', cval=1)
    neighborhood_count[~image.astype(np.bool)] = 0
    return neighborhood_count == 1

def get_neighbors(arr, y, x):
    h, w = arr.shape
    x1, x2 = max(x-1, 0), min(x+2, w)
    y1, y2 = max(y-1, 0), min(y+2, h)
    return arr[y1:y2, x1:x2]

def get_pixel_idx_list(labeled_image):
    labels = np.unique(labeled_image)
    labels = labels[labels != 0]
    return [np.flatnonzero(labeled_image == label) for label in labels]

def tensor_cropper(image, cp, crop_wind):
    y, x = cp
    half = crop_wind // 2

    y1, y2 = y - half, y + half + (crop_wind % 2)
    x1, x2 = x - half, x + half + (crop_wind % 2)

    # Handle edge cases by clipping indices to valid ranges
    y1, y2 = max(0, y1), min(image.shape[0], y2)
    x1, x2 = max(0, x1), min(image.shape[1], x2)

    return image[y1:y2, x1:x2].astype(np.float32)

def vector_to_angle_deg(preds):
    cos_vals, sin_vals = preds[:, 0], preds[:, 1]
    angles_rad = torch.atan2(sin_vals, cos_vals)
    angles_deg = torch.rad2deg(angles_rad)
    return angles_deg

def compute_target_angles(p_step):
    w_size = 2 * p_step + 1

    # Create grid of coordinates relative to center
    y, x = np.mgrid[-p_step:p_step+1, -p_step:p_step+1]

    # Create mask with border set to 0
    center_mask = np.ones((w_size, w_size), dtype=bool)
    center_mask[0, :] = False
    center_mask[-1, :] = False
    center_mask[:, 0] = False
    center_mask[:, -1] = False

    # Compute angles, shift by pi, convert to degrees
    vectormatrix = x + 1j * y
    angle_deg = np.degrees(np.angle(vectormatrix) + np.pi)
    angle_deg = np.round(angle_deg)

    # Function to zero out specified angles in the inner region
    def zero_angles(angles, center_mask, target_angles):
        for t in target_angles:
            angles[(angles == t) & center_mask] = 0
        return angles

    # Target angles to zero
    targets = [0, 45, 90, 135, 180, 225, 270, 315, 360]

    # Remove angles where inner mask is True
    angle_deg = zero_angles(angle_deg, center_mask, targets)
    angle_deg_mirr = np.flipud(angle_deg) * -1

    # Get unique angles
    unique_angles = np.unique(angle_deg).astype(np.int32)
    unique_angles_mirr = np.unique(angle_deg_mirr).astype(np.int32)
    return vectormatrix, angle_deg, angle_deg_mirr, unique_angles, unique_angles_mirr


def visualize(fig, ax, labeled_image, num_tracers):
    notebook = in_notebook()
    colored_labels = label2rgb(labeled_image, bg_label=0, kind='overlay')

    if not notebook:
        plt.ion()

    ax.clear()
    ax.set_title(f"tracer: {num_tracers}")
    ax.imshow(colored_labels)

    if notebook:
        display.clear_output(wait=True)
        display.display(fig)
    else:
        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.show(block=False)