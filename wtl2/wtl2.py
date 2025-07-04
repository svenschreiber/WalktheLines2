import numpy as np
import torch
from .tracer_net import TracerNet
from .nms import nms
from .utils import checkerboard, bwmorph_endpoints, get_neighbors, get_pixel_idx_list, tensor_cropper, vector_to_angle_deg, compute_target_angles, visualize
from skimage.morphology import skeletonize, remove_small_objects
from skimage.measure import label, regionprops
import scipy.ndimage as ndi
from scipy.interpolate import interp1d
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2


class WtL2:
    def __init__(self, checkpoint_path, device):
        self.model = TracerNet()
        self.device = device
        checkpoint = torch.load(checkpoint_path, weights_only=True, map_location=device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(device)
        self.model.eval()

        self.vectormatrix, self.angle_deg, self.angle_deg_mirr, self.unique_angles, self.unique_angles_mirr = compute_target_angles(p_step=3)
    
    def tracer_walk(self, img, cont, contour_uint16=True, visualization=False, verbose=True):
        img = img.astype(np.float32) * 256
        cont = cont.astype(np.float32)
        if not contour_uint16: cont *= 256

        pad_width = [(50, 50), (50, 50)]
        img = np.pad(img, pad_width + [(0, 0)], mode='constant', constant_values=0)
        cont = np.pad(cont, pad_width, mode='constant', constant_values=0)
        accumulator = np.zeros_like(cont)

        if visualization: fig, ax = plt.subplots()

        for flipped in [True, False]:
            img = np.flip(img, axis=1)
            cont = np.flip(cont, axis=1)
            accumulator = np.flip(accumulator, axis=1)

            thresh = np.max(cont) * 0.5
            cont_nms = nms(cont, r=3, s=2)
            cont_nms[cont_nms < thresh] = 0
            
            cb_mask = checkerboard(*cont.shape, block_size=2)
            if flipped: cont_nms[cb_mask > 0] = 0
            else:       cont_nms[cb_mask < 0] = 0
            cont_nms[cont_nms > 0] = 1

            cont_nms = skeletonize(cont_nms.astype(bool))
            cont_nms = remove_small_objects(cont_nms, min_size=2, connectivity=8)

            labeled_image, segments = label(cont_nms, return_num=True)

            endpoints = np.zeros((2, segments - 1, 2), dtype=np.int32)
            endpoints_ang = np.zeros((2, segments - 1))
            for i in range(0, segments - 1):
                segment = labeled_image == i + 1
                segment = ndi.binary_fill_holes(segment)
                endpoints_mask = bwmorph_endpoints(segment)

                coords = np.array(np.nonzero(endpoints_mask)).T
                if len(coords) == 1: coords = np.repeat(coords, 2, axis=0)
                if len(coords) != 2:
                    d = np.zeros(coords.size)
                    props = regionprops(segment.astype(np.int32))
                    x2, y2 = props[0].centroid
                    for j, (y1, x1) in enumerate(coords):
                        d[j] = np.hypot(x2 - x1, y2 - y1)
                    endpoints[:, i] = coords[np.argsort(d)[-2:][::-1]]
                else: endpoints[:, i] = coords

                for ep_idx in range(2):
                    neighbors = get_neighbors(segment, *coords[ep_idx])
                    neighbors[1, 1] = 0
                    dy, dx = np.unravel_index(np.argmax(neighbors), neighbors.shape)
                    angle_rad = np.atan2(dy - 1, dx - 1)
                    endpoints_ang[ep_idx, i] = np.degrees(angle_rad) % 360

            center_points = endpoints.reshape(-1, 2)
            Y, X = center_points.T
            prio = cont[Y, X]
            angles = endpoints_ang.flatten()
            group = labeled_image[Y, X]
            running = np.zeros(Y.shape, dtype=np.int32)
            pix_list = get_pixel_idx_list(labeled_image)

            interp_unique = interp1d(self.unique_angles, self.unique_angles, kind="nearest", bounds_error=False)
            interp_unique_mirr = interp1d(self.unique_angles_mirr, self.unique_angles_mirr, kind="nearest", bounds_error=False)

            inputs = np.concatenate([img, cont[...,None]], axis=2).astype(np.float32)
            crop_size = 13

            running[0] = 1
            step = 0
            temp = np.zeros(labeled_image.shape, dtype=np.uint8)
            num_tracers = running.sum()
            with tqdm(total=num_tracers, desc=f"Running Tracers [flipped={flipped}]", disable=not verbose) as pbar:
                while num_tracers > 0:

                    mask = running == 1
                    X = X[mask]
                    Y = Y[mask]
                    prio = prio[mask]
                    angles = angles[mask]
                    group = group[mask]
                    running = running[mask]

                    center_points = np.column_stack([Y, X])

                    crops = []
                    for y, x, angle in zip(Y, X, angles):
                        crop = tensor_cropper(inputs, [y, x], crop_size * 2 + 1)
                        rot_crop = ndi.rotate(crop, angle, reshape=False, order=1)
                        final_crop = tensor_cropper(rot_crop, [crop_size+1, crop_size+1], crop_size)
                        crops.append(final_crop)
                    with torch.no_grad():
                        tracer_input = torch.tensor(np.array(crops).transpose(0, 3, 2, 1)).to(self.device)
                        preds = self.model(tracer_input)
                        delta_angles = vector_to_angle_deg(preds).cpu().numpy()
                    angles = (angles + delta_angles) % 360

                    results = np.where(
                        angles >= 0,
                        interp_unique(angles),
                        interp_unique_mirr(angles)
                    )

                    complex_dir = np.array([
                        self.vectormatrix[self.angle_deg == res][0] if res > 0 else self.vectormatrix[self.angle_deg_mirr == res][0]
                        for res in results
                    ])
                    magnitudes = np.abs(complex_dir)
                    step_size = np.abs(np.random.randn(num_tracers, 1))
                    step_size[step_size < 1] = 1
                    step_size[step_size >= 2.5] = 3

                    dir = np.column_stack((np.imag(complex_dir), np.real(complex_dir))) / magnitudes[:, None]
                    dir = np.round(dir * step_size)
                    angles = np.degrees(np.atan2(*-dir.T))

                    new_center_points = (center_points + dir).astype(np.int32)
                    Y, X = new_center_points.T
                    labeled_image[Y, X] = group
                    accumulator[Y, X] += 0.05
                    temp[Y, X] = 255

                    for i, (y, x, g) in enumerate(zip(Y, X, group)):
                        g = int(g)
                        prio[i] = cont[y, x]
                        i_hel = np.ravel_multi_index([y, x], dims=segment.shape)
                        if i_hel in pix_list[g] or prio[i] <= 15000:
                            running[i] = 0
                        else:
                            pix_list[g] = np.append(pix_list[g], i_hel)

                    old_num_tracers = num_tracers
                    num_tracers = running.sum()
                    pbar.update(old_num_tracers - num_tracers)
                    if visualization: visualize(fig, ax, labeled_image, num_tracers)
                    if not flipped: 
                        cv2.imwrite(f"/home/sven/work/WalktheLines2/dev/viz/tracer/{step:03d}.png", temp)
                        cv2.imwrite(f"/home/sven/work/WalktheLines2/dev/viz/rgb.png", (img / 256).astype(np.uint8)[...,::-1])
                    step+=1

        if visualization: plt.close("all")
        result = (np.clip(accumulator, 0, 1) * 255).astype(np.uint8)
        return img, temp