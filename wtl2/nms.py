import numpy as np
from scipy.signal import convolve2d, convolve

# ported from https://github.com/tsogkas/matlab-utils/blob/master/nms.m
def contour_orient(E, r):
    if r <= 1:
        p = 12 / r / (r + 2) - 2
        f = np.array([1, p, 1]) / (2 + p)
        r = 1
    else:
        f = np.concatenate([np.arange(1, r+1), [r+1], np.arange(r, 0, -1)]) / (r + 1)**2
    E2 = np.pad(E, [r, r], mode='symmetric')
    E2 = convolve(convolve(E2, f[np.newaxis], mode='valid'), f[:, np.newaxis], mode='valid')
    grad_f = np.array([-1, 2, -1])
    Dx = convolve2d(E2, grad_f[np.newaxis], mode='same')
    Dy = convolve2d(E2, grad_f[:, np.newaxis], mode='same')
    f = np.array([[1, 0, -1], [0, 0, 0], [-1, 0, 1]])
    F = convolve2d(E2, f, mode='same') > 0
    Dy[F] = -Dy[F]
    return np.mod(np.arctan2(Dy, Dx), np.pi)

def nms(E, r=3, s=1):
    E = E.copy()
    O = contour_orient(E, r)
    
    Dx = np.cos(O)
    Dy = np.sin(O)
    
    ht, wd = E.shape
    E1 = np.pad(E, r+1, mode='edge')
    
    cs, rs = np.meshgrid(np.arange(wd), np.arange(ht))

    for i in range(-r, r+1):
        if i == 0: continue
        cs0 = i * Dx + cs
        dcs = cs0 - np.floor(cs0)
        cs0 = np.floor(cs0).astype(int)
        
        rs0 = i * Dy + rs
        drs = rs0 - np.floor(rs0)
        rs0 = np.floor(rs0).astype(int)
        
        rs0_p = rs0 + r + 1
        cs0_p = cs0 + r + 1

        rs0_p = np.clip(rs0_p, 0, ht + 2*r)
        cs0_p = np.clip(cs0_p, 0, wd + 2*r)
        
        E2 = (1 - dcs) * (1 - drs) * E1[rs0_p + 0, cs0_p + 0]
        E2 += dcs * (1 - drs) * E1[rs0_p + 0, cs0_p + 1]
        E2 += (1 - dcs) * drs * E1[rs0_p + 1, cs0_p + 0]
        E2 += dcs * drs * E1[rs0_p + 1, cs0_p + 1]

        E[E * 1.01 < E2] = 0

    for i in range(s):
        scale = (i) / s
        E[i, :] *= scale
        E[-i-1, :] *= scale
        E[:, i] *= scale
        E[:, -i-1] *= scale

    return E 