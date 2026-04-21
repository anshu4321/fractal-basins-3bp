"""3-body equations of motion (m=G=1). Shared by basin, Newton, HP verify."""
import numpy as np


def accelerations(r):
    """r: (3, 2) positions -> (3, 2) accelerations."""
    a = np.zeros((3, 2))
    for i in range(3):
        for j in range(3):
            if i == j: continue
            d = r[j] - r[i]
            a[i] += d / (d @ d) ** 1.5
    return a


def energy(r, v):
    T_kin = 0.5 * np.sum(v * v)
    V = 0.0
    for i in range(3):
        for j in range(i + 1, 3):
            V -= 1.0 / np.linalg.norm(r[i] - r[j])
    return T_kin + V


def angular_momentum_z(r, v):
    return sum(r[i, 0] * v[i, 1] - r[i, 1] * v[i, 0] for i in range(3))


def s3_permute(r, v, cycle="123"):
    """Apply S3 cyclic permutation. cycle='123' means (1->2, 2->3, 3->1)."""
    perm = {"123": [1, 2, 0], "132": [2, 0, 1]}[cycle]
    return r[perm], v[perm]
