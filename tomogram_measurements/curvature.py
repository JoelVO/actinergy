import numpy as np
from tqdm import tqdm

def fit_sphere_least_squares(points):
    # Ensure numpy array
    points = np.asarray(points)
    assert points.shape[1] == 3, "Input must be Nx3 array"

    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    # Construct linear system
    A = np.column_stack([2*x, 2*y, 2*z, np.ones(len(points))])
    b = x**2 + y**2 + z**2

    # Solve least squares
    C, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)

    a, b_, c, d = C
    center = np.array([a, b_, c])
    radius = np.sqrt(a**2 + b_**2 + c**2 + d)

    return center, radius

def compute_curvature(skeleton_position,normals,window=50):
    x = []
    for p in tqdm(skeleton_position):
        distances = np.linalg.norm(np.expand_dims(p,0)-skeleton_position,axis=-1)
        cloud = skeleton_position[distances<window]
        center, radius = fit_sphere_least_squares(cloud)
        dot_product = np.dot(center-p, normals[p[0],p[1],p[2]])
        x.append(np.sign(dot_product)/radius)

    return np.array(x)
