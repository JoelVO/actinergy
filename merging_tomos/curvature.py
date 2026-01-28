import numpy as np
import zarr
import os
import pathlib
from glob import glob

import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.decomposition import PCA

def fit_sphere_least_squares(points):
    """
    Fit a sphere to a 3D point cloud using least squares.

    Parameters
    ----------
    points : (N, 3) numpy array
        Input point cloud (x, y, z)

    Returns
    -------
    center : (3,) numpy array
        Sphere center (a, b, c)
    radius : float
        Sphere radius
    """

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

def compute_curvature(tomo_path,window=10,recompute=False):

    name = tomo_path.split('/')[-1].split('_Vol')[0]
    skeleton_path = str(pathlib.Path(tomo_path).parent).replace('tomos', 'skeletons')
    curvature_path = str(pathlib.Path(tomo_path).parent).replace('tomos', 'curvature')
    normals_path = str(pathlib.Path(tomo_path).parent).replace('tomos', 'normals')

    skeleton = zarr.load(os.path.join(skeleton_path, name+'_pm_ctl.zarr'))
    skeleton_position = np.stack(np.where(skeleton==1)).T
    normals = zarr.load(os.path.join(normals_path, name+'.zarr'))

    save_curvature_path = os.path.join(curvature_path, name+'.npy')
    if (not os.path.exists(save_curvature_path)) or recompute:
        x = {'positions':[],
                 'curvature':[]}
        for p in tqdm(skeleton_position):
            distances = np.linalg.norm(np.expand_dims(p,0)-skeleton_position,axis=-1)
            cloud = skeleton_position[distances<window]
            center, radius = fit_sphere_least_squares(cloud)
            dot_product = np.dot(center, normals[p[0],p[1],p[2]])

            x['positions'].append(p)
            x['curvature'].append(np.sign(dot_product)/radius)

        x['curvature'] = np.array(x['curvature'])
        x['positions'] = np.array(x['positions'])

        np.save(save_curvature_path,x)
    else:
        x = np.load(os.path.join(curvature_path, name+'.npy'),allow_pickle=True).item()

    return x

#########################
#########################
#########################
# Example on how to use the curvature computing
#########################
#########################
#########################

# fig, axs = plt.subplots(ncols=5, nrows=2, figsize=(15, 6))
#
# tomo_paths = glob('/Users/joel/PycharmProjects/lustre/actinergy/tomos/*.mrc')
#
# all_curvatures = []
# data = []
#
# for tomo_path in tomo_paths:
#     x = compute_curvature(tomo_path)
#     data.append(x)
#     all_curvatures.append(x['curvature'])
#
# vmin = np.min(np.concatenate(all_curvatures))
# vmax = np.max(np.concatenate(all_curvatures))
#
# sc = None
# for i, (tomo_path, x) in enumerate(zip(tomo_paths, data)):
#     pca = PCA(n_components=2)
#     membrane_projection = pca.fit_transform(x['positions'])
#
#     ax = axs[i // 5, i % 5]
#     sc = ax.scatter(
#         membrane_projection[:, 0],
#         membrane_projection[:, 1],
#         c=x['curvature'],
#         cmap='inferno',
#         s=1,
#         vmin=vmin,
#         vmax=vmax
#     )
#
#     ax.set_title(tomo_path.split('/')[-1], fontsize=8)
#     ax.set_adjustable('box')
#     ax.set_aspect(1)
#     ax.axis('off')
#
# cbar_ax = fig.add_axes([0.15, 0.03, 0.7, 0.02])
#
# cbar = fig.colorbar(
#     sc,
#     cax=cbar_ax,
#     orientation='horizontal'
# )
# cbar.set_label('Curvature')
#
# plt.tight_layout()
# plt.show()