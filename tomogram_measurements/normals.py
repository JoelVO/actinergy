import mrcfile
import numpy as np
import zarr
import torch
import os
from tqdm import tqdm
from utils.color_palette import get_color_palette
from scipy.ndimage import distance_transform_edt


# path = '/Users/joel/PycharmProjects/lustre/actinergy/segmentations'
# colors = get_color_palette()
# num_steps = 15
# norm_window = 3
# pixel_size = 10

skeletons = {}
# mask = mrcfile.open(f'{path}/{name}', permissive=True).data
# mask = np.array(mask, dtype=np.float32)
#
# cancer_mask = mrcfile.open(f'{path}/{name.replace("ctl","p815")}',
#                            permissive=True).data
# cancer_mask = np.array(cancer_mask,dtype=np.float32)
#
# tomo = mrcfile.open(f'{path.replace("segmentations","tomos")}/{name.split("_pm_ctl")[0]+"_Vol_px10.mrc"}',
#                     permissive=True).data
# tomo = np.array(tomo, dtype=np.float32)
#
# skeleton = zarr.open(os.path.join(path.replace("segmentations", "skeletons"),
#                                   name.replace('.mrc', '.zarr')))
# skeleton = np.array(skeleton, dtype=np.float32)
#
# normals_path = os.path.join(path.replace("segmentations","normals"),
#                            name.split('_pm_ctl')[0]+'.zarr')

def compute_normals(mask,skeleton,locations_p815):

    distance_to_bck = distance_transform_edt(mask)
    dx = dy = dz = 1.0
    dfdx, dfdy, dfdz = np.gradient(distance_to_bck, dx, dy, dz)

    # Second derivatives (and mixed)
    fxx, fxy, fxz = np.gradient(dfdx, dx, dy, dz)
    fyx, fyy, fyz = np.gradient(dfdy, dx, dy, dz)
    fzx, fzy, fzz = np.gradient(dfdz, dx, dy, dz)

    hessian = np.array([[fxx, fxy, fxz],
                        [fyx, fyy, fyz],
                        [fzx, fzy, fzz]])
    hessian = np.transpose(hessian, axes=(2, 3, 4, 0, 1))
    hessian = torch.tensor(hessian)

    sh = torch.zeros((mask.shape + (3,)))
    for h in tqdm(range(len(hessian))):
        layer = hessian[h].clone().flatten(0, 1)
        dl = torch.tensor(distance_to_bck[h]).flatten(0, 1)
        shl = []
        for l_, l in enumerate(layer):
            if dl[l_] == 0:
                shl.append(torch.zeros(3))
                continue

            eigenvalues, eigenvectors = torch.linalg.eig(l)
            # Convert eigenvalues to real if they are complex (common for non-symmetric matrices)
            eigenvalues = eigenvalues.real
            eigenvectors = eigenvectors.real
            # Find index of smallest eigenvalue
            max_index = torch.argmin(eigenvalues)

            # Get corresponding eigenvector
            shl.append(eigenvectors[:, max_index])
        sh[h] = torch.stack(shl).reshape(sh[h].shape)

    normals = sh.numpy()
    normals = np.multiply(normals, np.expand_dims(skeleton, -1))

    locations_skeleton = np.stack(np.where(skeleton > 0)).T
    locations_normals = []
    for loc in locations_skeleton:
        distances = np.linalg.norm(locations_p815 - np.expand_dims(loc, axis=0), axis=-1)
        distances_normal = np.linalg.norm(locations_p815 - np.expand_dims(loc + normals[loc[0], loc[1], loc[2]],
                                                                            axis=0), axis=-1)
        if np.amin(distances) < np.amin(distances_normal):
            normals[loc[0], loc[1], loc[2]] *= -1
        locations_normals.append(normals[loc[0], loc[1], loc[2]])

    return np.array(locations_normals)
