import mrcfile
import numpy as np
import zarr
import matplotlib.pyplot as plt
from glob import glob
import torch
import os

from skimage.metrics import normalized_root_mse
from sklearn.decomposition import PCA
from tqdm import tqdm
from sklearn.mixture import GaussianMixture
from utils.functions import average_gmm
from utils.cropping_membrane import crop_membrane
from utils.color_palette import get_color_palette
from scipy.ndimage import distance_transform_edt
from scipy.interpolate import Rbf,NearestNDInterpolator,LinearNDInterpolator


path = '/Users/joel/PycharmProjects/lustre/actinergy/segmentations'
colors = get_color_palette()
num_steps = 15
norm_window = 3
# pixel_size = 10

skeletons = {}
for _,name in enumerate(os.listdir(path)):
    if 'ctl' not in name:
        continue
    print(name)
    mask = mrcfile.open(f'{path}/{name}', permissive=True).data
    mask = np.array(mask, dtype=np.float32)

    cancer_mask = mrcfile.open(f'{path}/{name.replace("ctl","p815")}',
                               permissive=True).data
    cancer_mask = np.array(cancer_mask,dtype=np.float32)

    tomo = mrcfile.open(f'{path.replace("segmentations","tomos")}/{name.split("_pm_ctl")[0]+"_Vol_px10.mrc"}',
                        permissive=True).data
    tomo = np.array(tomo, dtype=np.float32)

    skeleton = zarr.open(os.path.join(path.replace("segmentations", "skeletons"),
                                      name.replace('.mrc', '.zarr')))
    skeleton = np.array(skeleton, dtype=np.float32)

    normals_path = os.path.join(path.replace("segmentations","normals"),
                               name.split('_pm_ctl')[0]+'.zarr')
    if (not os.path.exists(normals_path)):

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
        locations_cancer = np.stack(np.where(cancer_mask > 0)).T

        projection = np.zeros_like(skeleton)
        locations_skeleton = np.stack(np.where(skeleton > 0)).T
        for loc in tqdm(locations_skeleton):
            distances = np.linalg.norm(locations_cancer - np.expand_dims(loc, axis=0), axis=-1)
            distances_normal = np.linalg.norm(locations_cancer - np.expand_dims(loc + normals[loc[0], loc[1], loc[2]],
                                                                                axis=0), axis=-1)
            if np.amin(distances) < np.amin(distances_normal):
                normals[loc[0], loc[1], loc[2]] *= -1

        zarr.save(normals_path, normals)

    else:
        normals = zarr.open(os.path.join(path.replace("segmentations","normals"),
                           name.split('_pm_ctl')[0]+'.zarr'))

    projection = np.zeros_like(skeleton)
    locations_skeleton = np.stack(np.where(skeleton > 0)).T

    pca = PCA(n_components=2)
    membrane_projection = pca.fit_transform(locations_skeleton)

    projected_values_path = os.path.join(path.replace("segmentations", "projected_clouds"),
                             name.split('_pm_ctl')[0] + '.npy')

    print_cloud = np.zeros_like(tomo)
    locations_cancer = np.stack(np.where(cancer_mask > 0)).T

    # if (not os.path.exists(projected_values_path)) or (1>0):
    if (not os.path.exists(projected_values_path)):
        projected_values = []
        for loc in tqdm(locations_skeleton):
            # if loc[0] not in [50,100,150]:
            #     projected_values.append(np.nan)
            #     continue
            # normal = np.expand_dims(normals[loc[0], loc[1], loc[2]],axis=0)
            normal = normals[max(0,loc[0]-norm_window):min(tomo.shape[0],loc[0]+norm_window),
                             max(0,loc[1]-norm_window):min(tomo.shape[1],loc[1]+norm_window),
                             max(0,loc[2]-norm_window):min(tomo.shape[2],loc[2]+norm_window)]
            normal = torch.tensor(normal).flatten(0,2).mean(dim=0,keepdim=True).numpy()
            normal /= (np.linalg.norm(normal)+1e-3)

            distance_to_cancer = int(np.amin(np.linalg.norm(locations_cancer - np.expand_dims(loc, axis=0), axis=-1)))
            steps = np.expand_dims(np.arange(0,min(distance_to_cancer,num_steps)),axis=-1)
            # steps = np.expand_dims(np.arange(0, num_steps), axis=-1)

            cloud = np.array(np.multiply(normal, steps),dtype=np.int32)
            cloud += np.expand_dims(loc,axis=0)

            cloud[:, 0] = np.clip(cloud[:, 0], 0, tomo.shape[0] - 1)
            cloud[:, 1] = np.clip(cloud[:, 1], 0, tomo.shape[1] - 1)
            cloud[:, 2] = np.clip(cloud[:, 2], 0, tomo.shape[2] - 1)
            cloud = np.unique(cloud,axis=0)

            # print(np.mean(tomo[cloud[:,0],cloud[:,1],cloud[:,2]]),tomo[loc[0],loc[1],loc[2]])
            projected_values.append(np.mean(tomo[cloud[:,0],cloud[:,1],cloud[:,2]])/(tomo[loc[0],loc[1],loc[2]]+1e-3))
            # projected_values.append(
            #     np.mean(tomo[cloud[:, 0], cloud[:, 1], cloud[:, 2]]) - (tomo[loc[0], loc[1], loc[2]]))
            # projected_values.append(np.mean(tomo[cloud[:, 0], cloud[:, 1], cloud[:, 2]]))
            projection[loc[0], loc[1], loc[2]] = projected_values[-1]

            print_cloud[cloud[:,0],cloud[:,1],cloud[:,2]] = 1

            # print(np.array(projected_values).shape)

        zarr.save(os.path.join(path.replace("segmentations","clouds"),
                               name.split('_pm_ctl')[0]+'.zarr'),projection)

        projected_values = np.expand_dims(np.array(projected_values), axis=-1)
        projected_values = np.clip(projected_values,
                                   np.quantile(projected_values,0.01),
                                   np.quantile(projected_values,0.99))
        projected_values = np.concatenate((membrane_projection, projected_values), axis=-1)
        np.save(projected_values_path, projected_values)
    else:
        projected_values = np.load(projected_values_path)

    # fig,axs = plt.subplots(1,3)
    #
    # axs[0].imshow(tomo[50],cmap='gray')
    # axs[0].imshow(print_cloud[50],alpha=0.5)
    # axs[0].imshow(normals[50], alpha=0.5)
    #
    # axs[1].imshow(tomo[100], cmap='gray')
    # axs[1].imshow(print_cloud[100],alpha=0.5)
    # axs[1].imshow(normals[100], alpha=0.5)
    #
    # axs[2].imshow(tomo[150], cmap='gray')
    # axs[2].imshow(print_cloud[150],alpha=0.5)
    # axs[2].imshow(normals[150], alpha=0.5)
    # plt.show()

    colors = get_color_palette()
    means_path = path.replace('segmentations', 'params') + '/gmm_means.npy'
    weights_path = path.replace('segmentations', 'params') + '/gmm_weights.npy'

    means = [np.load(means_path)]
    weights = [np.load(weights_path)]

    membrane_distances = torch.load(os.path.join(path.replace("segmentations", "distances"),
                                                                          name.split('_pm_ctl')[0] + '_pm.pt'))
    gm = GaussianMixture(n_components=5, covariance_type='tied',
                     means_init=np.mean(means, axis=0),
                     weights_init=np.mean(weights, axis=0)).fit(
    membrane_distances['ctl_to_p815']['distance'].unsqueeze(-1))

    classes = gm.predict(membrane_distances['ctl_to_p815']['distance'].unsqueeze(-1))
    projected_classes = pca.transform(membrane_distances['ctl_to_p815']['position'])

    nx, ny = tomo.shape[1],tomo.shape[0]  # grid resolution
    x,y,z = projected_values.T

    xi = np.linspace(x.min(), x.max(), nx)
    yi = np.linspace(y.min(), y.max(), ny)
    X, Y = np.meshgrid(xi, yi)

    interp = LinearNDInterpolator(list(zip(x, y)), z)
    Z = interp(X, Y)


    fig,axs = plt.subplots(ncols=2)
    axs[0].pcolormesh(X, Y, Z, shading='auto', cmap='gray')
    axs[0].scatter(x, y, c=z, s=1, cmap='gray')

    axs[1].pcolormesh(X, Y, Z, shading='auto', cmap='gray')
    axs[1].scatter(x, y, c=z, s=1, cmap='gray')
    axs[1].scatter(projected_classes[:,0],projected_classes[:,1],c=np.array(colors['regions'])[classes],
                alpha=0.1,s=1)

    axs[0].set_box_aspect(1 / 3)
    axs[1].set_box_aspect(1 / 3)
    plt.show()


    # break


