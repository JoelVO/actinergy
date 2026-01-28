import mrcfile
import numpy as np
import zarr
import os
import pathlib
from glob import glob
from pathlib import Path
from scipy.ndimage import distance_transform_edt
from scipy.ndimage import label
from scipy.spatial.distance import cdist

import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.decomposition import PCA
from scipy.interpolate import LinearNDInterpolator
from utils.color_palette import get_color_palette
import torch
from sklearn.mixture import GaussianMixture


def compute_filaments_angles(tomo_path,cloud_radius,min_volume):
    path = str(Path(tomo_path).parent.parent)
    tomo_name = tomo_path.split('/')[-1].split('_Vol')[0]

    if not os.path.exists(os.path.join(path,'filaments_angles',tomo_name+'.npy')):
        skeleton = zarr.load(os.path.join(path,'skeletons',tomo_name+'_pm_ctl.zarr'))
        normals = zarr.load(os.path.join(path,'normals',tomo_name+'.zarr'))
        membrane_segmentation = mrcfile.open(os.path.join(path,'segmentations',tomo_name+'_pm_ctl.mrc'),
                                                          permissive=True).data
        membrane_segmentation = np.array(membrane_segmentation,dtype=np.float32)

        membrane_segmentation_cancer = mrcfile.open(os.path.join(path,'segmentations',tomo_name+'_pm_p815.mrc'),
                                                          permissive=True).data
        membrane_segmentation += np.array(membrane_segmentation_cancer,dtype=np.float32)

        actin_segmentation = mrcfile.open(os.path.join(path,'actin_segmentations',tomo_name+'.mrc'),
                                                          permissive=True).data
        actin_segmentation = np.array(actin_segmentation,dtype=np.float32)

        actin_segmentation = np.multiply(actin_segmentation,1-membrane_segmentation)

        cloud = distance_transform_edt(1-membrane_segmentation)
        cloud = np.array(cloud<cloud_radius,dtype=np.float32)

        restricted_segmentation = np.multiply(actin_segmentation,cloud)
        labeled_mask, num_components = label(restricted_segmentation)

        position_ctl = np.stack(np.where(membrane_segmentation>0)).T
        position_p815 = np.stack(np.where(membrane_segmentation_cancer>0)).T
        position_skeleton = np.stack(np.where(skeleton>0)).T
        filaments_angles = {'position':[],
                            'angle':[]}
        for l in tqdm(range(1,num_components+1)):
            if np.sum(labeled_mask==l) < min_volume:
                continue
            positions = np.stack(np.where(labeled_mask==l)).T
            centroid = positions.mean(axis=0)

            distance_ctl = np.amin(np.linalg.norm(np.expand_dims(centroid,0) - position_ctl,axis=1))
            distance_p815 = np.amin(np.linalg.norm(np.expand_dims(centroid,0) - position_p815,axis=1))

            if distance_p815 < distance_ctl:
                continue

            pca = PCA(n_components=1)
            pca.fit(positions - centroid)

            direction = pca.components_[0]

            centroid = np.array(centroid,dtype=np.int32)

            projection_to_skeleton = np.linalg.norm(np.expand_dims(centroid,0) - position_skeleton,axis=1)
            projection_to_skeleton = position_skeleton[np.argmin(projection_to_skeleton)]
            normal = normals[projection_to_skeleton[0],projection_to_skeleton[1],projection_to_skeleton[2]]

            normal, direction = normal/np.linalg.norm(normal), direction/np.linalg.norm(direction)
            angle = 180*np.arccos(np.sum(normal*direction))/np.pi
            angle = np.abs(angle-90)

            filaments_angles['angle'].append(angle)
            filaments_angles['position'].append(projection_to_skeleton)

        np.save(os.path.join(path,'filaments_angles',tomo_name+'.npy'),filaments_angles)
    else:
        filaments_angles = np.load(os.path.join(path,'filaments_angles',tomo_name+'.npy'),allow_pickle=True).item()

    return filaments_angles

#############################
#############################
#############################
# The following is an example on how to use the code
# to compute the filaments orientation and interaction
# with the membrane
#############################
#############################
#############################

# tomo_path = '/Users/joel/PycharmProjects/lustre/actinergy/tomos/Position_28_2_Vol_px10.mrc'
# cloud_radius = 25
# min_volume = 100
#
# path = str(Path(tomo_path).parent.parent)
# tomo_name = tomo_path.split('/')[-1].split('_Vol')[0]
# filaments_angles = compute_filaments_angles(tomo_name,cloud_radius)
#
# colors = get_color_palette()
# means_path = os.path.join(path,'params','gmm_means.npy')
# weights_path = os.path.join(path,'params','gmm_weights.npy')
#
# means = [np.load(means_path)]
# weights = [np.load(weights_path)]
#
# membrane_distances = torch.load(os.path.join(path,'distances',tomo_name+'_pm.pt'))
# gm = GaussianMixture(n_components=5, covariance_type='tied',
#                  means_init=np.mean(means, axis=0),
#                  weights_init=np.mean(weights, axis=0)).fit(
# membrane_distances['ctl_to_p815']['distance'].unsqueeze(-1))
#
# classes = gm.predict(membrane_distances['ctl_to_p815']['distance'].unsqueeze(-1))
# pca = PCA(n_components=2)
# projected_classes = pca.fit_transform(membrane_distances['ctl_to_p815']['position'])
#
# nx, ny = np.array(np.amax(projected_classes)-np.amin(projected_classes, axis=0),dtype=np.int32)
# projection = cdist(membrane_distances['ctl_to_p815']['position'],
#                    filaments_angles['position'])
# projection = np.argmin(projection,axis=0)
# projection = projected_classes[projection]
#
# plt.scatter(projected_classes[:,0],projected_classes[:,1],c=np.array(colors['regions'])[classes])
# plt.scatter(projection[:,0],projection[:,1],c=filaments_angles['angle'],cmap='seismic')
# plt.colorbar()
# plt.show()
#
# dists = cdist(projected_classes,projection)
# closest_position = np.argmin(dists,axis=0)
# distance_to_filaments = np.array(membrane_distances['ctl_to_p815']['distance'])[closest_position]
# # classes_to_filaments = np.array(classes[closest_position])
# # for c in np.unique(classes):
# #     plt.hist(np.array(filaments_angles['angle'])[classes_to_filaments == c],
# #              density=True,bins=15,alpha=0.5,
# #              color=colors['regions'][c])
# #
# # plt.show()
#
# plt.scatter(distance_to_filaments,filaments_angles['angle'])
# plt.xlabel('distance to cancer cell')
# plt.ylabel('angle')
# plt.show()
#
# plt.hist(distance_to_filaments,density=True,bins=25)
# plt.xlabel('distance to cancer cell')
# plt.ylabel('density')
# plt.show()
