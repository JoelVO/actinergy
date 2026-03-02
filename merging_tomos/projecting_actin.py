import numpy as np
import matplotlib.pyplot as plt
import mrcfile
import nrrd
from glob import glob
import os
import zarr
from scipy.ndimage import distance_transform_edt
from tqdm import tqdm
from sklearn.decomposition import PCA
import torch


def project_actin():
    path = '/Users/joel/PycharmProjects/lustre/actinergy/actin_segmentations'

    for actin_folder in glob(path + '/*'):
        print(actin_folder)
        save_path = actin_folder.replace('actin_segmentations', 'projected_actin')
        save_path = save_path + '.npy'
        if os.path.exists(save_path):
            continue


        actin_filtered = mrcfile.open(os.path.join(actin_folder,'actin_filtered.mrc')).data
        actin_filtered = np.array(actin_filtered,dtype=np.float32)

        skeleton_path = actin_folder.replace('actin_segmentations', 'skeletons')
        skeleton_path = skeleton_path.replace('_Vol_px10', '_pm_ctl.zarr')
        skeleton = np.array(zarr.open(skeleton_path))
        location_skeleton = np.stack(np.where(skeleton > 0)).T
        pca = PCA(n_components=2)
        membrane_projection = pca.fit_transform(location_skeleton)

        if os.path.exists(os.path.join(actin_folder,'roi.nrrd')):
            roi,headers = nrrd.read(os.path.join(actin_folder,'roi.nrrd'))
            roi = np.transpose(roi,(2,1,0))
        else:
            actin_filtered = torch.from_numpy(actin_filtered).float().unsqueeze(0).unsqueeze(0)
            actin_filtered = torch.nn.functional.interpolate(actin_filtered,
                                                             size=(skeleton.shape[-3],skeleton.shape[-2],skeleton.shape[-1]),
                                                             mode='trilinear')[0][0].numpy()

            roi = np.ones_like(actin_filtered)

        actin_filtered = np.multiply(actin_filtered,roi)



        data = {'membrane_projection': membrane_projection,}
        for radius in [10,25,50]:
            cloud = distance_transform_edt(1-skeleton)
            cloud = np.array(cloud<radius,dtype=np.float32)

            close_actin = np.multiply(actin_filtered,cloud)
            location_actin = np.stack(np.where(close_actin>0)).T

            projected_actin = np.zeros(len(location_skeleton))
            for _ in tqdm(range(len(location_actin))):
                d = np.linalg.norm(location_skeleton-np.expand_dims(location_actin[_],axis=0),axis=-1)
                projected_actin[np.argmin(d)] = 1

            data[radius] = projected_actin


        np.save(save_path,data,allow_pickle=True)


for projection_path in glob('/Users/joel/PycharmProjects/lustre/actinergy/projected_actin/*.npy'):
    x = np.load(projection_path,allow_pickle=True).item()

    fig,axs = plt.subplots(ncols=3)
    axs[0].scatter(np.array(x['membrane_projection'])[:,0],
                np.array(x['membrane_projection'])[:,1],c=x[10],cmap='viridis',s=2)
    axs[0].set_adjustable('box')
    axs[0].set_aspect(1)
    axs[0].axis('off')
    axs[0].set_title('10 nn cloud')

    axs[1].scatter(np.array(x['membrane_projection'])[:, 0],
                   np.array(x['membrane_projection'])[:, 1], c=x[25], cmap='viridis',s=2)
    axs[1].set_adjustable('box')
    axs[1].set_aspect(1)
    axs[1].axis('off')
    axs[1].set_title('25 nn cloud')

    axs[2].scatter(np.array(x['membrane_projection'])[:, 0],
                   np.array(x['membrane_projection'])[:, 1], c=x[50], cmap='viridis',s=2)
    axs[2].set_adjustable('box')
    axs[2].set_aspect(1)
    axs[2].axis('off')
    axs[2].set_title('50 nn cloud')

    name = projection_path.split('/')[-1].split('.npy')[0]
    fig.savefig(f'/Users/joel/PycharmProjects/lustre/actinergy/figures_projected_actin/{name}.png',
                bbox_inches='tight',
                pad_inches=0,)

