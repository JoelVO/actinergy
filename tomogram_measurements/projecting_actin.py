import numpy as np
import mrcfile
import nrrd
import os
from scipy.ndimage import distance_transform_edt
from tqdm import tqdm
import torch


def project_actin(actin_path,skeleton,location_skeleton,roi_path,tomo_shape,radius):

    actin_filtered = mrcfile.open(actin_path).data
    actin_filtered = np.array(actin_filtered,dtype=np.float32)

    if os.path.exists(roi_path):
        roi,headers = nrrd.read(roi_path)
        roi = np.transpose(roi,(2,1,0))
    else:
        actin_filtered = torch.from_numpy(actin_filtered).float().unsqueeze(0).unsqueeze(0)
        actin_filtered = torch.nn.functional.interpolate(actin_filtered,
                                                         size=(tomo_shape[-3],tomo_shape[-2],tomo_shape[-1]),
                                                         mode='trilinear')[0][0].numpy()

        roi = np.ones_like(actin_filtered)

    actin_filtered = np.multiply(actin_filtered,roi)

    cloud = distance_transform_edt(1-skeleton)
    cloud = np.array(cloud<radius,dtype=np.float32)

    close_actin = np.multiply(actin_filtered,cloud)
    location_actin = np.stack(np.where(close_actin>0)).T
    projected_actin = np.zeros(len(location_skeleton))
    for _ in tqdm(range(len(location_actin))):
        d = np.linalg.norm(location_skeleton-np.expand_dims(location_actin[_],axis=0),axis=-1)
        projected_actin[np.argmin(d)] = 1

    return projected_actin

