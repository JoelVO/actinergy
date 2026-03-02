from skeletonizing import skeletonize
import numpy as np
import os
import mrcfile
from ctl_p815_distance import dist_to_membrane
from normals import compute_normals
from curvature import  compute_curvature
from projecting_actin import project_actin
from pathlib import Path

def get_locations(mask_path):
    mask = mrcfile.open(mask_path,permissive=True).data
    mask = np.array(mask,dtype=np.float32)
    return np.stack(np.where(mask>0)).T

def reconstruct_mask(tomo_shape,locations,values=1):
    mask = np.zeros(tomo_shape)
    mask[locations[:,0],locations[:,1],locations[:,2]] = values
    return mask

def aggregate_data(tomo_path,save_path):
    if not os.path.exists(save_path):

        aggregated_data = {'tomogram_path':tomo_path}
        tomogram = mrcfile.open(tomo_path)
        aggregated_data['tomo_shape'] = tomogram.data.shape
        aggregated_data['voxel_size'] = tomogram.voxel_size.item()[0]

        segmentation_ctl_path = tomo_path.replace('tomos',
                                             'segmentations/membrane').replace('Vol_px10.mrc',
                                                                               'pm_ctl.mrc')
        segmentation_p815_path = segmentation_ctl_path.replace('ctl','p815')

        aggregated_data['mask_ctl'] = get_locations(segmentation_ctl_path)
        aggregated_data['mask_p815'] = get_locations(segmentation_p815_path)
        aggregated_data['skeleton_ctl'] = np.stack(np.where(skeletonize(segmentation_ctl_path)>0)).T
        aggregated_data['skeleton_p815'] = np.stack(np.where(skeletonize(segmentation_p815_path)>0)).T

        if 'distance_ctl_to_p815' not in aggregated_data.keys():
            print('computing distances from ctl to p815')
            aggregated_data['distance_ctl_to_p815'] = dist_to_membrane(aggregated_data['skeleton_ctl'],
                                                                       aggregated_data['skeleton_p815'],
                                                                       aggregated_data['voxel_size'])

        if 'normals' not in aggregated_data.keys():
            print('computing normals of ctl')
            aggregated_data['normals'] = compute_normals(mask=reconstruct_mask(tomo_shape=aggregated_data['tomo_shape'],
                                                                               locations=aggregated_data['mask_ctl']),
                                                         skeleton=reconstruct_mask(tomo_shape=aggregated_data['tomo_shape'],
                                                                                   locations=aggregated_data['skeleton_ctl']),
                                                         locations_p815=aggregated_data['skeleton_p815'])

        if 'curvature' not in aggregated_data.keys():
            print('computing curvature of ctl')
            aggregated_data['curvature'] = {}
            for window in [25,50,75]:
                aggregated_data['curvature'][window] = compute_curvature(skeleton_position=aggregated_data['skeleton_ctl'],
                                                             normals=reconstruct_mask(
                                                                 tomo_shape=aggregated_data['tomo_shape'] + (3,),
                                                                 locations=aggregated_data['skeleton_ctl'],
                                                                 values=aggregated_data['normals']),
                                                             window=window)

        if 'actin_projection' not in aggregated_data.keys():
            print('projecting actin on ctl')
            actin_path = tomo_path.replace('tomos', 'segmentations/actin').replace('.mrc', '')
            actin_path = os.path.join(actin_path, 'actin_filtered.mrc')
            roi_path = actin_path.replace('actin_filtered.mrc', 'roi.nrrd')
            aggregated_data['actin_projection'] = {}
            for radius in [10,25,50]:
                aggregated_data['actin_projection'][radius] = project_actin(actin_path=actin_path,
                                                                skeleton=reconstruct_mask(aggregated_data['tomo_shape'],
                                                                                          aggregated_data['skeleton_ctl']),
                                                                location_skeleton=aggregated_data['skeleton_ctl'],
                                                                roi_path=roi_path,
                                                                tomo_shape=aggregated_data['tomo_shape'],
                                                                radius=radius
                                                                )

        parent_path = str(Path(save_path).parent)
        if not os.path.exists(parent_path):
            os.makedirs(parent_path,exist_ok=True)

    else:
        aggregated_data = np.load(save_path,allow_pickle=True).item()

    np.save(save_path,aggregated_data,allow_pickle=True)
    return aggregated_data


######################################################################################
######################################################################################
######################################################################################
# Example on how to compute the data in a whole dataset of tomograms
from glob import glob

for tomo_path in glob('/Users/joel/PycharmProjects/lustre/actinergy/tomos/*'):
    print(tomo_path)
    save_path = tomo_path.replace('tomos',
                                           'tomogram_measurements').replace('.mrc',
                                                                            '.npy')
    aggregate_data(tomo_path=tomo_path,
                   save_path=save_path)

######################################################################################
######################################################################################
######################################################################################