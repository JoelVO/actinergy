import mrcfile
import numpy as np
from scipy.ndimage import distance_transform_edt
import scipy
import zarr

def skeletonize(tomo_path: str):
    """
    :param tomo_path: path for membrane segmentation. mrc file expected
    :param save_path: path to store the skeleton of the membranes
    :return: the skeleton of the tomograms will be store as a .npy file
    """
    tomo = mrcfile.open(tomo_path,
                        permissive=True).data
    tomo = np.array(tomo,dtype=np.float32)

    s = 3
    kernel = np.ones((s,s,s))/s
    tomo = scipy.signal.convolve(tomo,kernel,mode='same')
    tomo = np.array(tomo>0.5,dtype=np.float32)
    skeleton = np.zeros_like(tomo)

    for slice in range(tomo.shape[0]):
        distance = distance_transform_edt(tomo[slice])

        positions = np.stack(np.where(distance>0))

        positions_01 = positions + np.expand_dims(np.array([-1,0]),-1)
        positions_21 = positions + np.expand_dims(np.array([1,0]),-1)
        positions_10 = positions + np.expand_dims(np.array([0,-1]),-1)
        positions_12 = positions + np.expand_dims(np.array([0,1]),-1)

        positions_01 = np.clip(positions_01,0,distance.shape[0]-1)
        positions_21 = np.clip(positions_21,0,distance.shape[0]-1)
        positions_10 = np.clip(positions_10,0,distance.shape[1]-1)
        positions_12 = np.clip(positions_12,0,distance.shape[1]-1)

        comparison_x = np.logical_and(distance[positions[0],positions[1]] >= distance[positions_01[0],positions_01[1]],
                                      distance[positions[0],positions[1]] >= distance[positions_21[0],positions_21[1]])
        comparison_y = np.logical_and(distance[positions[0],positions[1]] >= distance[positions_10[0],positions_10[1]],
                                      distance[positions[0],positions[1]] >= distance[positions_12[0],positions_12[1]])
        comparison = np.logical_and(comparison_x,comparison_y)
        peaks = np.zeros_like(distance)
        peaks[positions[0][comparison],positions[1][comparison]] = 1

        skeleton[slice] = peaks

    return skeleton
