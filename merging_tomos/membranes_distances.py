import os

import zarr
import numpy as np
from tqdm import tqdm
import torch
from pathlib import Path


def dist_to_membrane(anchor,other,pixel_size):
    dist = {'position':[],
            'distance':[]}
    for _ in tqdm(range(len(anchor))):
        d = np.sum((other-np.expand_dims(anchor[_],axis=0))**2,axis=-1)
        if len(d) == 0:
            continue
        dist['position'].append(anchor[_])
        dist['distance'].append(pixel_size*np.sqrt(np.amin(d)))

    return {'position':torch.tensor(np.array(dist['position'])),
            'distance':torch.tensor(np.array(dist['distance']))}


def compute_distance(name,data_path,save_path,pixel_size):
    ctl = zarr.load(f'{data_path}/{name}_ctl.zarr')
    p815 = zarr.load(f'{data_path}/{name}_p815.zarr')

    sk_ctl = torch.tensor(np.stack(np.where(ctl>0)).T).type(torch.float32)
    sk_p815 = torch.tensor(np.stack(np.where(p815>0)).T).type(torch.float32)

    if (len(sk_ctl) != 0) and (len(sk_p815)!=0):
        sk_p815 = sk_p815.numpy()
        sk_ctl = sk_ctl.numpy()

        ctl_to_p815 = dist_to_membrane(anchor=sk_ctl,other=sk_p815,pixel_size=pixel_size)
        p815_to_ctl = dist_to_membrane(anchor=sk_p815,other=sk_ctl,pixel_size=pixel_size)

        torch.save({'ctl_to_p815':ctl_to_p815,
                    'p815_to_ctl':p815_to_ctl},
                   f'{save_path}/{name}.pt')





def compute_distances_dataset(path:str, save_path:str=None, pixel_size:float=10):
    """
    Compute the distances between the p815 and ctl membranes.
    It expects to receive the skeletons from the ctl (sk_ctl) and p815 (sk_p815) plasma membranes.
    Given a point at the sk_ctl th distance assigned to it is the minimum Euclidian distance to any point in sk_p815
    :param path: path containing the sk_ctl and sk_815. Their names are expected to end with _ctl.zarr
    and _p815.zarr resp.
    :param save_path: path where to save the distances. It will save is in a sibling directory to path
    called 'distances'.
    :param pixel_size: tomo pixel size.
    :return:
    """
    if save_path is None:
        save_path = path.replace('skeletons','distances')
    for name in os.listdir(f'{path}'):
        if 'p815' in name:
            continue
        if os.path.exists(f"{save_path}/{name.replace('_ctl.zarr','')}.pt"):
            continue

        distance_path = Path(save_path)
        distance_path.parent.mkdir(parents=True,exist_ok=True)

        compute_distance(name=name.replace('_ctl.zarr',''),
                         data_path=path,
                         save_path=save_path,
                         pixel_size=pixel_size)

