import numpy as np
from tqdm import tqdm


def dist_to_membrane(anchor,other,pixel_size):
    dist = []
    for _ in tqdm(range(len(anchor))):
        d = np.sum((other-np.expand_dims(anchor[_],axis=0))**2,axis=-1)
        dist.append(pixel_size*np.sqrt(np.amin(d)))

    return np.array(dist)

