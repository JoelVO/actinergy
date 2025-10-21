import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from utils.cropping_membrane import crop_membrane
from utils.color_palette import get_color_palette
from utils.functions import average_gmm
import mrcfile
from glob import glob
import cv2
import imageio.v2 as imageio
from io import BytesIO
from tqdm import tqdm

def contrast_stretch(img, input_black, input_white):
    # Convert to float32 for precise math
    img = img.astype(np.float32)

    # Clip input range to avoid division by zero
    input_black = max(input_black, 0)
    input_white = min(input_white, 255)
    if input_white == input_black:
        raise ValueError("input_white and input_black cannot be the same")

    # Apply contrast stretching
    stretched = (img - input_black) * (255.0 / (input_white - input_black))
    stretched = np.clip(stretched, 0, 255)  # Ensure values are in byte range
    return stretched.astype(np.uint8)


path = '/Users/joel/PycharmProjects/lustre/actinergy/distances'
colors = get_color_palette()


distances,membranes = {},{}
for _,name in enumerate(os.listdir(path)):
    membrane = torch.load(f'{path}/{name}')
    membrane = crop_membrane(membrane,low_lim=0.1,up_lim=0.9)
    distances[name] = membrane['ctl_to_p815']['distance']
    membranes[name] = membrane['ctl_to_p815']

    if len(distances[name]) == 0:
        continue

#distances = torch.cat(distances)
means_path = path.replace('distances', 'params') + '/gmm_means.npy'
weights_path = path.replace('distances', 'params') + '/gmm_weights.npy'
if not (os.path.exists(means_path) and os.path.exists(weights_path)):
    gm, means, covariances, weights = average_gmm(n_components=5,
                                                  covariance_type='tied',
                                                  means_init=[[75],[150],[200],[250],[410]],
                                                  fit_data=torch.cat(list(distances.values())).unsqueeze(-1))
else:
    means = [np.load(means_path)]
    weights = [np.load(weights_path)]

gm = GaussianMixture(n_components=5, covariance_type='tied',
                     means_init=np.mean(means,axis=0),
                     weights_init=np.mean(weights,axis=0)).fit(
    torch.cat(list(distances.values())).unsqueeze(-1))


for tomo_path in glob('/Users/joel/PycharmProjects/lustre/actinergy/tomos/*.mrc'):
    print(tomo_path)

    if '39' not in tomo_path:
        continue

    '''
    if os.path.exists(tomo_path.replace('tomos','gifs').replace('mrc','gif')):
        continue
    if not os.path.exists(tomo_path.replace('Vol_px10','pm_ctl')
                            .replace('tomos','segmentations')):
        continue
    '''
    tomo = mrcfile.open(tomo_path,permissive=True).data
    tomo = np.array(tomo,dtype=np.float32)

    tomo -= np.amin(tomo)
    tomo = 255*tomo/np.amax(tomo)
    tomo = np.array(tomo,dtype=np.uint8)


    membrane = mrcfile.open(tomo_path.replace('Vol_px10','pm_ctl')
                            .replace('tomos','segmentations'),permissive=True).data
    membrane = np.expand_dims(np.array(membrane,dtype=np.bool),-1)
    membrane = torch.tensor(membrane).squeeze()
    # membrane = np.logical_not(membrane)

    if not os.path.exists(tomo_path.replace('tomos','distances')
                                    .replace('Vol_px10.mrc','pm.pt')):
        continue

    membrane_distances = torch.load(tomo_path.replace('tomos','distances')
                                    .replace('Vol_px10.mrc','pm.pt'))
    classes = gm.predict(membrane_distances['ctl_to_p815']['distance'].unsqueeze(-1))
    if not os.path.exists(tomo_path.replace('tomos','masks')
               .replace('mrc','pt')):
        membrane_classes = -torch.ones(membrane.shape).unsqueeze(-1).repeat((1,1,1,3))
        membrane_positives = torch.stack(torch.where(membrane>0.5)).transpose(0,1)
        for mp in tqdm(membrane_positives):
            d = torch.sum((membrane_distances['ctl_to_p815']['position']-mp.unsqueeze(0))**2,dim=-1)
            c = classes[torch.argmin(d)]
            c = np.argsort(np.ndarray.flatten(gm.means_))[c]

            rgb = tuple(int(colors['regions'][c].lstrip('#')[i:i + 2], 16) for i in (0, 2, 4))
            membrane_classes[mp[0],mp[1],mp[2]]=torch.tensor(rgb)

        membrane_classes = membrane_classes.type(torch.uint8)
        torch.save(membrane_classes,tomo_path.replace('tomos','masks')
                   .replace('mrc','pt'))
    else:
        membrane_classes = torch.load(tomo_path.replace('tomos','masks')
               .replace('mrc','pt'))
    frames = []
    for _ in tqdm(range(len(tomo))):
        slice = tomo[_]
        #slice = cv2.equalizeHist(slice)
        #slice = cv2.convertScaleAbs(slice, alpha=0.8, beta=-10)
        #slice = contrast_stretch(slice,66,194)
        slice = contrast_stretch(slice, 151, 188)
        slice = cv2.cvtColor(slice, cv2.COLOR_GRAY2RGB)

        slice = np.where(membrane_classes[_]==-1,slice,membrane_classes[_])

        fig = plt.figure()
        plt.imshow(slice)
        plt.axis('off')

        fig.savefig(f'/Users/joel/PycharmProjects/scratch/{_}.png')

        buf = BytesIO()
        plt.savefig(buf, format='png')
        plt.close(fig)  # Close the figure to free memory
        buf.seek(0)

        # Read the image from the buffer and add it to the frames list
        frames.append(imageio.imread(buf))

    imageio.mimsave(tomo_path.replace('tomos','gifs').replace('mrc','gif'),
                    frames, fps=5)

