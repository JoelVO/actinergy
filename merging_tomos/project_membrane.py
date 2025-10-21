import numpy as np
import zarr
import matplotlib.pyplot as plt
from glob import glob
import torch
import os
from sklearn.decomposition import PCA
from tqdm import tqdm
from sklearn.mixture import GaussianMixture
from utils.functions import average_gmm
from utils.cropping_membrane import crop_membrane
from utils.color_palette import get_color_palette

# for tomo_path in glob('/Users/joel/PycharmProjects/lustre/actinergy/tomos/*.mrc'):
#     # print(tomo_path)
#     if os.path.exists(tomo_path.replace('tomos','figures/projected_membranes').replace('.mrc','.pdf')):
#         continue
#     if not os.path.exists(tomo_path.replace('tomos', 'masks')
#                                   .replace('mrc', 'pt')):
#         continue
#     membrane_classes = torch.load(tomo_path.replace('tomos', 'masks')
#                                   .replace('mrc', 'pt'))
#     membrane = torch.stack(torch.where(membrane_classes[...,0]!=-1)).transpose(0,1)
#
#     pca = PCA(n_components=2)
#     membrane_pca = pca.fit_transform(membrane)
#     hex_color = ['#{:02x}{:02x}{:02x}'.format(*rgb) for rgb in membrane_classes[membrane[:,0],membrane[:,1],membrane[:,2]]]
#
#     fig = plt.figure()
#     plt.scatter(membrane_pca[:,0],membrane_pca[:,1],c=hex_color)
#     plt.axis('off')
#     fig.savefig(tomo_path.replace('tomos','figures/projected_membranes').replace('.mrc','.pdf'))
#     plt.close(fig)


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

if not (os.path.exists(means_path) and os.path.exists(weights_path)):
    np.save(means_path,gm.means_)
    np.save(path.replace('distances','params')+'/gmm_cov.npy',gm.covariances_)
    np.save(weights_path,gm.weights_)

regions = {r:[] for r in colors['regions']}
for _,name in enumerate(os.listdir(path)):
    if len(distances[name]) == 0:
        continue
    pred = gm.predict(distances[name].unsqueeze(-1))
    for c in range(5):
        regions[list(regions.keys())[c]].append(distances[name][pred==np.argsort(gm.means_.flatten())[c]])
regions = {r:torch.cat(regions[r]) for r in regions.keys()}

means_region_path = path.replace('distances', 'params') + '/gmm_means_region.npy'
weights_region_path = path.replace('distances', 'params') + '/gmm_weights_region.npy'
gm_region, means, covariances, weights = average_gmm(n_components=3,
                                              covariance_type='full',
                                              means_init=None,
                                              fit_data=torch.cat((regions['#FAB433'],regions['#229328'])).unsqueeze(-1))


gm_region = GaussianMixture(n_components=3, covariance_type='full',
                     means_init=np.mean(means,axis=0),
                     weights_init=np.mean(weights,axis=0)).fit(
    torch.cat((regions['#FAB433'],regions['#229328'])).unsqueeze(-1))

for name in membranes.keys():
    print(name)
    pred = gm.predict(membranes[name]['distance'].unsqueeze(-1))
    pred_regions = gm_region.predict(membranes[name]['distance'].unsqueeze(-1))

    pca = PCA(n_components=2)
    membrane_pca = pca.fit_transform(membranes[name]['position'])

    fig = plt.figure()
    for _ in reversed(range(5)):
        plt.scatter(membrane_pca[pred == np.argsort(gm.means_.flatten())[_], 0],
                    membrane_pca[pred == np.argsort(gm.means_.flatten())[_], 1],
                    c=colors['regions'][_],s=2)
    plt.axis('off')
    fig.savefig(path.replace('distances', 'figures') + f'/projected_membranes/{name.replace(".pt", ".png")}')
    plt.close(fig)

    fig = plt.figure()
    for _ in range(5):
        if not ((_==1) or (_==2)):
            plt.scatter(membrane_pca[pred==np.argsort(gm.means_.flatten())[_], 0],
                    membrane_pca[pred==np.argsort(gm.means_.flatten())[_], 1],
                    c=colors['regions'][_],s=2)
        else:
            for r in range(3):
                condition = np.logical_or(pred==np.argsort(gm.means_.flatten())[1],
                                           pred==np.argsort(gm.means_.flatten())[2])
                condition = np.logical_and(condition,
                                           pred_regions==r)
                plt.scatter(membrane_pca[condition,0],
                            membrane_pca[condition,1],
                            c=colors['shades_peaks'][r],s=2)
    plt.axis('off')
    fig.savefig(path.replace('distances','figures')+f'/projected_membranes/{name.replace(".pt","_regions.png")}')
    plt.close(fig)



