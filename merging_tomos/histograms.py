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
from skeletonizing_dataset import skeletonize_dataset
from membranes_distances import compute_distances_dataset


"""
Get the color palette, compute the skeletons and measures the distances
"""
colors = get_color_palette()

segmentations_path = '/Users/joel/PycharmProjects/lustre/actinergy/segmentations'
distances_path = segmentations_path.replace('segmentations',
                                                              'distances')
skeletonize_dataset(segmentations_path=segmentations_path)
compute_distances_dataset(path=distances_path)

"""
Load the distances from the whole dataset.
It only takes the membrane between the 0.1 and 0.9 percentiles on closeness to tomo's center to account for miss
and end of lamella.
"""
distances,membranes = {},{}
for _,name in enumerate(os.listdir(distances_path)):
    membrane = torch.load(f'{distances_path}/{name}')
    membrane = crop_membrane(membrane,low_lim=0.1,up_lim=0.9)
    distances[name] = membrane['ctl_to_p815']['distance']
    membranes[name] = membrane['ctl_to_p815']

    if len(distances[name]) == 0:
        continue

"""
Computes 20 independent Gaussian Mixture Models (gmm) to account for stochastic behaviors.
gmm have 5 components, but the amount can be modified with n_components value.
Covariance type was chosen empirically.
"""
n_components = 5
means_path = distances_path.replace('distances', 'params') + '/gmm_means.npy'
weights_path = distances_path.replace('distances', 'params') + '/gmm_weights.npy'
if not (os.path.exists(means_path) and os.path.exists(weights_path)):
    gm, means, covariances, weights = average_gmm(n_components=n_components,
                                                  covariance_type='tied',
                                                  fit_data=torch.cat(list(distances.values())).unsqueeze(-1))
else:
    means = [np.load(means_path)]
    weights = [np.load(weights_path)]

gm = GaussianMixture(n_components=n_components, covariance_type='tied',
                     means_init=np.mean(means,axis=0),
                     weights_init=np.mean(weights,axis=0)).fit(
    torch.cat(list(distances.values())).unsqueeze(-1))

if not (os.path.exists(means_path) and os.path.exists(weights_path)):
    np.save(means_path,gm.means_)
    np.save(distances_path.replace('distances','params')+'/gmm_cov.npy',gm.covariances_)
    np.save(weights_path,gm.weights_)


"""
gmm on yellow and green clusters
"""
regions = {r:[] for r in colors['regions']}
for _,name in enumerate(os.listdir(distances_path)):
    if len(distances[name]) == 0:
        continue
    pred = gm.predict(distances[name].unsqueeze(-1))
    for c in range(n_components):
        regions[list(regions.keys())[c]].append(distances[name][pred==np.argsort(gm.means_.flatten())[c]])
regions = {r:torch.cat(regions[r]) for r in regions.keys()}

means_region_path = distances_path.replace('distances', 'params') + '/gmm_means_region.npy'
weights_region_path = distances_path.replace('distances', 'params') + '/gmm_weights_region.npy'
gm_region, means, covariances, weights = average_gmm(n_components=3,
                                              covariance_type='full',
                                              means_init=None,
                                              fit_data=torch.cat((regions['#FAB433'],regions['#229328'])).unsqueeze(-1))


gm_region = GaussianMixture(n_components=3, covariance_type='full',
                     means_init=np.mean(means,axis=0),
                     weights_init=np.mean(weights,axis=0)).fit(
    torch.cat((regions['#FAB433'],regions['#229328'])).unsqueeze(-1))

"""
Plotting and saving cummulated histogram
"""
#if not os.path.exists('/Users/joel/PycharmProjects/lustre/actinergy/figures/cummulated_histogram.pdf'):
if True:
    fig = plt.figure()
    hist = plt.hist(torch.cat(list(distances.values())),bins=100,density=True,color=colors['neutral']['membrane'])
    for _,m in enumerate(np.sort(gm.means_.flatten())):
        domain = np.linspace(0,500,200)
        # pdf = lambda x: np.exp(-((x-m)**2)/(2*gm.covariances_[0,0]))/np.sqrt(2*np.pi*(gm.covariances_[0,0]))
        # plt.plot(domain,pdf(domain),'--',c=colors['regions'][_],label=int(m))
        plt.plot(domain, np.amax(hist[0])*gm.predict_proba(np.expand_dims(domain,axis=-1))[:,_],
                 c=colors['regions'][_],label=int(m))
    plt.legend()
    plt.xlabel('Angstroms')
    plt.ylabel('Probability density')
    fig.savefig(f'{segmentations_path.replace("segmentations","figures")}/cummulated_histogram.pdf')
    plt.close(fig)

if True:
    fig = plt.figure()
    hist = plt.hist(torch.cat((regions['#FAB433'],regions['#229328'])),bins=50,
                    density=True,color=colors['neutral']['membrane'])
    for _,m in enumerate(np.sort(gm_region.means_.flatten())):
        domain = np.linspace(100,250,200)
        plt.plot(domain, np.amax(hist[0])*gm_region.predict_proba(np.expand_dims(domain,axis=-1))[:,_],
                 c=colors['shades_peaks'][_],label=int(m),linewidth=5)
    plt.legend()
    plt.xlabel('Angstroms')
    plt.ylabel('Probability density')
    fig.savefig(f'{segmentations_path.replace("segmentations","figures")}/cummulated_histogram_regions.pdf')
    plt.close(fig)

"""
Plotting and saving individual histograms
"""
#if not os.path.exists('/Users/joel/PycharmProjects/lustre/actinergy/figures/individual_histogram.pdf'):
if True:
    fig,axs = plt.subplots(nrows=len(os.listdir(distances_path)),figsize=(10, 10))
    for _,name in enumerate(os.listdir(distances_path)):

        if len(distances[name]) == 0:
            continue
        pred = gm.predict(distances[name].unsqueeze(-1))
        for c in range(5):
            axs[_].hist(distances[name][pred==np.argsort(gm.means_.flatten())[c]],bins=25,
                        density=False,color=colors['regions'][c])
        axs[_].set_xlim(0, 400)
        axs[_].set_title(name)

    axs[-1].set_xlabel('Angstroms')
    plt.tight_layout()
    fig.savefig(f'{segmentations_path.replace("segmentations","figures")}/individual_histogram.pdf')
    plt.close(fig)

if True:
    fig,axs = plt.subplots(nrows=len(os.listdir(distances_path)),figsize=(10, 10))
    for _,name in enumerate(os.listdir(distances_path)):

        if len(distances[name]) == 0:
            continue
        pred = gm.predict(distances[name].unsqueeze(-1))
        pred_region = gm_region.predict(distances[name].unsqueeze(-1))
        for c in range(5):
            if c not in [1,2]:
                axs[_].hist(distances[name][pred==np.argsort(gm.means_.flatten())[c]],bins=25,
                        density=False,color=colors['neutral']['membrane'])
            else:
                for r in range(3):
                    condition = np.logical_and(pred_region==np.argsort(gm_region.means_.flatten())[r],
                                               np.logical_or(pred==1,pred==2))
                    axs[_].hist(distances[name][condition], bins=25,
                                density=False, color=colors['shades_peaks'][r])
        axs[_].set_xlim(0, 400)
        axs[_].set_title(name)

    axs[-1].set_xlabel('Angstroms')
    plt.tight_layout()
    fig.savefig(f'{segmentations_path.replace("segmentations","figures")}/individual_histogram_regions.pdf')
    plt.close(fig)

"""
Saving tomo masks and creating gifs with overlaid with them.
It expects for segmentations_path to have a sibling path called "tomos" with all the tomograms.

"""
for tomo_path in glob(f'{segmentations_path.replace("segmentations","tomos")}/*.mrc'):
    print(tomo_path)

    if os.path.exists(tomo_path.replace('tomos','gifs').replace('mrc','gif')):
        continue
    if not os.path.exists(tomo_path.replace('Vol_px10','pm_ctl')
                            .replace('tomos','segmentations')):
        continue

    #Loads the tomo
    tomo = mrcfile.open(tomo_path,permissive=True).data
    tomo = np.array(tomo,dtype=np.float32)

    tomo -= np.amin(tomo)
    tomo = 255*tomo/np.amax(tomo)
    tomo = np.array(tomo,dtype=np.uint8)

    #Loads the membrane segmentation
    membrane = mrcfile.open(tomo_path.replace('Vol_px10','pm_ctl')
                            .replace('tomos','segmentations'),permissive=True).data
    membrane = np.expand_dims(np.array(membrane,dtype=np.bool),-1)
    membrane = torch.tensor(membrane).squeeze()

    #It skips tomograms if their distance was not computed as earlier in the code
    if not os.path.exists(tomo_path.replace('tomos','distances')
                                    .replace('Vol_px10.mrc','pm.pt')):
        continue

    #Loads the membranes distances
    membrane_distances = torch.load(tomo_path.replace('tomos','distances')
                                    .replace('Vol_px10.mrc','pm.pt'))

    #Clusters the distances between membranes with the gmm we fitted before
    classes = gm.predict(membrane_distances['ctl_to_p815']['distance'].unsqueeze(-1))
    if not os.path.exists(tomo_path.replace('tomos','masks')
               .replace('mrc','pt')):
        #Colors the membrane following the clustering and our color palette and leaves everytihng else the same
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

    #Creates a gif for each tomo
    frames = []
    for _ in tqdm(range(len(tomo))):
        slice = tomo[_]
        slice = cv2.equalizeHist(slice)
        slice = cv2.cvtColor(slice, cv2.COLOR_GRAY2RGB)

        slice = np.where(membrane_classes[_]==-1,slice,membrane_classes[_])

        fig = plt.figure()
        plt.imshow(slice)
        plt.axis('off')

        buf = BytesIO()
        plt.savefig(buf, format='png')
        plt.close(fig)  # Close the figure to free memory
        buf.seek(0)

        # Read the image from the buffer and add it to the frames list
        frames.append(imageio.imread(buf))

    imageio.mimsave(tomo_path.replace('tomos','gifs').replace('mrc','gif'),
                    frames, fps=5)


