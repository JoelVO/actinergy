import os
import torch
import numpy as np
from utils.cropping_membrane import crop_membrane
from utils.color_palette import get_color_palette
from scipy.stats import wasserstein_distance
from sklearn.cluster import AgglomerativeClustering


path = '/Users/joel/PycharmProjects/lustre/actinergy/distances'
colors = get_color_palette()

distances = {}
for _,name in enumerate(os.listdir(path)):
    membrane = torch.load(f'{path}/{name}')
    membrane = crop_membrane(membrane,low_lim=0.1,up_lim=0.9)
    hist,edges = np.histogram(membrane['ctl_to_p815']['distance'],bins=100,density=True,range=(0,400))
    distances[name] = hist


    if len(distances[name]) == 0:
        continue

D = np.zeros((len(distances.keys()),len(distances.keys())))

for n,name in enumerate(distances.keys()):
    for m,target in enumerate(distances.keys()):

        D[n,m] = wasserstein_distance(distances[name],distances[target])

model = AgglomerativeClustering(n_clusters=3, metric='precomputed', linkage='average')
labels = model.fit_predict(D)

for c in np.unique(labels):
    print(c)
    for l,label in enumerate(labels):
        if label == c:
            print(list(distances.keys())[l])

