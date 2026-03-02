import numpy as np
import pylab as pl
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

aggregated_data = np.load('/Users/joel/PycharmProjects/lustre/actinergy/tomogram_measurements/Position_28_2_Vol_px10.npy',
                          allow_pickle=True).item()
aggregated_data['distance_ctl_to_p815'] = np.array(aggregated_data['distance_ctl_to_p815'])

pca = PCA(n_components=2)
projected_skeleton = pca.fit_transform(aggregated_data['skeleton_ctl'])

fig,axs = pl.subplots(ncols=3,nrows=3)
axs[0,0].scatter(projected_skeleton[:,0],projected_skeleton[:,1],
               c=aggregated_data['distance_ctl_to_p815'],
               s=1,
               cmap='viridis')
axs[0,0].set_adjustable('box')
axs[0,0].set_aspect(1)
axs[0,0].axis('off')
axs[0,0].set_title('distance from ctl to p815')

axs[0,1].scatter(projected_skeleton[:,0],projected_skeleton[:,1],
               c=aggregated_data['distance_ctl_to_p815'],
               s=1,
               cmap='viridis')
axs[0,1].scatter(projected_skeleton[:,0][aggregated_data['actin_projection'][10]==1],
                 projected_skeleton[:,1][aggregated_data['actin_projection'][10]==1],
                   c='#FAB433',
                   s=1)
axs[0,1].set_adjustable('box')
axs[0,1].set_aspect(1)
axs[0,1].axis('off')
axs[0,1].set_title('actin projection over distances')

axs[0,2].scatter(projected_skeleton[:,0],projected_skeleton[:,1],
               c=aggregated_data['curvature'][50],
               s=1,
               cmap='seismic')
axs[0,2].scatter(projected_skeleton[:,0][aggregated_data['actin_projection'][10]==1],
                 projected_skeleton[:,1][aggregated_data['actin_projection'][10]==1],
                   c='#FAB433',
                   s=1)
axs[0,2].set_adjustable('box')
axs[0,2].set_aspect(1)
axs[0,2].axis('off')
axs[0,2].set_title('actin projection over curvature')

for _,k in enumerate(aggregated_data['curvature'].keys()):
    axs[1,_].scatter(projected_skeleton[:,0],projected_skeleton[:,1],
                   c=aggregated_data['curvature'][k],
                   s=1,
                   cmap='seismic')
    axs[1,_].set_adjustable('box')
    axs[1,_].set_aspect(1)
    axs[1,_].set_title(f'window: {k}')
    axs[1,_].axis('off')

for _,k in enumerate(aggregated_data['actin_projection'].keys()):
    axs[2,_].scatter(projected_skeleton[:,0],projected_skeleton[:,1],
                   c=aggregated_data['actin_projection'][k],
                   s=1,
                   cmap='viridis')
    axs[2,_].set_adjustable('box')
    axs[2,_].set_aspect(1)
    axs[2,_].set_title(f'{k} nn')
    axs[2, _].axis('off')
plt.show()