import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import imageio
from io import BytesIO
import torch
from sklearn.decomposition import PCA
import os
from tqdm import tqdm

# def rotate_x(points, theta):
#     R = np.array([
#         [1, 0, 0],
#         [0, np.cos(theta), -np.sin(theta)],
#         [0, np.sin(theta),  np.cos(theta)],
#     ])
#     return points @ R.T
#
# def rotate_y(points, theta):
#     R = np.array([
#         [ np.cos(theta), 0, np.sin(theta)],
#         [ 0,             1, 0            ],
#         [-np.sin(theta), 0, np.cos(theta)]
#     ])
#     return points @ R.T

def rotate_z(points, theta):
    R = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta),  np.cos(theta), 0],
        [0,              0,             1],
    ])
    return points @ R.T

mask_path = '/Users/joel/PycharmProjects/lustre/actinergy/masks'

for mask_name in os.listdir(mask_path):
    print(mask_name)
    # mask = torch.load('/Users/joel/PycharmProjects/lustre/actinergy/masks/Position_28_2_Vol_px10.pt')
    mask = torch.load(os.path.join(mask_path, mask_name))
    cloud_point = torch.stack(torch.where(mask.amin(dim=-1) < 255))
    pca = PCA(n_components=2)
    membrane_pca = pca.fit_transform(cloud_point.permute((1,0)))
    x,y = membrane_pca.T
    z = np.zeros_like(x)

    points = np.stack((x,y,z),axis=-1)
    theta = np.radians(30)
    x,y,z = rotate_z(points, theta).T

    colors = mask[cloud_point[0],cloud_point[1],cloud_point[2]]/255
    frames = []
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_box_aspect((mask.shape[2],mask.shape[0],mask.shape[1]))

    for angle in tqdm(range(0, 360, 2)):  # 2° step = 180 frames
        ax.cla()  # Clear axis

        ax.scatter(z,y,x, s=1, c=colors, marker='.')
        ax.view_init(elev=0,azim=angle)  # Rotate camera

        plt.tight_layout()
        plt.axis('off')
        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        frames.append(imageio.imread(buf))

    plt.close()

    # ---------------------------------------------------
    # 3. Save to GIF
    # ---------------------------------------------------
    imageio.mimsave(os.path.join(mask_path.replace('masks',
                                                   'gifs/rotating_membrane'),mask_name.replace('.pt','.gif')),
                    frames, fps=1)

 