import mrcfile
import numpy as np
from scipy.ndimage import distance_transform_edt
import pyvista as pv
import torch
import matplotlib.pyplot as plt

actin = mrcfile.open('/Users/joel/PycharmProjects/scratch/actin_giant.mrc').data
actin = np.asarray(actin,dtype=np.float32)

x = distance_transform_edt(actin)

x[x<2.3]=0
grid = pv.ImageData(dimensions=x.shape)
grid.point_data["EDT"] = x.flatten(order="F")

# plotter = pv.Plotter()
#
# plotter.add_volume(grid, scalars="EDT", cmap="viridis", opacity="sigmoid")
#
# contours = grid.contour(isosurfaces=[x.max() * 0.25, x.max() * 0.5, x.max() * 0.75])
# plotter.add_mesh(contours, opacity=0.3, cmap="coolwarm")
# plotter.add_axes()
# plotter.show()
x[x>0]=1

x = torch.from_numpy(x).unsqueeze(0).unsqueeze(0).type(torch.float32)
kernel = torch.ones((1,1,3,3,3))
x = torch.nn.functional.conv3d(x, kernel, padding='same')
x = x.squeeze() > 0

x = x.type(torch.uint8).numpy()
mrcfile.write('/Users/joel/PycharmProjects/lustre/actinergy/actin_segmentations/Position_28_2.mrc',
              x,
              overwrite=True,
              voxel_size=10)