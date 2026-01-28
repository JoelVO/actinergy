import nrrd
import numpy as np
import mrcfile

data, header = nrrd.read('/Users/joel/Downloads/actin.nrrd')
data = np.array(data,dtype=np.uint8)
mrcfile.write('/Users/joel/PycharmProjects/lustre/actinergy/actin_segmentations/Position_28_2.mrc',
              np.transpose(data,axes=(2,1,0)),
              overwrite=True,
              voxel_size=10)

print(data.shape)