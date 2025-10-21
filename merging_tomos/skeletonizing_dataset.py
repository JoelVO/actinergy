import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.skeletonizing import skeletonize
from glob import glob
from tqdm import tqdm
from pathlib import Path

def skeletonize_dataset(segmentations_path: str):
    """
    Function to turn segmentations into skeletons.
    It expects .mrc files and saves the skeletons as .zarr
    :param segmentations_path: path of segmentations
    :return: save skeletons in a sibling folder of segmentations_path named 'skeletons'
    """
    for segmentation in tqdm(glob(segmentations_path+'/*.mrc')):
        if os.path.exists(segmentation.replace('segmentations','skeletons')
                    .replace('.mrc','.zarr')):
            continue

        parent_path = Path(segmentation.replace('segmentations','skeletons')
                           .replace('.mrc','.zarr'))
        parent_path.parent.mkdir(parents=True,exist_ok=True)

        skeletonize(segmentation,segmentation.replace('segmentations','skeletons')
                    .replace('.mrc','.zarr'))
