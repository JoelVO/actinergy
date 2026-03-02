import numpy as np
from sklearn.mixture import GaussianMixture
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt
import imageio.v2 as imageio
from io import BytesIO

def average_gmm(n_components,covariance_type,fit_data,means_init=None):
    """
    :param n_components: gmm components
    :param covariance_type: covariance type from gmm
    :param fit_data: data to fit models on
    :param means_init: possibility to provide initial means
    :return:
    """
    means = []
    covariances = []
    weights = []
    for _ in range(20):
        gm = GaussianMixture(n_components=n_components,covariance_type=covariance_type,
                             means_init=means_init).fit(fit_data)
        means.append(np.expand_dims(np.sort(np.ndarray.flatten(gm.means_)),axis=-1))
        covariances.append(gm.covariances_)
        weights.append(gm.weights_)

    return gm, means,covariances,weights


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

def create_gif(tomo,mask,save_path,fps,cmap='inferno',vmin=None,vmax=None):
    frames = []
    tomo -= np.amin(tomo)
    tomo = 255 * tomo / np.amax(tomo)
    tomo = np.array(tomo, dtype=np.uint8)
    for _ in tqdm(range(len(tomo))):
        slice = tomo[_]
        slice = cv2.equalizeHist(slice)
        slice = cv2.cvtColor(slice, cv2.COLOR_GRAY2RGB)


        fig = plt.figure()
        plt.imshow(slice)
        if (vmin is not None) and (vmax is not None):
            plt.imshow(mask[_],alpha=0.5,cmap=cmap,vmin=vmin,vmax=vmax)
        else:
            plt.imshow(mask[_],alpha=0.5,cmap=cmap)
        plt.axis('off')

        buf = BytesIO()
        plt.savefig(buf, format='png',  bbox_inches='tight', pad_inches=0, transparent=True)
        plt.close(fig)  # Close the figure to free memory
        buf.seek(0)

        # Read the image from the buffer and add it to the frames list
        frames.append(imageio.imread(buf))

    imageio.mimsave(save_path, frames, fps=fps)