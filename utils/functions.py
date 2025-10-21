import numpy as np
from sklearn.mixture import GaussianMixture
import torch

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