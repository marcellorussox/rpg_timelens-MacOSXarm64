import numpy as np
from scipy.ndimage import morphology

import torch as th
from torchvision import transforms


def find_channels_mean_and_std(image):
    mean_list = []
    std_list = []
    for channel_idx in range(image.size(1)):
        mean = image[:, channel_idx, ...].mean()
        std = image[:, channel_idx, ...].std() + 1e-5
        std_list.append(std)
        mean_list.append(mean)
    return mean_list, std_list


def normalize_image(image, mean_list, std_list):
    output_image = image.clone()
    for channel_idx in range(image.size(1)):
        output_image[:, channel_idx, ...] -= mean_list[channel_idx]
        output_image[:, channel_idx, ...] /= std_list[channel_idx]
    return output_image


def denormalize_image(image, mean_list, std_list):
    output_image = image.clone()
    for channel_idx in range(image.size(1)):
        output_image[:, channel_idx, ...] *= std_list[channel_idx]
        output_image[:, channel_idx, ...] += mean_list[channel_idx]
    return output_image


def pil_image_to_tensor(pil_image):
    """Returns 3 x h x w float tensor given h x w x 3 uint8 PIL image.
    
    Note, that the function scales range from [0, 255] to [0.0, 1.0].
    """
    return transforms.ToTensor()(pil_image)


def tensor_to_pil_image(tensor):
    """Returns h x w x 3 uint8 PIL image given 3 x h x w float tensor.
    
    Note, that the function scales range from [0.0, 1.0] to [0, 255].
    """
    return transforms.ToPILImage()(tensor)


def unsqueeze_front_n(tensor, n):
    """Adds singletone dimensions in the front."""
    return tensor[(None,) * n]


def unsqueeze_back_n(tensor, n):
    """Adds singletone dimensions in the back."""
    return tensor[(...,) + (None,) * n]


def create_meshgrid(width, height):
    x, y = th.meshgrid([th.arange(0, width), th.arange(0, height)])
    x, y = (x.transpose(0, 1).float(), y.transpose(0, 1).float())
    return x, y


def dilate(image, window_size):
    """Returns dilated (expanded) image.
    Args:
        image: is a boolen tensor of size (height x width) image
               that will be dilated.
        window_size: size of the window that will be used for the
                     dilation.
    """
    dilation_window = np.ones((window_size, window_size), dtype=bool)
    dilated_image = th.from_numpy(
        morphology.binary_dilation(image.cpu().numpy(), dilation_window).astype(
            np.uint8
        )
    )
    return dilated_image
