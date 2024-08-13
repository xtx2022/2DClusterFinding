import torch
from torchsr.models import edsr
import torch.nn.functional as F
import numpy as np

def min_max_normalize(tensor, new_min=0.0, new_max=1.0):
    """
    Perform min-max normalization on a tensor.
    
    Parameters:
        tensor (torch.Tensor): The input tensor to normalize.
        new_min (float): The new minimum value of the normalized tensor.
        new_max (float): The new maximum value of the normalized tensor.
    
    Returns:
        torch.Tensor: The normalized tensor.
    """
    # Ensure tensor is of type float for normalization
    tensor = tensor.float()
    # Calculate min and max values
    tensor_min = tensor.min()
    tensor_max = tensor.max()
    # Avoid division by zero
    if tensor_max == tensor_min:
        return torch.full_like(tensor, new_min)
    # Apply min-max normalization formula
    normalized_tensor = (tensor - tensor_min) / (tensor_max - tensor_min)
    # Scale to the desired range [new_min, new_max]
    normalized_tensor = normalized_tensor * (new_max - new_min) + new_min
    return normalized_tensor

def get_high_resolution_image(_images_: list, _scale_: int = 2, zero_padding: int = 2, num_of_images: int = -1):
    '''
    Input images: A list of low resolution images in np.arrays 
    Output images: A list of high resolution images in np.arrays 
    Model: Pretrained EDSR model with zero padding
    '''
    if num_of_images < 0:
        num_of_images = len(_images_)
    hr_model = edsr(scale=_scale_, pretrained=True)
    high_resolution_images = []
    for i in range(num_of_images):
        print('Processing image', i , '...')
        min_val = np.min(_images_[i])
        max_val = np.max(_images_[i])
        lr_t = min_max_normalize(torch.from_numpy(_images_[i]).unsqueeze(0).float()).repeat(3, 1, 1).unsqueeze(0)
        sr_t = hr_model(F.pad(lr_t, (zero_padding, zero_padding, zero_padding, zero_padding), mode='constant', value=0.0))
        margin = _scale_ * zero_padding
        sr = torch.mean(min_max_normalize(sr_t[:,:,margin:-margin,margin:-margin].squeeze(0), min_val, max_val), dim = 0).detach().cpu().numpy()
        high_resolution_images.append(sr)
    return high_resolution_images