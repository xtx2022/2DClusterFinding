from basic_utilities import *
from edsr_model import *

def frac_part(array: np.array, center, half_size = 2):
    sub_arr = extract_subarr(array, center, half_size).T
    total_intensity = np.sum(sub_arr)
    result = np.array([- half_size + 0.5, - half_size + 0.5])
    if total_intensity == 0:
        return None
    for idx, x in np.ndenumerate(sub_arr):
        result += np.array(idx) * (x / total_intensity)
    return result

def find_local_maxima(matrix: np.array, threshold: int = 0):
    local_maxima = []
    for (i, j), value in np.ndenumerate(matrix):
        is_local_maxima = True
        if value < threshold: is_local_maxima = False
        # Check all eight possible neighbors
        neighbors = [
            (i-1, j-1), (i-1, j), (i-1, j+1),
            (i, j-1),             (i, j+1),
            (i+1, j-1), (i+1, j), (i+1, j+1)
        ]
        for u, v in neighbors:
            if 0 <= u < len(matrix) and 0 <= v < len(matrix[0]) and matrix[u][v] > value: 
                is_local_maxima = False
                break
        if is_local_maxima: local_maxima.append(np.array([j, i])) 
    return local_maxima

def remove_isolated_pixels(image: np.array, threshold: int = 0):
    # Make a copy of the image to modify
    processed_image = image.copy()
    for (i, j), value in np.ndenumerate(image):
        if value <= threshold: break
        to_delete = True
        neighbors = [
            (i-1, j-1), (i-1, j), (i-1, j+1),
            (i, j-1),             (i, j+1),
            (i+1, j-1), (i+1, j), (i+1, j+1)
        ]
        for u, v in neighbors:
            if 0 <= u < image.shape[0] and 0 <= v < image.shape[1] and image[u][v] > threshold: 
                to_delete = False
                break
        if to_delete: processed_image[i, j] = threshold
    return processed_image
    
def get_float_result(_images_: list, _found_: list, half_size = 2, to_int: bool = False) -> list:
    if to_int:
        _float_result_ = [[0 for _ in range(len(row))] for row in _found_]
        for i in range(len(_found_)):
            for j in range(len(_found_[i])):
                _float_result_[i][j] = np.round(_found_[i][j] + frac_part(_images_[i], _found_[i][j], half_size) - 0.5).astype(int)
    else:
        _float_result_ = [[0.0 for _ in range(len(row))] for row in _found_]
        for i in range(len(_found_)):
            for j in range(len(_found_[i])):
                _float_result_[i][j] = _found_[i][j] + frac_part(_images_[i], _found_[i][j], half_size)
    return _float_result_

def find_points(_images_: list, algorithm: str, half_size: int = 2, _threshold_: int = 1000, num_of_images: int = -1) -> list: 
    '''
    algorithms: 'local maxima', 'local maxima denoised', 'local maxima denoised double', 
    'hr local maxima denoised', 'hr local maxima denoised double'
    '''
    if num_of_images < 0:
        num_of_images = len(_images_)
    _found_ = []
    if algorithm == 'local maxima':
        for i in range (num_of_images):
            _found_.append(find_local_maxima(_images_[i].raw_image, _threshold_))
        _found_ = get_float_result([_image_.raw_image for _image_ in _images_], _found_, half_size)
    if algorithm == 'local maxima denoised':
        _denoised_ = [remove_isolated_pixels(_image_.raw_image) for _image_ in _images_]
        for i in range (num_of_images):
            _found_.append(find_local_maxima(_denoised_[i], _threshold_))
        _found_ = get_float_result(_denoised_, _found_, half_size)
    if algorithm == 'local maxima denoised double':
        _denoised_ = [remove_isolated_pixels(_image_.raw_image) for _image_ in _images_]
        for i in range (num_of_images):
            _found_.append(find_local_maxima(_denoised_[i], _threshold_))
        _mid_result_ = get_float_result(_denoised_, _found_, half_size, to_int=True)    
        _found_ = get_float_result(_denoised_, _mid_result_, half_size)
    if algorithm == 'hr local maxima denoised':
        _denoised_ = [remove_isolated_pixels(_image_.raw_image) for _image_ in _images_]
        _hr_images_ = get_high_resolution_image(_denoised_, 4, 2, num_of_images)
        for i in range (len(_hr_images_)):
            _found_.append(find_local_maxima(_hr_images_[i], _threshold_))
        _found_ = get_float_result(_hr_images_, _found_, 4 * half_size)
        _found_ = [[value / 4 for value in sublist] for sublist in _found_]
    if algorithm == 'hr local maxima denoised double':
        _denoised_ = [remove_isolated_pixels(_image_.raw_image) for _image_ in _images_]
        _hr_images_ = get_high_resolution_image(_denoised_, 4, 2, num_of_images)
        for i in range (len(_hr_images_)):
            _found_.append(find_local_maxima(_hr_images_[i], _threshold_))
        _mid_result_ = get_float_result(_hr_images_, _found_, half_size, to_int=True) 
        _found_ = get_float_result(_hr_images_, _mid_result_, 2 * half_size)
        _found_ = [[value / 4 for value in sublist] for sublist in _found_]
    return _found_