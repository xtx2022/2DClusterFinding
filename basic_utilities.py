import numpy as np
import matplotlib.pyplot as plt

class image_unit:
    def __init__(self) -> None:
        self.num_of_points = 0
        self.cord_of_points = np.array([])
        self.raw_image = np.array([])
    def print_data(self) -> None:
        print(self.num_of_points)
        print(self.cord_of_points)
        print(self.raw_image)
        # Plotting the array as a grayscale image
        plt.imshow(self.raw_image, cmap='viridis')
        plt.colorbar()  # Adding a colorbar to show intensity scale
        plt.title('Visualization')
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        plt.show()

# For every image:
# Line 1: number of points(10)
# Line 2-11: exact positions of the points
# Others: grayscale image
def read_in_data(file_path:str) -> list:
    with open(file_path, 'r') as file:
        _images_ = []
        cop = [] # Store cordinates of points tempoarily
        ri = [] # Store raw image tempoarily
        for line in file:
            numbers = []
            numbers = [float(s) for s in line.strip().split()]
            if len(numbers) == 1:
                if len(_images_) > 0:
                    _images_[-1].raw_image = np.array(ri).T
                    ri.clear()
                _images_.append(image_unit())
                _images_[-1].num_of_points = int(numbers[0])
            elif len(numbers) == 2:
                cop.append(numbers)
                if len(cop) == _images_[-1].num_of_points:
                    _images_[-1].cord_of_points = np.array(cop)
                    _images_[-1].cord_of_points = _images_[-1].cord_of_points
                    cop.clear()
            else:
                ri.append(numbers)
        _images_[-1].raw_image = np.array(ri).T
        return _images_

def extract_subarr(array: np.array, center, half_size = 2): # A 5x5 array by default, half_size is 2
    """
    Extract a subarray centered around the specified data point (center_x, center_y).

    Parameters:
    array (numpy.ndarray): The input array from which to extract the subarray.
    center = (center_x, center_y)
        center_x (int): The x-coordinate (row) of the center data point.
        center_y (int): The y-coordinate (column) of the center data point.

    Returns:
    numpy.ndarray: The extracted subarray, Completing zeros if the center point is too close to the border.

    """
    (center_x, center_y) = center
    if (center_x < 0 or center_x >= array.shape[0] or center_y < 0 or center_y >= array.shape[1]):
        return None
    left = max(center_x - half_size, 0)
    left_pad = max(half_size - center_x, 0)
    right = min(center_x + half_size + 1, array.shape[0])
    right_pad = max(center_x + half_size + 1 - array.shape[0], 0)
    down = max(center_y - half_size, 0)
    down_pad = max(half_size - center_y, 0)
    up = min(center_y + half_size + 1, array.shape[1])
    up_pad = max(center_y + half_size + 1 - array.shape[1], 0)
    # Extract the subarray centered around (center_x, center_y)
    subarray = array[down : up,
                     left : right]
    padded_arr = np.pad(subarray, ((down_pad, up_pad), (left_pad, right_pad)), mode='constant', constant_values=0)
    return padded_arr