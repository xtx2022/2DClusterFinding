from basic_utilities import *
import cv2
import math

class judge_data:
    def __init__(self) -> None:
        self.dist = []
        self.rev_dist = []
        self.large_error = []
        self.large_error_points = []
        self.not_found = []
        self.not_found_points = []
        self.dist_array = np.array([])
    
    def print_data(self) -> None:
        # Testing code
        print(self.dist)  
        print(self.rev_dist)
        print(self.large_error)
        print(self.large_error_points)
        print(self.not_found)
        print(self.not_found_points)
        print(self.dist_array)
    
    def plot_histogram(self) -> None:
        # Plotting the histogram
        plt.hist(self.dist_array[0], bins=50, edgecolor='black')  # Adjust bins as needed
        plt.title('Histogram of Delta x')
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.show()

        plt.hist(self.dist_array[1], bins=50, edgecolor='black')  # Adjust bins as needed
        plt.title('Histogram of Delta y')
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.show()

        plt.hist(self.dist_array[2], bins=50, edgecolor='black')  # Adjust bins as needed
        plt.title('Histogram of Delta r')
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.show()
    
    def show_error(self, _images_: list, _found_: list, max_image_num: int = -1) -> None:
        if (len(self.large_error) == 0 and len(self.not_found) == 0):
            print('None')
        
        if max_image_num >= 0:
            image_num1 = min(len(self.large_error), max_image_num)
            image_num2 = min(len(self.not_found), max_image_num)
        else:
            image_num1 = len(self.large_error)
            image_num2 = len(self.not_found)
                             

        for i in range (image_num1):
            print('Image ' + str(self.large_error[i]) + ' Raw Data')
            print(extract_subarr(_images_[self.large_error[i]].raw_image, self.large_error_points[i]))
            marked_imag = mark_point(_images_[self.large_error[i]].raw_image, [tuple(np.round(row - 0.5).astype(int)) for row in _found_[self.large_error[i]]])
            marked_imag = mark_point(marked_imag, [tuple(row) for row in np.round(_images_[self.large_error[i]].cord_of_points - 0.5).astype(int)], (255, 0, 0))

            plt.imshow(marked_imag)
            plt.title('Image ' + str(self.large_error[i]) + ' Visualization')
            plt.xlabel('X-axis')
            plt.ylabel('Y-axis')
            plt.show()
        
        for i in range (image_num2):
            print('Image ' + str(self.not_found[i]) + ' Raw Data')
            print('Not Found:', self.not_found_points[i])
            print(extract_subarr(_images_[self.not_found[i]].raw_image, self.not_found_points[i]))
            marked_imag = mark_point(_images_[self.not_found[i]].raw_image, [tuple(row) for row in np.round(_images_[self.not_found[i]].cord_of_points - 0.5).astype(int)], (255, 0, 0))
            marked_imag = mark_point(marked_imag, [tuple(np.round(row - 0.5).astype(int)) for row in _found_[self.not_found[i]]])

            plt.imshow(marked_imag)
            plt.title('Image ' + str(self.not_found[i]) + ' Visualization')
            plt.xlabel('X-axis')
            plt.ylabel('Y-axis')
            plt.show()

def distance(p, q):
    # Function to calculate Euclidean distance between points p and q
    return math.sqrt((q[0] - p[0])**2 + (q[1] - p[1])**2)

def smallest_distance_to_set(point, point_set):
    # point is a tuple (x, y) representing the point P
    # point_set is a list of tuples [(x1, y1), (x2, y2), ...] representing the set S
    
    if not point_set:
        return float('inf')  # If point_set is empty, return infinity
    
    x_distance = float('inf')
    y_distance = float('inf')
    min_distance = float('inf')
    
    for q in point_set:
        dist = distance(point, q)
        if dist < min_distance:
            min_distance = dist
            x_distance = point[0] - q[0]
            y_distance = point[1] - q[1]
    
    return [x_distance, y_distance, min_distance]

# Define the color (B, G, R) and thickness of the cross marks
# color = (0, 255, 0)  # Green color
# cross_length = 10  # Length of the cross arms
# thickness = 2  # Thickness of the lines
def mark_point(image: np.array, points: list, color = (0, 255, 0), type = 'normal', cross_length = 2, thickness = 1) -> np.array:
    # Ensure the image is a NumPy array with the correct dtype
    if image.dtype != np.uint8:
        marked_image = (image / np.max(image) * 255).astype(np.uint8)
    else:
        marked_image = np.copy(image)
    if len(marked_image.shape) == 2 or (len(marked_image.shape) == 3 and marked_image.shape[2] == 1):
        marked_image = cv2.applyColorMap(marked_image, cv2.COLORMAP_TWILIGHT_SHIFTED)
    if type == 'normal':
        for (x, y) in points:
            cv2.line(marked_image, (x - cross_length, y), (x + cross_length, y), color, thickness)
            cv2.line(marked_image, (x, y - cross_length), (x, y + cross_length), color, thickness)
    elif type == 'skew':
        for (x, y) in points:
            cv2.line(marked_image, (x - cross_length, y - cross_length), (x + cross_length, y + cross_length), color, thickness)
            cv2.line(marked_image, (x + cross_length, y - cross_length), (x - cross_length, y + cross_length), color, thickness)
    # Display the image with marked points
    return marked_image

def judge(_images_: list, _found_: list, threshold_error: float = 1.5) -> judge_data:
    _data_ = judge_data()
    cycles = min(len(_images_), len(_found_))

    for i in range (cycles):
        for j in range(len(_found_[i])):
            _data_.dist.append(smallest_distance_to_set(_found_[i][j], [tuple(row) for row in _images_[i].cord_of_points]))
            if _data_.dist[-1][2] > threshold_error:
                _data_.large_error.append(i)
                _data_.large_error_points.append(tuple(round(num - 0.5) for num in _found_[i][j]))
        for j in range(len(_images_[i].cord_of_points)):
            _data_.rev_dist.append(smallest_distance_to_set(_images_[i].cord_of_points[j], _found_[i]))
            if _data_.rev_dist[-1][2] > threshold_error:
                _data_.not_found.append(i)
                _data_.not_found_points.append(tuple(np.round(_images_[i].cord_of_points[j] - 0.5).astype(int)))

    _data_.dist_array = np.array(_data_.dist).T 

    return _data_

def compare_results(results: list, names: list = []) -> None:
    if len(names) == 0:
        names = [''] * len(results)

    for i in range(len(results)):
        large_error = len(results[i].large_error)
        not_found = len(results[i].not_found)
        print(names[i], 'Large Error:', large_error)
        print(names[i], 'Not Found:', not_found)
        true_total = int(sum(len(arr) for arr in results[i].rev_dist) / 3) # x, y, r
        total_found = int(sum(len(arr) for arr in results[i].dist) / 3)
        print(names[i], 'Actual Number of Clusters:', true_total)
        print(names[i], 'Number of Found Clusters:', total_found)
        print(names[i], 'Precision:', 1.0 - large_error / total_found)
        print(names[i], 'Recall:', 1.0 - not_found / true_total)
                                        
    # Plot the histograms with specified border colors and transparent fills
    for i in range(len(results)):
        plt.hist(results[i].dist_array[0], bins=100, density=True, alpha=0.5, label=names[i])  # Histogram for data
    # Customize the plot
    plt.title('Histogram of Delta x')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.legend(loc='upper right')  # Add a legend to differentiate the datasets
    plt.show()

    # Plot the histograms
    for i in range(len(results)):
        plt.hist(results[i].dist_array[1], bins=100, density=True, alpha=0.5, label=names[i])  # Histogram for data
    # Customize the plot
    plt.title('Histogram of Delta y')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.legend(loc='upper right')  # Add a legend to differentiate the datasets
    plt.show()

    # Plot the histograms
    for i in range(len(results)):
        plt.hist(results[i].dist_array[2], bins=100, density=True, alpha=0.5, label=names[i])  # Histogram for data
    # Customize the plot
    plt.title('Histogram of Delta r')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.legend(loc='upper right')  # Add a legend to differentiate the datasets
    plt.show()