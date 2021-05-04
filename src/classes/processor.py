import cv2
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from math import ceil, floor


class ImageProcessor():

    def __init__(self, image, threshold):
        if image is None: 
            raise Exception("Image not Found")

        ret, image_binary = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
        width, height = np.shape(image)

        # Private Variables
        self._image = image
        self._width = width
        self._height = height
        self._binary_image = image_binary
        self._modified_image = image_binary
        self._groups = []

    def erode_image(self, kernel):
        self._morf_image(True, kernel)

    def dilate_image(self, kernel):
        self._morf_image(False, kernel)

    def plot_image(self):
        fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2)

        # Plotando imagem original
        ax1.set_title('Original Image')
        ax1.imshow(self._image, 'gray')
        ax1.axis(False)

        ax2.set_title('Histogram')
        ax2.plot(self._calculate_histogram(self._image))

        # Plotando imagem binarizada
        ax3.set_title('Binary Image')
        ax3.imshow(self._binary_image, 'gray')
        ax3.axis(False)

        ax4.set_title('Histogram')
        ax4.plot(self._calculate_histogram(self._binary_image))

        # Plotando results
        ax5.set_title('Result')
        img_plot = cv2.cvtColor(self._image, cv2.COLOR_BGR2RGB)
        for g in self._groups: cv2.rectangle(img_plot, g[0], g[1], (0, 255, 0), 1)
        ax5.imshow(img_plot, 'gray')
        ax5.axis(False)

        ax6.set_title('Histogram')
        ax6.plot(self._calculate_histogram(self._image))

        plt.show()

    def segmentate(self, kernel):
        mask = np.zeros((self._width, self._height), dtype=np.uint8)
        groups = []
        for i in range(0, self._width):
            for j in range(0, self._height):
                if mask[i, j] == 0 and self._modified_image[i, j] == 0:
                    mask[i, j] = 1
                    neigbhood = self._get_neigbhood(i, j, kernel, include_itself=False)
                    current_group = ((j, i), (j, i))
                    while len(neigbhood) > 0:
                        (curr_x, curr_y) = neigbhood.pop()
                        if mask[curr_x, curr_y] == 0:
                            mask[curr_x, curr_y] = 1
                            x1 = current_group[0][1]
                            y1 = current_group[0][0]
                            x2 = current_group[1][1]
                            y2 = current_group[1][0]

                            x1 = min(x1, curr_x)
                            y1 = min(y1, curr_y)
                            x2 = max(x2, curr_x)
                            y2 = max(y2, curr_y)

                            current_group = ((y1, x1), (y2, x2))
                            neigbhood += self._get_neigbhood(curr_x, curr_y, kernel, include_itself=False)
                        
                    groups.append(current_group)

        self._groups = groups
                        
    def _morf_image(self, erode, kernel):
        kernel_w,kernel_h = np.shape(kernel)
        kernel_count = np.count_nonzero(kernel)

        # Validating Kernel
        if kernel_w % 2 == 0: 
            raise Exception("Invalid Kernel Size: width can't be multiple of 2")
        if kernel_h % 2 == 0: 
            raise Exception("Invalid Kernel Size: height can't be multiple of 2")
        if kernel_w != kernel_h: 
            raise Exception("Invalid Kernel Size: not a square matrix")

        imgbuff = np.zeros((self._width, self._height), dtype=np.uint8)

        for x in range(0, self._width):
            for y in range(0, self._height):
                neigbhood = self._get_neigbhood(x, y, kernel)
                hit_count = len(neigbhood)

                if erode:
                    # Is Fit
                    if hit_count == kernel_count:
                        imgbuff[x,y] = 0
                    else:
                        imgbuff[x,y] = 255
                else:
                    # Is Fit or Hit
                    if hit_count > 0:
                        imgbuff[x,y] = 0
                    else:
                        imgbuff[x,y] = 255
        
        self._modified_image = imgbuff.copy()

    def _get_neigbhood(self, x, y, kernel, include_itself=True):
        kernel_w, kernel_h = np.shape(kernel)

        # Translating x,y
        new_x = x - floor(kernel_w / 2)
        new_y = y - floor(kernel_h / 2)

        neigbhoods = []
        for i in range(0, kernel_w):
            for j in range(0, kernel_h):
                rel_x = new_x + i
                rel_y = new_y + j 
                if inRange(rel_x, 0, self._width) and inRange(rel_y, 0, self._height):
                    if kernel[i][j] != 0 and self._modified_image[rel_x, rel_y] == 0:
                        if include_itself or (x != rel_x or y != rel_y):
                            neigbhoods.append((rel_x, rel_y))

        return neigbhoods

    def _calculate_histogram(self, image):
        hist = np.zeros(256, dtype=np.uint8)
        for i in range(0, self._width):
            for j in range(0, self._height):
                col = image[i, j]
                hist[col] = hist[col] + 1
        
        return hist
    

def inRange(value, _min, _max):
    return value >= _min and value < _max
