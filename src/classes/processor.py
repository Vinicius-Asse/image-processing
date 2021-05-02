import cv2
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from math import ceil


class ImageProcessor():

    def __init__(self, image, threshold):
        if (image is None): raise Exception("Image not Found")

        ret, image_binary = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
        width,height = np.shape(image)

        # Private Variables
        self._image = image
        self._width = width
        self._height = height
        self._binary_image = image_binary
        self._modified_image = image_binary

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

        # Plotando imagem modificada
        ax5.set_title('Modified Image')
        img_plot = cv2.cvtColor(self._modified_image, cv2.COLOR_BGR2RGB)
        ax5.imshow(img_plot, 'gray')
        ax5.axis(False)

        ax6.set_title('Histogram')
        ax6.plot(self._calculate_histogram(self._modified_image))

        plt.show()

    def segmentate(self):
        mask = np.zeros((self._width, self._height), dtype=np.uint8)
        groups = []
        for i in range(0, self._width):
            for j in range(0, self._height):
                new_group = self._expand_neigbhood(i, j, mask)
                if new_group is not None: groups.append(new_group)

        print (groups)
    

    def _expand_neigbhood(self, x, y, mask, group=[]):
        if x > 0 and x < self._width:
            if y > 0 and y < self._height:
                is_open = mask[x, y] != 0
                mask[x, y] = 1
                if is_open:
                    if self._modified_image[x, y] == 0:
                        return (x, y),
                        






    def _morf_image(self, erode, kernel):
        kernel_w,kernel_h = np.shape(kernel)

        # Validating Kernel
        if kernel_w % 2 == 0: raise Exception("Invalid Kernel Size: width can't be multiple of 2")
        if kernel_h % 2 == 0: raise Exception("Invalid Kernel Size: height can't be multiple of 2")
        if kernel_w != kernel_h: raise Exception("Invalid Kernel Size: not a square matrix")

        width,height = np.shape(self._binary_image)

        imgbuff = np.zeros((width, height), dtype=np.uint8)

        for x in range(0, width):
            for y in range(0, height):
                fit = self._check_fit(x, y, self._binary_image, kernel)

                if erode:
                    # Is Fit
                    if fit == 1:
                        imgbuff[x,y] = 0
                    # Is Hit
                    elif fit == 0:
                        imgbuff[x,y] = 255
                    # Is Miss
                    elif fit == -1:
                        imgbuff[x,y] = 255
                else:
                    # Is Fit
                    if fit == 1:
                        imgbuff[x,y] = 0
                    # Is Hit
                    elif fit == 0:
                        imgbuff[x,y] = 0
                    # Is Miss
                    elif fit == -1:
                        imgbuff[x,y] = 255
        
        self._modified_image = imgbuff.copy()

    def _check_fit(self, x, y, image, kernel):

        image_w,image_h = np.shape(image)
        kernel_w,kernel_h = np.shape(kernel)

        # Translating x,y
        new_x = x - ceil(kernel_w / 2)
        new_y = y - ceil(kernel_h / 2)

        has_miss = False
        has_hit = False
        for i in range(0, kernel_w):
            for j in range(0, kernel_h):
                img_x = new_x + i
                img_y = new_y + j
                if img_x < 0 or img_x > image_w or img_y < 0 or img_y > image_h:
                    has_miss = True
                elif kernel[i][j] == 1:
                    if image[img_x, img_y] == 0:
                        has_hit = True
                    else:
                        has_miss = True

                if has_miss and has_hit:
                    return 0 # RETURNING HIT

        if has_hit == False:
            return -1 # RETURNING MISS
        elif has_miss == False:
            return 1 # RETURNING FIT

    def _calculate_histogram(self, image):
        width,height = np.shape(image)
        hist = np.zeros(256, dtype=np.uint8)
        for i in range(0, width):
            for j in range(0, height):
                col = image[i, j]
                hist[col] = hist[col] + 1
        
        return hist
