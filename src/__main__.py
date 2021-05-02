from src.classes.processor import ImageProcessor
from os import path
from cv2 import imread
from numpy import array

def main():
    image = imread("input.png", 0)

    kernel = [[0,1,0],
              [1,1,1],
              [0,1,0]]

    processor = ImageProcessor(image, 175)
    processor.erode_image(kernel)
    processor.segmentate()
    processor.plot_image()

if __name__ == "__main__":
    main()