from src.classes.processor import ImageProcessor
from os import path
from cv2 import imread
import numpy as np

def main():
    image = imread("input.png", 0)

    kernel1 =[[0,1,0],
              [0,1,0],
              [0,1,0]]

    kernel2 = np.ones((5, 5), dtype=np.uint8)

    processor = ImageProcessor(image, 175)

    processor.dilate_image(kernel1)
    processor.segmentate([[0,1,0],[1,1,1],[0,1,0]])
    processor.plot_image()

    processor.dilate_image(kernel2)
    processor.dilate_image(kernel2)
    processor.segmentate([[0,1,0],[1,1,1],[0,1,0]])
    processor.plot_image()



if __name__ == "__main__":
    main()