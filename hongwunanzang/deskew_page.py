# coding: utf-8
import skimage.io as io
from skimage.filters import threshold_otsu, threshold_adaptive

from skimage.transform import rotate
import numpy as np
import sys
import math

def binarisation(src_image):
    if len(src_image.shape) == 3:
        image = (src_image.sum(axis=2) / 3).astype('ubyte')
    else:
        image = src_image
    thresh = threshold_otsu(image)
    binary = (image > thresh).astype('ubyte')
    binary1 = 1 - binary
    im = 255 - np.multiply(255 - image, binary1)
    block_size = 35
    binary = threshold_adaptive(image, block_size, offset=20)
    binary = binary.astype('ubyte')
    return binary

# 100
def deskew_page(image, binary=None):
    if binary is None:
        binary = binarisation(image)
        binary = (1 - binary).astype('ubyte')
    image_height, image_width = binary.shape
    y0 = image_height/2
    max_angle = 0
    max_pixel_sum = 0
    for angle in np.arange(1.5, -1.6, -0.1):
        accumulated_pixels_lst = []
        k = math.tan(angle * math.pi / 180.0)
        for x0 in range(0, 101):
            accumulated_pixels = 0
            for y in range(750, 1201):
                x = x0 + int((y-y0)*k)
                if x >= 0:
                    accumulated_pixels = accumulated_pixels + binary[y,x]
            accumulated_pixels_lst.append(accumulated_pixels)
        first_derivation = np.diff(accumulated_pixels_lst)
        max_accumulated_pixels = np.max(first_derivation)
        max_pos = np.argmax(first_derivation)
        # print angle, max_accumulated_pixels, max_pos
        if max_accumulated_pixels > max_pixel_sum:
            max_angle = angle
            max_pixel_sum = max_accumulated_pixels
    return -max_angle

if __name__ == '__main__':
    image = io.imread(sys.argv[1], 0)
    angle = deskew_page(image)
    if angle != 0.0:
        image_new = rotate(image, angle, mode='edge')
        page_image = (image_new * 255).astype('ubyte')
        io.imsave(sys.argv[2], page_image)



