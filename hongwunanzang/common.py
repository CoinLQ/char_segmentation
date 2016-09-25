# coding: utf-8
import numpy as np
from skimage.filters import threshold_otsu, threshold_adaptive

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
    binary = threshold_adaptive(im, block_size, offset=20)
    binary = binary.astype('ubyte')
    return binary

def binarisation1(src_image):
    if len(src_image.shape) == 3:
        image = (src_image.sum(axis=2) / 3).astype('ubyte')
    else:
        image = src_image
    block_size = 35
    binary = threshold_adaptive(image, block_size, offset=20)
    binary = binary.astype('ubyte')
    return binary

def centroid(arr):
    length = arr.shape[0]
    moment = 0
    mass = 0
    for i in range(length):
        mass = mass + arr[i]
        moment = moment + arr[i] * i
    return (moment / mass)

def geometric_center(arr):
    length = arr.shape[0]
    s = 0
    e = 0
    for i in range(length):
        if arr[i] > 0:
            s = i
            break
    for i in range(length-1, -1, -1):
        if arr[i] > 0:
            e = i
            break
    return (s + e) / 2