# coding: utf-8
import numpy as np

from skimage import io
from skimage.filters import threshold_otsu, threshold_adaptive
from skimage.segmentation import clear_border
from skimage.measure import label
from skimage.morphology import rectangle, binary_dilation
from skimage.measure import regionprops
import sys

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

def find_bar_rectangle(src_image, save=False):
    image = io.imread(src_image, 0)
    bw = binarisation(image)
    image_height, image_width = bw.shape
    image = image[:, 50:image_width-50]
    bw = bw[:, 50:image_width-50]
    image_height, image_width = bw.shape
    bw = (1 - bw).astype('ubyte')
    bw = clear_border(bw)

    selem = rectangle(20, 65)
    bw = binary_dilation(bw, selem)

    label_image = label(bw, connectivity=2)

    regions = []
    for region in regionprops(label_image):
        minr, minc, maxr, maxc = region.bbox
        if (maxr - minr) < 400:
            continue
        cc = image_height / 2
        if maxc < image_width/2 or minc > image_width/2:
            pass
        else:
            if (maxc - minc) < 250: # 将显示卷号的页面跳过
                continue
        if (maxr > cc and minr < cc):
            distance_diff = abs(abs(maxr - cc) - abs(minr - cc))
            if distance_diff < 100:
                continue
        print minr, minc, maxr, maxc
        regions.append( (minr, minc, maxr, maxc) )
    merged_regions = []
    count = len(regions)
    print count
    minr = 0
    minc = 0
    maxr = 0
    maxc = 0
    for i in range(count):
        if i == 0:
            minr, minc, maxr, maxc = regions[i]
        else:
            if abs(regions[i][0] - minr) < 100:
                minr = min(regions[i][0], minr)
                minc = min(regions[i][1], minc)
                maxr = max(regions[i][2], maxr)
                maxc = max(regions[i][3], maxc)
            else:
                merged_regions.append( (minr, minc, maxr, maxc) )
                minr, minc, maxr, maxc = regions[i]
    if maxr != 0:
        merged_regions.append((minr, minc, maxr, maxc))

    # if count == 2 or count == 4:
    #     for i in range(count/2):
    #         minr = min(regions[i*2][0], regions[i*2+1][0])
    #         minc = min(regions[i*2][1], regions[i*2+1][1])
    #         maxr = max(regions[i * 2][2], regions[i * 2 + 1][2])
    #         maxc = max(regions[i * 2][3], regions[i * 2 + 1][3])
    #         merged_regions.append((minr, minc, maxr, maxc))
    if len(merged_regions) != 2:
        print '%s has %d regions.' % (src_image, count)

    if save:
        for i in range(len(merged_regions)):
            minr, minc, maxr, maxc = merged_regions[i]
            region_image = image[minr:maxr+1, minc:maxc+1]
            print minc, minr, maxc+1, maxr+1
            dstname = src_image.replace('.', '_%s.' % (i+1))
            io.imsave(dstname, region_image)

    return regions

if __name__ == '__main__':
    find_bar_rectangle(sys.argv[1], True)
