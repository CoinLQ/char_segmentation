# coding: utf-8
import numpy as np

from skimage import io
from skimage.filters import threshold_otsu, threshold_adaptive
from skimage.segmentation import clear_border
from skimage.measure import label
from skimage.measure import regionprops
from skimage.transform import rotate
import sys
import subprocess
from operator import itemgetter

from deskew_page import deskew_page

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

def split_page(src_image, min_page_no):
    image = io.imread(src_image, 0)
    bw = binarisation(image)
    image_height, image_width = bw.shape
    bw = (1 - bw).astype('ubyte')
    bw = clear_border(bw)
    label_image = label(bw, connectivity=2)
    #image_label_overlay = label2rgb(label_image, image=image)

    minr = 0
    minc = 0
    maxr = 0
    maxc = 0
    regions = []
    vol_regions = []
    #fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
    for region in regionprops(label_image):
        minr, minc, maxr, maxc = region.bbox
        w = maxc - minc
        h = maxr - minr
        if h > 1000 and h < 1700 and w > 100 and w < 300:
            #print minr, minc, maxr, maxc
            vol_regions.append( (minr, minc, maxr, maxc) )
        if minc < 200 and maxc > (image_width-200):
            continue
        if (maxr - minr) < 1300:
            continue
        if (maxc - minc) < 800:
            continue
        regions.append( (minr, minc, maxr, maxc))
        print minr, minc, maxr, maxc
        #rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
        #                          fill=False, edgecolor='red', linewidth=1)
        #ax.add_patch(rect)
    count = len(regions)
    regions.sort(key=itemgetter(1))
    #print 'count: ', count
    # if count >= 2: # 将距离很近的相邻两个框合并
    #     pop_index = []
    #     for i in range(count - 1):
    #         if (abs(regions[i + 1][1] - regions[i][1]) < 120) or (abs(regions[i + 1][3] - regions[i][3]) < 120):
    #             pop_index.append(i)
    #             minr = min(regions[i][0], regions[i + 1][0])
    #             minc = min(regions[i][1], regions[i + 1][1])
    #             maxr = max(regions[i][2], regions[i + 1][2])
    #             maxc = min(regions[i][3], regions[i + 1][3])
    #             regions[i + 1] = (minr, minc, maxr, maxc)
    #     if pop_index:
    #         new_regions = []
    #         for i in range(count):
    #             if i not in pop_index:
    #                 new_regions.append(regions[i])
    #         regions = new_regions

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
            if abs(regions[i][1] - minc) < 300:
                minr = min(regions[i][0], minr)
                minc = min(regions[i][1], minc)
                maxr = max(regions[i][2], maxr)
                maxc = max(regions[i][3], maxc)
            else:
                merged_regions.append((minr, minc, maxr, maxc))
                print minr, minc, maxr, maxc
                minr, minc, maxr, maxc = regions[i]
    if maxr != 0:
        merged_regions.append((minr, minc, maxr, maxc))
        print minr, minc, maxr, maxc
    regions = merged_regions
    count = len(regions)

    #print 'count: ', count
    save_count = 0
    if count >= 2:
        for i in range(1, -1, -1):
            minr, minc, maxr, maxc = regions[i]
            #print minr, minc, maxr, maxc
            page_no = min_page_no + (1-i)
            page_image = image[minr:maxr+1, minc:maxc+1]
            page_binary_image = bw[minr:maxr+1, minc:maxc+1]
            angle = deskew_page(page_image, page_binary_image)
            if angle != 0.0:
                image_new = rotate(page_image, angle, mode='edge')
                page_image = (image_new * 255).astype('ubyte')
            io.imsave('%03d.jpg' % page_no, page_image)
            save_count = save_count + 1
    elif count == 1:
        print 'vol_regions: ', len(vol_regions)
        minr, minc, maxr, maxc = regions[0]
        c_center = (minc + maxc)/2
        if c_center < image_width / 2: # left part
            #print 'left part'
            page_no = min_page_no
            if vol_regions:
                vol_center = (vol_regions[0][1] + vol_regions[0][3]) / 2
                if vol_center > image_width / 2:
                    page_image = image[minr:maxr + 1, maxc + 100:image_width - 20]
                    page_binary_image = bw[minr:maxr + 1, maxc + 100:image_width - 20]
                    angle = deskew_page(page_image, page_binary_image)
                    if angle != 0.0:
                        image_new = rotate(page_image, angle, mode='edge')
                        page_image = (image_new * 255).astype('ubyte')
                    io.imsave('%03d.jpg' % page_no, page_image)
                    page_no = page_no + 1
                    save_count = save_count + 1
            page_image = image[minr:maxr + 1, minc:maxc + 1]
            io.imsave('%03d.jpg' % page_no, page_image)
            save_count = save_count + 1
        else:
            #print 'right part'
            page_image = image[minr:maxr + 1, minc:maxc + 1]
            page_binary_image = bw[minr:maxr + 1, minc:maxc + 1]
            angle = deskew_page(page_image, page_binary_image)
            if angle != 0.0:
                image_new = rotate(page_image, angle, mode='edge')
                page_image = (image_new * 255).astype('ubyte')
            io.imsave('%03d.jpg' % min_page_no, page_image)
            save_count = save_count + 1

            if vol_regions:
                vol_center = (vol_regions[0][1] + vol_regions[0][3]) / 2
                if vol_center < image_width / 2:
                    page_image = image[minr:maxr + 1, 20:minc - 100]
                    page_binary_image = bw[minr:maxr + 1, 20:minc - 100]
                    angle = deskew_page(page_image, page_binary_image)
                    if angle != 0.0:
                        image_new = rotate(page_image, angle, mode='edge')
                        page_image = (image_new * 255).astype('ubyte')
                    io.imsave('%03d.jpg' % (min_page_no + 1), page_image)
                    save_count = save_count + 1
    else: # count == 0
        for region in regionprops(label_image):
            minr, minc, maxr, maxc = region.bbox
            if (maxr - minr) < 1000:
                continue
            regions.append((minr, minc, maxr, maxc))
        if len(regions) > 0:
            minr, minc, maxr, maxc = regions[0]
            c_center = (minc + maxc) / 2
            if c_center < image_width / 2:
                page_image = image[20:image_height-20, 20:image_width/2 - 20]
                page_binary_image = bw[20:image_height-20, 20:image_width/2 - 20]
                angle = deskew_page(page_image, page_binary_image)
                if angle != 0.0:
                    image_new = rotate(page_image, angle, mode='edge')
                    page_image = (image_new * 255).astype('ubyte')
                io.imsave('%03d.jpg' % min_page_no, page_image)
                save_count = save_count + 1
            else:
                page_image = image[20:image_height - 20, image_width/2 + 20:image_width-20]
                page_binary_image = bw[20:image_height - 20, image_width/2 + 20:image_width-20]
                angle = deskew_page(page_image, page_binary_image)
                if angle != 0.0:
                    image_new = rotate(page_image, angle, mode='edge')
                    page_image = (image_new * 255).astype('ubyte')
                io.imsave('%03d.jpg' % min_page_no, page_image)
                save_count = save_count + 1
    return save_count


    #if count > 2:
    #    print '=====%s region count: %s=====' % (src_image, count)
    #else:
    #    print '%s region count: %s' % (src_image, count)

    #ax.imshow(image_label_overlay)
    #plt.show()
    #io.imsave(dst_image, image)

if __name__ == '__main__':
    if sys.argv[1].find('.') != -1:
        min_page_no = int(sys.argv[2])
        split_page(sys.argv[1], min_page_no)
    else:
        file_list_out = subprocess.check_output('ls %s*.jpg' % sys.argv[1], shell=True)
        file_list = file_list_out.rstrip().split()
        min_page_no = int(sys.argv[2])
        last_page_count = 0
        for i in range(len(file_list)):
            src_image = file_list[i]
            min_page_no = min_page_no + last_page_count
            last_page_count = split_page(src_image, min_page_no)

