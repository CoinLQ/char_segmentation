# coding: utf-8
from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from skimage import io
from skimage.measure import label
from skimage.measure import regionprops

import sys
import json
import redis
from operator import itemgetter

from common import binarisation, centroid, geometric_center

redis_client = redis.StrictRedis(host='localhost', port=6379, db=0)

# 小字区宽度阈值
SMALL_FONT_REGION_WIDTH = 35

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


class WhiteRegion:
    """
    [top, bottom]
    """
    def __init__(self, top, bottom):
        self.top = top
        self.bottom = bottom
        self.height = self.bottom - self.top + 1

    def set_top_bottom(self, top, bottom):
        self.top = top
        self.bottom = bottom
        self.height = self.bottom - self.top + 1

    def set_top(self, top):
        self.top = top
        self.height = self.bottom - self.top + 1

    def set_bottom(self, bottom):
        self.bottom = bottom
        self.height = self.bottom - self.top + 1


def find_top(binary_line, region_height, region_width, start):
    white_regions = []
    cur_white_region_top = -1
    cur_non_white_region_top = -1
    if binary_line[0] == 0:
        cur_white_region_top = 0
    else:
        cur_non_white_region_top = 0
    for i in range(start + 1, region_height):
        if binary_line[i] == 0:
            if binary_line[i - 1] != 0:  # i is the begin of a new white region
                cur_white_region_top = i
                if i - cur_non_white_region_top >= region_width:
                    break
        else:
            if binary_line[i - 1] == 0:  # i is the begin of a new non-white region
                cur_non_white_region_top = i
                white_regions.append(WhiteRegion(cur_white_region_top, i - 1))
    #while (white_regions and white_regions[-1].height < region_width / 2):
    #    white_regions.pop()
    # 合并相邻的空白区域
    pop_index = []
    white_regions_length = len(white_regions)
    for i in range(white_regions_length - 1):
        if (white_regions[i + 1].top - white_regions[i].bottom) <= region_width / 4 and \
                np.sum(binary_line[white_regions[i].bottom : white_regions[i + 1].top]) <= region_width / 4:
            white_regions[i + 1].set_top(white_regions[i].top)
            pop_index.append(i)
    if pop_index:
        new_white_regions = []
        for i in range(white_regions_length):
            if i not in pop_index:
                new_white_regions.append(white_regions[i])
        white_regions = new_white_regions
    if white_regions:
        top = white_regions[0].bottom
    else:
        top = 0
    return top


def find_bottom(binary_line, region_height, region_width, start):
    white_regions = []
    cur_white_region_bottom = -1
    cur_non_white_region_bottom = -1
    if binary_line[start] == 0:
        cur_white_region_bottom = start
    else:
        cur_non_white_region_bottom = start
    for i in range(start - 1, -1, -1):
        if binary_line[i] == 0:
            if binary_line[i + 1] != 0:  # i is the bottom of a new white region
                cur_white_region_bottom = i
                if cur_non_white_region_bottom - i >= region_width:
                    break
        else:
            if binary_line[i + 1] == 0:  # i is the bottom of a new non-white region
                cur_non_white_region_bottom = i
                white_regions.append(WhiteRegion(i + 1, cur_white_region_bottom))
    #while (white_regions and white_regions[-1].height < region_width / 4):
    #    white_regions.pop()
    # 合并相邻的空白区域
    pop_index = []
    white_regions_length = len(white_regions)
    for i in range(white_regions_length - 1):
        if (white_regions[i].top - white_regions[i + 1].bottom) <= region_width / 4 and \
                np.sum(binary_line[white_regions[i + 1].bottom : white_regions[i].top]) <= region_width / 4 :
            white_regions[i + 1].set_bottom(white_regions[i].bottom)
            pop_index.append(i)
    if pop_index:
        new_white_regions = []
        for i in range(white_regions_length):
            if i not in pop_index:
                new_white_regions.append(white_regions[i])
        white_regions = new_white_regions
    if white_regions:
        bottom = white_regions[0].top
    else:
        bottom = region_height - 1
    return bottom


def find_min_pos(binary_line, start, middle, end):
    # 相对于start的坐标
    if start < middle:
        min_pos_up = np.argmin(binary_line[start : middle])
    else:
        min_pos_up = 0
    if middle < end:
        min_pos_down = np.argmin(binary_line[middle: end]) + (middle - start)
    else:
        min_pos_down = middle - start
    if binary_line[min_pos_up + start] < binary_line[min_pos_down + start]:
        min_pos = min_pos_up
    elif binary_line[min_pos_up + start] > binary_line[min_pos_down + start]:
        min_pos = min_pos_down
    else:
        if (middle - start - min_pos_up) < (min_pos_down - (middle - start)):
            min_pos = min_pos_up
        else:
            min_pos = min_pos_down
    return min_pos


def find_nearest_label_line(label_lines, start, binary_line, thresh=None):
    min_sum_of_pixels = 10000
    min_pos = 0
    if thresh is None:
        thresh = 12
    for pos in label_lines:
        distance = abs(pos - start)
        if distance < thresh and pos < binary_line.shape[0]:
            sum_of_pixels = binary_line[pos]
            if sum_of_pixels < min_sum_of_pixels:
                min_sum_of_pixels = sum_of_pixels
                min_pos = pos
    return min_pos, min_sum_of_pixels


def find_label_line(label_lines, start, end=None):
    for pos in label_lines:
        if pos >= start and (end is None or pos <= end):
            return pos
    return -1


class BBoxLineRegion:
    def __init__(self):
        self.bbox_lst = []
        self.left = 10000
        self.right = 0


def get_line_region_lst(label_image):
    total_bbox_lst = []
    for region in regionprops(label_image):
        if region.area >= 6:
            minr, minc, maxr, maxc = region.bbox
            total_bbox_lst.append([minr, minc, maxr, maxc])

    total_bbox_lst.sort(key=itemgetter(2), reverse=True)
    total_bbox_lst.sort(key=itemgetter(3), reverse=True)
    line_lst = []
    line_region = BBoxLineRegion()
    for bbox in total_bbox_lst:
        if bbox[3] - bbox[1] >= 70: # 如果box的宽大于70，可能是两列的字有粘连
            continue
        middle = (bbox[1] + bbox[3]) / 2
        if line_region.right == 0:
            # line_region.bbox_lst.append(bbox)
            line_region.left = bbox[1]
            line_region.right = bbox[3]
            continue
        line_middle = (line_region.left + line_region.right) / 2
        if (middle >= line_region.left and middle <= line_region.right) or (
                line_middle >= bbox[1] and line_middle <= bbox[3]):
            # line_region.bbox_lst.append(bbox)
            if line_region.left > bbox[1]:
                line_region.left = bbox[1]
            if line_region.right < bbox[3]:
                line_region.right = bbox[3]
        else:
            if (line_region.right - line_region.left > 5):
                line_lst.append(line_region)
            line_region = BBoxLineRegion()
    if line_region.right != 0:  # and (line_region.right - line_region.left > 5):
        line_lst.append(line_region)
    new_line_lst = []
    for line_region in line_lst:
        if line_region.right - line_region.left >= 100: # 对宽度超过100的列分成两列
            line_region1 = BBoxLineRegion()
            line_region2 = BBoxLineRegion()
            line_middle = (line_region.left + line_region.right) / 2
            line_region1.left = line_middle
            line_region1.right = line_region.right
            line_region2.left = line_region.left
            line_region2.right = line_middle
            new_line_lst.append(line_region1)
            new_line_lst.append(line_region2)
        elif line_region.right - line_region.left >= 18:
            new_line_lst.append(line_region)
    line_lst = new_line_lst
    for i in range(len(line_lst) - 1):
        distance = line_lst[i].left - line_lst[i + 1].right
        if distance <= 25:
            middle = (line_lst[i].left + line_lst[i + 1].right) / 2
            line_lst[i].left = middle
            line_lst[i + 1].right = middle

    # 重新添加bbox
    line_count = len(line_lst)
    i = 0
    left = line_lst[i].left
    right = line_lst[i].right
    line_width = right - left
    temp_lst = []
    for bbox in total_bbox_lst:
        while bbox[3] <= left:
            i = i + 1
            if i >= line_count:
                break
            left = line_lst[i].left
            right = line_lst[i].right
            line_width = right - left

            for t_bbox in temp_lst:
                # print t_bbox
                if (right - t_bbox[1]) > line_width / 3 and t_bbox[3] > left:
                    if t_bbox[3] - t_bbox[1] >= 70:
                        new_bbox = [t_bbox[0], max(left, t_bbox[1]), t_bbox[2], min(right, t_bbox[3])]
                        t_bbox = new_bbox
                    line_lst[i].bbox_lst.append(t_bbox)
            temp_lst = []
        if bbox[1] < left:
            temp_lst.append(bbox)
            if bbox[3] > left:
                if (bbox[3] - left) > line_width / 3:
                    if bbox[3] - bbox[1] >= 70:
                        new_bbox = [bbox[0], max(left, bbox[1]), bbox[2], bbox[3]]
                        bbox = new_bbox
                    line_lst[i].bbox_lst.append(bbox)
                    pass
                else:
                    center = (bbox[1] + bbox[3]) / 2
                    if bbox[3] - bbox[1] >= 70:
                        new_bbox = [bbox[0], max(left, bbox[1]), bbox[2], bbox[3]]
                        bbox = new_bbox
                    if center > left and center < right:
                        line_lst[i].bbox_lst.append(bbox)
                        pass
        elif bbox[1] < right:
            if bbox[3] <= right:
                line_lst[i].bbox_lst.append(bbox)
                pass
            else:
                if (right - bbox[1]) > line_width / 3:
                    line_lst[i].bbox_lst.append(bbox)
                    pass
                else:
                    center = (bbox[1] + bbox[3]) / 2
                    if center > left and center < right:
                        line_lst[i].bbox_lst.append(bbox)
                        pass

    for line_region in line_lst:
        line_width = line_region.right - line_region.left
        bbox_lst = line_region.bbox_lst
        bbox_count = len(bbox_lst)

        bbox_lst.sort(key=itemgetter(2), reverse=True)
        bbox_lst.sort(key=itemgetter(0))
        # 合并有交叠的框
        if line_width <= SMALL_FONT_REGION_WIDTH: #　对小字区不做合并
           continue
        while True:
            pop_index = []
            for i in range(bbox_count - 1):
                j = i + 1
                k = i + 2

                # 第一个字
                if (i == 0 and (bbox_lst[i][2] - bbox_lst[i][0] <= 20)) or \
                        (i == bbox_count - 2 and (bbox_lst[j][2] - bbox_lst[j][0] <= 20)):
                    pop_index.append(i)
                    maxr = max(bbox_lst[j][2], bbox_lst[i][2])
                    bbox_lst[j][0] = bbox_lst[i][0]
                    bbox_lst[j][2] = maxr
                    bbox_lst[j][1] = min(bbox_lst[i][1], bbox_lst[j][1])
                    bbox_lst[j][3] = max(bbox_lst[i][3], bbox_lst[j][3])
                    continue

                # i包含j
                if bbox_lst[j][0] >= bbox_lst[i][0] and \
                                bbox_lst[j][2] <= bbox_lst[i][2]:
                    pop_index.append(i)
                    bbox_lst[j][0] = bbox_lst[i][0]
                    bbox_lst[j][2] = bbox_lst[i][2]
                    bbox_lst[j][1] = min(bbox_lst[i][1], bbox_lst[j][1])
                    bbox_lst[j][3] = max(bbox_lst[i][3], bbox_lst[j][3])
                    continue
                distance = bbox_lst[i][2] - bbox_lst[j][0]
                height1 = bbox_lst[i][2] - bbox_lst[i][0]
                height2 = bbox_lst[j][2] - bbox_lst[j][0]
                if distance > height1 * 0.6 or distance > height2 * 0.6:
                    pop_index.append(i)
                    bbox_lst[j][0] = bbox_lst[i][0]
                    bbox_lst[j][2] = max(bbox_lst[j][2], bbox_lst[i][2])
                    bbox_lst[j][1] = min(bbox_lst[j][1], bbox_lst[i][1])
                    bbox_lst[j][3] = max(bbox_lst[j][3], bbox_lst[i][3])
                    continue

                # 对相邻的三个框一起判断
                if i < bbox_count - 2:
                    if bbox_lst[j][0] <= bbox_lst[i][2] and \
                                    bbox_lst[j][2] < bbox_lst[k][0]:  # i与j交错，j与k不交错
                        minr = bbox_lst[i][0]
                        maxr = max(bbox_lst[j][2], bbox_lst[i][2])
                        if maxr - minr < 55 or (bbox_lst[j][2] - bbox_lst[j][0] <= 30) or \
                                (bbox_lst[i][2] - bbox_lst[j][0] > 15):
                            pop_index.append(i)
                            bbox_lst[j][0] = minr
                            bbox_lst[j][1] = min(bbox_lst[j][1], bbox_lst[i][1])
                            bbox_lst[j][2] = maxr
                            bbox_lst[j][3] = max(bbox_lst[j][3], bbox_lst[i][3])
                            continue
                    elif bbox_lst[j][0] > bbox_lst[i][2] and \
                                    bbox_lst[j][2] >= bbox_lst[k][0]:  # i与j不交错，j与k交错
                        minr = bbox_lst[j][0]
                        maxr = max(bbox_lst[j][2], bbox_lst[k][2])
                        if maxr - minr < 55 or (bbox_lst[j][2] - bbox_lst[j][0] <= 30) or \
                                (bbox_lst[j][2] - bbox_lst[k][0] > 15):
                            pop_index.append(j)
                            bbox_lst[k][0] = minr
                            bbox_lst[k][1] = min(bbox_lst[j][1], bbox_lst[k][1])
                            bbox_lst[k][2] = maxr
                            bbox_lst[k][3] = max(bbox_lst[j][3], bbox_lst[k][3])
                            continue
                    else:
                        connected = ( bbox_lst[j][0] <= bbox_lst[i][2] and bbox_lst[j][2] >= bbox_lst[k][0] ) #:  # i与j,k都交错
                        if bbox_lst[j][2] - bbox_lst[j][0] <= 35:
                            if bbox_lst[j][0] - bbox_lst[i][2] < bbox_lst[k][0] - bbox_lst[j][2]:
                                minr = bbox_lst[i][0]
                                maxr = max(bbox_lst[j][2], bbox_lst[i][2])
                                if connected or maxr - minr < 55:
                                    pop_index.append(i)
                                    bbox_lst[j][0] = minr
                                    bbox_lst[j][1] = min(bbox_lst[j][1], bbox_lst[i][1])
                                    bbox_lst[j][2] = maxr
                                    bbox_lst[j][3] = max(bbox_lst[j][3], bbox_lst[i][3])
                                    continue
                            else:
                                minr = bbox_lst[j][0]
                                maxr = max(bbox_lst[j][2], bbox_lst[k][2])
                                if connected or maxr - minr < 55:
                                    pop_index.append(j)
                                    bbox_lst[k][0] = minr
                                    bbox_lst[k][1] = min(bbox_lst[j][1], bbox_lst[k][1])
                                    bbox_lst[k][2] = maxr
                                    bbox_lst[k][3] = max(bbox_lst[j][3], bbox_lst[k][3])
                                    continue
                if (abs(height1-56) > 14 and abs(height2-56) > 14) and distance >= 0:
                    pop_index.append(i)
                    bbox_lst[j][0] = bbox_lst[i][0]
                    bbox_lst[j][2] = max(bbox_lst[j][2], bbox_lst[i][2])
                    bbox_lst[j][1] = min(bbox_lst[j][1], bbox_lst[i][1])
                    bbox_lst[j][3] = max(bbox_lst[j][3], bbox_lst[i][3])
                    continue

            if pop_index:
                new_bbox_lst = []
                for i in range(bbox_count):
                    if i not in pop_index:
                        new_bbox_lst.append(bbox_lst[i])
                line_region.bbox_lst = new_bbox_lst
                bbox_lst = new_bbox_lst
                bbox_count = len(bbox_lst)
            else:
                break
    return line_lst

def get_label_lines(line_lst, region_left, region_right, height, cur_line_region=None):
    local_label_regions = []
    if cur_line_region is None:
        middle = (region_left + region_right) / 2
        for line_region in line_lst:
            if line_region.left < middle and middle < line_region.right:
                local_label_regions = line_region.bbox_lst
    else:
        local_label_regions = cur_line_region.bbox_lst

    label_lines = []
    if local_label_regions:
        if local_label_regions[0][2] - local_label_regions[0][0] <= 30:
            label_lines.append(max(local_label_regions[0][0] - 4, 0))
        else:
            label_lines.append(max(local_label_regions[0][0] - 2, 0))
        for i in range(len(local_label_regions) - 1):
            h1 = local_label_regions[i][2] - local_label_regions[i][0]
            h2 = local_label_regions[i + 1][2] - local_label_regions[i + 1][0]
            y1 = local_label_regions[i][2]
            y2 = local_label_regions[i + 1][0]
            if y1 < y2:
                if (y2 - y1 < 30):
                    if (h1 < 35 or h2 < 35):
                        label_lines.append(y1 + int((y2 - y1) * h2 * 1.0 / (h1 + h2)))
                    else:
                        label_lines.append((y1 + y2) / 2)
                else:
                    label_lines.append(y1)
                    label_lines.append(y2)
            else:
                if y1 - y2 < 20:
                    label_lines.append((y1 + y2) / 2)
                else:
                    label_lines.append(y2)
                    label_lines.append(y1)
        label_lines.append(min(local_label_regions[-1][2] + 4, height - 1))
    return label_lines

def get_region_label_lines(label_lines, region_top, region_bottom):
    region_label_lines = []
    for l in label_lines:
        if l >= region_top - 10 and l <= region_bottom + 10:
            region_label_lines.append(l - region_top)
    return region_label_lines

MAX_DIFF_FROM_HORIZONTAL = 9


def horizontal_align(line_lst, region_lst, char_lst):
    max_line_no = region_lst[-1][u'line_no']
    line_no_lst = [[], []] # 右半部分，左半部分
    for region in region_lst:
        line_no = region[u'line_no']
        mark = region.get(u'mark', None)
        if line_no <= 6 and mark != u'<': # 不考虑带小字的列区域
            text = region[u'text']
            if len(text) == 17: # 只将字数为17的列区域用于计算水平切分线平均位置
                line_no_lst[0].append(line_no)
        if line_no >= max_line_no - 5 and mark != u'<':
            text = region[u'text']
            if len(text) == 17:
                line_no_lst[1].append(line_no)
    for i in range(2): # 右半部分，左半部分
        if len(line_no_lst[i]) < 3: # 如果列区域不到3个，则忽略
            continue
        total_horizontal_pos_lst = [[], [], [], [], [], [], []]
        for char in char_lst:
            line_no = char[u'line_no']
            if line_no > line_no_lst[i][-1]: # 比列表最大的line_no还大，则跳出循环
                break
            if line_no not in line_no_lst[i]:
                continue
            if i == 0:
                line_idx = line_no - 1
            else:
                line_idx = line_no - (max_line_no - 5)
            char_no = char[u'char_no']
            if char_no == 1:
                top = char[u'top']
                total_horizontal_pos_lst[line_idx].append(top)
            bottom = char[u'bottom']
            total_horizontal_pos_lst[line_idx].append(bottom)
        # 计算中位值
        median_lst = []
        for j in range(18):
            pos_lst = []
            for horizontal_pos_lst in total_horizontal_pos_lst:
                if not horizontal_pos_lst:
                    continue
                pos_lst.append(horizontal_pos_lst[j])
            median = np.median(pos_lst) # 为减小异常值的影响，利用中位值计算
            median_lst.append(int(round(median)))
        for char in char_lst: # 修正char的top, bottom属性
            line_no = char[u'line_no']
            if line_no > line_no_lst[i][-1]: # 比列表最大的line_no还大，则跳出循环
                break
            if line_no not in line_no_lst[i]:
                continue
            char_no = char[u'char_no']
            top = char[u'top']
            if abs(top - median_lst[char_no-1]) > MAX_DIFF_FROM_HORIZONTAL:
                char[u'top'] = median_lst[char_no-1]
            bottom = char[u'bottom']
            if abs(bottom - median_lst[char_no]) > MAX_DIFF_FROM_HORIZONTAL:
                char[u'bottom'] = median_lst[char_no]


def charseg(imagepath, region_lst, page_id, to_binary=True):
    image = io.imread(imagepath, 0)
    if to_binary:
        binary = binarisation(image)
        binary_image = (binary * 255).astype('ubyte')
    else:
        binary_image = image
    char_lst = []

    bw = (1 - binary).astype('ubyte')
    image_height, image_width = bw.shape
    label_image = label(bw, connectivity=2)
    line_lst = get_line_region_lst(label_image)

    last_avg_height = 0
    last_region_left = -1
    label_lines = []
    for region in region_lst:
        region_top = region[u'top']
        region_bottom = region[u'bottom']
        region_left = region[u'left']
        region_right = region[u'right']
        text = region[u'text']
        if not text:
            continue

        line_no = region[u'line_no']
        region_no = region[u'region_no']
        height = region_bottom - region_top
        region_width = region_right - region_left

        mark = region.get(u'mark', None)
        #if mark is not None:
        #    continue

        region_image = image[region_top : region_bottom + 5, region_left : region_right + 5]
        binary_region_image = binary_image[region_top: region_bottom + 5, region_left: region_right + 5]
        binary = (binary_region_image / 255).astype('ubyte')
        binary = 1 - binary

        if region_left != last_region_left:
            label_lines = get_label_lines(line_lst, region_left, region_right, image_height)
        else:
            last_region_left = region_left
        if region_top != 0:
            region_label_lines = get_region_label_lines(label_lines, region_top, region_bottom)
        else:
            region_label_lines = label_lines

        binary_line = binary.sum(axis=1)

        step = 5
        if region_width <= SMALL_FONT_REGION_WIDTH:
            step = 2
        start = region_label_lines[0]
        end = region_label_lines[-1]
        start = max(0, start - step)
        end = min(end + 1 + step, height)
        h = end - start
        text = text.strip(u'　')
        # eprint(text)
        # eprint(region_label_lines)
        if u'　' not in text:
            char_count = len(text)
            if not char_count:
                continue
            avg_height = h * 1.0 / char_count
            # if last_avg_height != 0 and abs(avg_height - last_avg_height) > 10:
            #     avg_height = last_avg_height
            # else:
            #     last_avg_height = avg_height
            if avg_height > 54:
                avg_height = 50
            rel_top = 0
            rel_bottom = 0
            for i in range(char_count):
                # 取字的平均高度
                avg_height_key = u'ch_h:%s:avg' % text[i]
                cur_avg_height = redis_client.get(avg_height_key)
                if cur_avg_height is None:
                    cur_avg_height = avg_height
                else:
                    cur_avg_height = float(cur_avg_height)
                    if mark == u'@':
                        cur_avg_height = avg_height
                    elif cur_avg_height < avg_height / 2:
                        cur_avg_height = avg_height
                if rel_bottom:
                    rel_top = rel_bottom
                else:
                    rel_top = int(i * avg_height) + start
                rel_bottom = rel_top + int(cur_avg_height)
                if rel_bottom >= end - 1:
                    rel_bottom = end - 1
                min_pos, min_sum_of_pixels = find_nearest_label_line(region_label_lines, rel_bottom - 1, binary_line, 15)
                if min_sum_of_pixels < 10000:
                    #eprint('min_pos, min_sum_of_pixels: ', min_pos, min_sum_of_pixels)
                    rel_bottom = min_pos
                else:
                    start1 = rel_bottom - 1 - int(avg_height / 4)
                    if start1 < start:
                        start1 = start
                    end1 = rel_bottom - 1 + int(avg_height / 4)
                    if end1 > end - 1:
                        end1 = end - 1
                    pos = -1 #find_label_line(region_label_lines, start1, end1)
                    if pos > 0:
                        rel_bottom = pos
                    else:
                        if binary_line[rel_bottom - 1] != 0:
                            min_pos = find_min_pos(binary_line, start1, rel_bottom - 1, end1)
                            rel_bottom = min_pos + start1

                top = rel_top + region_top
                bottom = rel_bottom + region_top
                # if i == char_count - 1:
                #     if (bottom - top) < avg_height:
                #         bottom = top + int(avg_height)
                #         rel_bottom = rel_top + int(avg_height)
                if i == char_count - 1:  # 最后一个字
                    c = geometric_center(binary_line[rel_top:rel_bottom])
                    new_rel_bottom = rel_top + 2 * c
                    if new_rel_bottom + region_top > image_height:
                        new_rel_bottom = image_height - region_top
                    if rel_bottom < new_rel_bottom and np.sum(binary_line[rel_bottom:new_rel_bottom]) < 5:
                        rel_bottom = new_rel_bottom

                char_no = i + 1
                char = {
                    u'top': top,
                    u'bottom': bottom,
                    u'left': region_left,
                    u'right': region_right,
                    u'char': text[i],
                    u'line_no': line_no,
                    u'region_no': region_no,
                    u'char_no': char_no,
                }
                char_lst.append(char)
                if rel_top >= rel_bottom:
                    eprint('rel_top, rel_bottom: ', rel_top, rel_bottom)
        else:
            text_segs = text.split(u'　')
            text_segs_cnt = len(text_segs)
            cur_pos = start
            start_pos = start
            added_char_count = 0

            # 找到region_label_lines对应不同文本段的分隔索引
            density_lst = []
            for i in range(len(region_label_lines) - 1):
                density = np.sum(binary_line[region_label_lines[i] : region_label_lines[i+1]]) * 1.0 / (region_label_lines[i+1] - region_label_lines[i])
                density_lst.append((i, density))
            density_lst.sort(key=itemgetter(1))
            label_line_seg_index_lst = []
            if text_segs_cnt > 1:
                for i in range(text_segs_cnt - 1):
                    label_line_seg_index_lst.append(density_lst[i][0])
            label_line_seg_index_lst.sort()
            del density_lst

            for text_segs_idx in range(text_segs_cnt):
                start_pos = cur_pos
                txt = text_segs[text_segs_idx]
                if text_segs_idx != text_segs_cnt - 1:
                    end_pos = region_label_lines[ label_line_seg_index_lst[text_segs_idx] ] + step
                    h = end_pos - cur_pos
                    cur_pos = region_label_lines[ label_line_seg_index_lst[text_segs_idx] + 1 ] - step
                else:
                    end_pos = end
                    h = end_pos - cur_pos
                    cur_pos = end
                char_count = len(txt)
                if not char_count:
                    continue
                avg_height = h * 1.0 / char_count
                #print('avg_height:', avg_height)
                rel_top = 0
                rel_bottom = 0
                for i in range(char_count):
                    # 取字的平均高度
                    avg_height_key = u'ch_h:%s:avg' % text[i]
                    cur_avg_height = redis_client.get(avg_height_key)
                    if cur_avg_height is None:
                        cur_avg_height = avg_height
                    else:
                        cur_avg_height = float(cur_avg_height)
                        if mark == u'@':
                            cur_avg_height = region_width / 50.0 * cur_avg_height
                        elif cur_avg_height < avg_height / 2:
                            cur_avg_height = avg_height
                    if rel_bottom:
                        rel_top = rel_bottom
                    else:
                        rel_top = int(i * avg_height) + start_pos
                    rel_bottom = rel_top + int(cur_avg_height)
                    if rel_bottom >= end_pos:
                        rel_bottom = end_pos
                    min_pos, min_sum_of_pixels = find_nearest_label_line(region_label_lines, rel_bottom - 1,
                                                                         binary_line, 0.2 * region_width)
                    if rel_bottom >= end_pos:
                        rel_bottom = end_pos
                    if min_sum_of_pixels < 10000:
                        rel_bottom = min_pos
                    else:
                        start1 = rel_bottom - 1 - int(cur_avg_height / 3)
                        if start1 < start_pos:
                            start1 = start_pos
                        end1 = rel_bottom - 1 + int(cur_avg_height / 3)
                        if end1 > end_pos - 1:
                            end1 = end_pos - 1
                        if binary_line[rel_bottom - 1] != 0:
                            min_pos = find_min_pos(binary_line, start1, rel_bottom - 1, end1)
                            rel_bottom = min_pos + start1
                    if i == char_count - 1: # 最后一个字
                        c = geometric_center(binary_line[rel_top:rel_bottom])
                        new_rel_bottom = rel_top + 2 * c
                        if new_rel_bottom + region_top > image_height:
                            new_rel_bottom = image_height - region_top
                        if rel_bottom < new_rel_bottom and np.sum(binary_line[rel_bottom:new_rel_bottom]) < 5:
                            rel_bottom = new_rel_bottom

                    top = rel_top + region_top
                    bottom = rel_bottom + region_top
                    added_char_count = added_char_count + 1
                    char_no = added_char_count
                    char = {
                        u'top': top,
                        u'bottom': bottom,
                        u'left': region_left,
                        u'right': region_right,
                        u'char': txt[i],
                        u'line_no': line_no,
                        u'region_no': region_no,
                        u'char_no': char_no,
                    }
                    char_lst.append(char)
                    if rel_top >= rel_bottom:
                        eprint('rel_top, rel_bottom: ', rel_top, rel_bottom)

    # plugins
    # line_lst, region_lst, char_lst
    horizontal_align(line_lst, region_lst, char_lst)
    return char_lst

if __name__ == '__main__':
    imagepath = sys.argv[1]
    if len(sys.argv) == 4 and sys.argv[3] == 'binary':
        to_binary = True
    else:
        to_binary = False
    page_id = sys.argv[1][:-4]
    with open(sys.argv[2], 'r') as f:
        region_lst = json.load(f, 'utf-8')
        char_lst = charseg(imagepath, region_lst, page_id.decode('utf-8'), to_binary)
        output = json.dumps(char_lst, ensure_ascii=False, indent=True)
        print(output.encode('utf-8'))

        image = io.imread(imagepath, 0)
        # save char images
        for char in char_lst:
            top = char[u'top']
            bottom = char[u'bottom']
            left = char[u'left']
            right = char[u'right']
            line_no = char[u'line_no']
            region_no = char[u'region_no']
            char_no = char[u'char_no']
            char_image = image[top: bottom, left: right]
            char_filename = u'%s-%s-%s-%s.jpg' % (page_id, line_no, region_no, char_no)
            try:
                io.imsave(char_filename, char_image)
            except Exception, e:
                eprint(e, ", line_no: ", line_no, ", char_no: ", char_no,
                       ", top: ", top, ", bottom: ", bottom)

        # plot
        fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
        colors = ['red', 'blue', 'darkgreen', 'm', 'c']
        color_index = 0
        for char in char_lst:
            top = char[u'top']
            bottom = char[u'bottom']
            left = char[u'left']
            right = char[u'right']
            rect = mpatches.Rectangle((left, top), right - left, bottom - top,
                                      fill=False, edgecolor=colors[color_index], linewidth=2)
            ax.add_patch(rect)
            color_index = (color_index + 1) % 5
        ax.imshow(image)
        ax.margins(x=0.02, y=0.02)
        plt.show()