# coding: utf-8
from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from skimage import io
from skimage.filters import threshold_otsu, threshold_adaptive
from skimage.segmentation import clear_border
from skimage.measure import label
from skimage.morphology import closing, square
from skimage.measure import regionprops
from skimage.color import label2rgb
import sys, json
from operator import itemgetter

from charseg import find_top, find_min_pos, get_line_region_lst, get_label_lines
from charseg import find_nearest_label_line, find_label_line
from common import binarisation

class BBoxLineRegion:
    def __init__(self):
        self.bbox_lst = []
        self.left = 10000
        self.right = 0

LEFT_MARKS = [u'<', u'【', u'@', u'(', u'$', u'{']
RIGHT_MARKS = [u'>', u'】', u'@', u')', u'$', u'}']

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

def add_to_region_lst(text, left, right, top, bottom, line_no, region_no, page_bar_no, region_lst, mark=None):
    region = {
        u'text': text,
        u'left': left,
        u'right': right,
        u'top': top,
        u'bottom': bottom,
        u'line_no': line_no,
        u'region_no': region_no,
        u'page_bar_no': page_bar_no,
    }
    if mark:
        region[u'mark'] = mark
    region_lst.append(region)

def region_seg(image, binary_image, image_height, page_bar_no, line_no, line_region, text, region_lst):
    mark = None
    if text[0] == u'$' and text[-1] == u'$': # 一般$在首尾
        text = text[1: -1]
    elif text[0] == u'(' and text[-1] == u')':
        text = text[1: -1]
    elif text[0] == u'@' and text[-1] == u'@': # 一般@在首尾
        text = text[1: -1]
        mark = u'@'
    if u'(' in text or u')' in text:
        text = text.replace(u'(', u'').replace(u')', u'')
    if u'<' not in text: # and u'【' not in text:
        region = {
            u'text': text,
            u'left': line_region.left,
            u'right': line_region.right,
            u'top': 0,
            u'bottom': image_height,
            u'line_no': line_no,
            u'region_no': 1,
            u'page_bar_no': page_bar_no,
        }
        if mark:
            region[u'mark'] = mark
        region_lst.append(region)
        return

    # 洪武南藏
    avg_height = 52
    avg_small_height = 52

    binary = binary_image[:, line_region.left:line_region.right]
    binary_line = binary.sum(axis=1)

    region_width = line_region.right - line_region.left
    label_lines = get_label_lines(None, line_region.left, line_region.right, image_height, line_region)
    step = avg_height / 12
    start = label_lines[0]
    end = label_lines[-1]
    # for i in xrange(0, image_height, step):
    #     if np.sum(binary_line[i:i + step]) >= 10:
    #         start = i
    #         break
    # for i in xrange(image_height - 1, 0, -step):
    #     if np.sum(binary_line[i:i + step]) >= 10:
    #         end = i
    #         break
    start = max(0, start)
    end = min(end + 1 + step, image_height)
    h = end - start

    if u'<' in text:
        right_small_cnt = 0
        left_small_cnt = 0
        right_small_flag = False
        left_small_flag = False
        char_cnt = len(text)
        equivalent_char_cnt = 0

        cur_pos = 0
        cur_start = start
        passed_char_cnt = 0
        passed_small_char_cnt = 0
        region_no = 1
        while cur_pos <= char_cnt - 3:
            right_mark_pos = text.find(u'<', cur_pos)
            if right_mark_pos == -1:
                break
            if cur_pos < right_mark_pos:
                y = cur_start - 1
                cur_text = text[cur_pos : right_mark_pos]
                cur_text_length = right_mark_pos - cur_pos

                local_pos = 0
                while local_pos < cur_text_length:
                    local_pos_new = cur_text.find(u'　', local_pos)
                    if local_pos_new != -1: # 包含空格
                        passed_char_cnt = local_pos_new - local_pos
                        local_pos = local_pos_new + 1
                        y = y + int(passed_char_cnt * avg_height)
                        min_pos, min_sum_of_pixels = find_nearest_label_line(label_lines, y,
                                                                         binary_line, 0.25 * region_width)
                        if min_sum_of_pixels < 10000:
                            y = min_pos
                        else:
                            start1 = y - int(avg_height / 3)
                            if start1 < cur_start:
                                start1 = cur_start
                            end1 = y + int(avg_height / 3)
                            if end1 > end - 1:
                                end1 = end - 1
                            if binary_line[y] != 0:
                                min_pos = find_min_pos(binary_line, start1, y, end1)
                                y = min_pos + start1 - 1
                        y = find_label_line(label_lines, y + avg_height / 2) # 找到空白区域的bottom
                    else: # 没有空格
                        passed_char_cnt = cur_text_length - local_pos
                        local_pos = cur_text_length
                        y = y + int(passed_char_cnt * avg_height)
                        min_pos, min_sum_of_pixels = find_nearest_label_line(label_lines, y,
                                                                             binary_line, 0.25 * region_width)
                        if min_sum_of_pixels < 10000:
                            y = min_pos
                        else:
                            start1 = y - int(avg_height / 3)
                            if start1 < cur_start:
                                start1 = cur_start
                            end1 = y + int(avg_height / 3)
                            if end1 > end - 1:
                                end1 = end - 1
                            if binary_line[y] != 0:
                                min_pos = find_min_pos(binary_line, start1, y, end1)
                                y = min_pos + start1
                add_to_region_lst(text[cur_pos : right_mark_pos],
                                  line_region.left, line_region.right,
                                  cur_start, y,
                                  line_no, region_no,
                                  page_bar_no, region_lst)
                region_no = region_no + 1
                cur_start = y
            # right small
            right_end_mark_pos = text.find(u'>', right_mark_pos)
            if right_end_mark_pos != -1:
                if right_end_mark_pos != char_cnt - 1 and text[right_end_mark_pos + 1] == u'<':
                    # left , right 都存在
                    left_mark_pos = right_end_mark_pos + 1
                    left_end_mark_pos = text.find(u'>', left_mark_pos)
                    if left_end_mark_pos == char_cnt - 1:
                        y = end - 1
                    else:
                        right_small_cnt = right_end_mark_pos - right_mark_pos - 1
                        left_small_cnt = left_end_mark_pos - left_mark_pos - 1
                        passed_small_char_cnt = passed_small_char_cnt + max(right_small_cnt, left_small_cnt)

                        y = cur_start + int(max(right_small_cnt, left_small_cnt) * avg_small_height)
                        min_pos, min_sum_of_pixels = find_nearest_label_line(label_lines, y,
                                                                             binary_line, 0.25 * region_width)
                        if min_sum_of_pixels < 10000:
                            y = min_pos
                        else:
                            start1 = y - int(avg_small_height / 2)
                            if start1 < cur_start:
                                start1 = cur_start
                            end1 = y + int(avg_small_height / 2)
                            if end1 > end - 1:
                                end1 = end - 1
                            if binary_line[y] != 0:
                                min_pos = find_min_pos(binary_line, start1, y, end1)
                                y = min_pos + start1

                    # right region
                    add_to_region_lst(text[right_mark_pos + 1: right_end_mark_pos],
                                      line_region.left + region_width / 2, line_region.right,
                                      cur_start, y,
                                      line_no, region_no,
                                      page_bar_no, region_lst, u'<')
                    region_no = region_no + 1

                    # left region
                    add_to_region_lst(text[left_mark_pos + 1: left_end_mark_pos],
                                      line_region.left, line_region.left + region_width / 2,
                                      cur_start, y,
                                      line_no, region_no,
                                      page_bar_no, region_lst, u'<')
                    region_no = region_no + 1

                    cur_pos = left_end_mark_pos + 1
                else:
                    # 只存在right
                    if right_end_mark_pos == char_cnt - 1:
                        y = end - 1
                    else:
                        right_small_cnt = right_end_mark_pos - right_mark_pos - 1
                        passed_small_char_cnt = passed_small_char_cnt + (right_end_mark_pos - right_mark_pos - 1)
                        y = cur_start - 1 + int(right_small_cnt * avg_small_height)
                        min_pos, min_sum_of_pixels = find_nearest_label_line(label_lines, y,
                                                                             binary_line, 0.25 * region_width)
                        if min_sum_of_pixels < 10000:
                            y = min_pos
                        else:
                            start1 = y - int(avg_small_height / 2)
                            if start1 < cur_start:
                                start1 = cur_start
                            end1 = y + int(avg_small_height / 2)
                            if end1 > end - 1:
                                end1 = end - 1
                            if binary_line[y] != 0:
                                min_pos = find_min_pos(binary_line, start1, y, end1)
                                y = min_pos + start1

                    add_to_region_lst(text[right_mark_pos + 1: right_end_mark_pos],
                                          line_region.left + region_width / 2, line_region.right,
                                          cur_start, y,
                                          line_no, region_no,
                                          page_bar_no, region_lst, u'<')
                    region_no = region_no + 1

                    cur_pos = right_end_mark_pos + 1
                cur_start = y
        if cur_pos <= char_cnt - 1:
            add_to_region_lst(text[cur_pos: ],
                              line_region.left, line_region.right,
                              cur_start, end,
                              line_no, region_no,
                              page_bar_no, region_lst)

# 處理經過二值化的圖片
def layout_seg(image, page_text):
    if page_text.endswith(u'\r\n'):
        separator = u'\r\n'
    else:
        separator = u'\n'
    texts = page_text.rstrip(separator).split(separator)
    bw = binarisation(image)
    image_height, image_width = bw.shape
    bw = (1 - bw).astype('ubyte')

    label_image = label(bw, connectivity=2)
    line_region_lst = get_line_region_lst(label_image)

    region_lst = []
    line_idx = 0
    text_len = len(texts)
    page_bar_no = texts[0].strip()
    for i in range(1, text_len):
        text = texts[i].rstrip()
        if text:
            region_seg(image, bw, image_height, page_bar_no, i, line_region_lst[line_idx], text, region_lst)
            line_idx = line_idx + 1
        else:
            left = line_region_lst[line_idx].right
            right = line_region_lst[line_idx-1].left
            region = {
                u'text': text,
                u'left': left,
                u'right': right,
                u'top': 0,
                u'bottom': image_height,
                u'line_no': i,
                u'region_no': 1,
                u'page_bar_no': page_bar_no,
            }
            region_lst.append(region)
    return region_lst

if __name__ == '__main__':
    image = io.imread(sys.argv[1], 0)
    text = u''
    with open(sys.argv[2], 'r') as f:
        text = f.read().decode('utf-8')
    region_lst = layout_seg(image, text)
    output = json.dumps(region_lst, ensure_ascii=False, indent=True)
    print(output.encode('utf-8'))

    # plot
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
    colors = ['red', 'blue', 'darkgreen', 'm', 'c']
    color_index = 0
    for region in region_lst:
        top = region[u'top']
        bottom = region[u'bottom']
        left = region[u'left']
        right = region[u'right']
        rect = mpatches.Rectangle((left, top), right - left, bottom - top,
                                  fill=False, edgecolor=colors[color_index], linewidth=2)
        ax.add_patch(rect)
        color_index = (color_index + 1) % 5
    ax.imshow(image)
    ax.margins(x=0.02, y=0.02)
    plt.show()

