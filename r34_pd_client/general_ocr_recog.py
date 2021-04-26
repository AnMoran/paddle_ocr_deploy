# encode    utf-8
# author    wangpengyuan
# time      2020-10-28
# mail      ceb_pengyuan@163.com
# use       the code to recognize general image 
import os
import sys
from paddle_serving_client import Client

import numpy as np
import string
import cv2
from shapely.geometry import Polygon
import pyclipper
from PIL import Image, ImageDraw, ImageFont
import time
import copy
import math

##############################################
#                                            #
#               检测阶段代码                  #
#                                            #
##############################################
#检测部分的前处理过程 长边缩放至960,同时保持长宽比
class DBProcessTest(object):
    """
    DB pre-process for Test mode
    """

    def __init__(self, params):
        super(DBProcessTest, self).__init__()
        self.resize_type = 0
        if 'test_image_shape' in params:
            self.image_shape = params['test_image_shape']
            # print(self.image_shape)
            self.resize_type = 1
        if 'max_side_len' in params:
            self.max_side_len = params['max_side_len']
        else:
            self.max_side_len = 2400

    def resize_image_type0(self, im):
        """
        resize image to a size multiple of 32 which is required by the network
        args:
            img(array): array with shape [h, w, c]
        return(tuple):
            img, (ratio_h, ratio_w)
        """
        max_side_len = self.max_side_len
        h, w, _ = im.shape

        resize_w = w
        resize_h = h

        # limit the max side
        if max(resize_h, resize_w) > max_side_len:
            if resize_h > resize_w:
                ratio = float(max_side_len) / resize_h
            else:
                ratio = float(max_side_len) / resize_w
        else:
            ratio = 1.
        resize_h = int(resize_h * ratio)
        resize_w = int(resize_w * ratio)
        if resize_h % 32 == 0:
            resize_h = resize_h
        elif resize_h // 32 <= 1:
            resize_h = 32
        else:
            resize_h = (resize_h // 32 - 1) * 32
        if resize_w % 32 == 0:
            resize_w = resize_w
        elif resize_w // 32 <= 1:
            resize_w = 32
        else:
            resize_w = (resize_w // 32 - 1) * 32
        try:
            if int(resize_w) <= 0 or int(resize_h) <= 0:
                return None, (None, None)
            im = cv2.resize(im, (int(resize_w), int(resize_h)))
        except:
            print(im.shape, resize_w, resize_h)
            sys.exit(0)
        ratio_h = resize_h / float(h)
        ratio_w = resize_w / float(w)
        return im, (ratio_h, ratio_w)

    def resize_image_type1(self, im):
        resize_h, resize_w = self.image_shape
        ori_h, ori_w = im.shape[:2]  # (h, w, c)
        im = cv2.resize(im, (int(resize_w), int(resize_h)))
        ratio_h = float(resize_h) / ori_h
        ratio_w = float(resize_w) / ori_w
        return im, (ratio_h, ratio_w)

    def normalize(self, im):
        img_mean = [0.485, 0.456, 0.406]
        img_std = [0.229, 0.224, 0.225]
        im = im.astype(np.float32, copy=False)
        im = im / 255
        im[:, :, 0] -= img_mean[0]
        im[:, :, 1] -= img_mean[1]
        im[:, :, 2] -= img_mean[2]
        im[:, :, 0] /= img_std[0]
        im[:, :, 1] /= img_std[1]
        im[:, :, 2] /= img_std[2]
        channel_swap = (2, 0, 1)
        im = im.transpose(channel_swap)
        return im

    def __call__(self, im):
        if self.resize_type == 0:
            im, (ratio_h, ratio_w) = self.resize_image_type0(im)
        else:
            im, (ratio_h, ratio_w) = self.resize_image_type1(im)
        im = self.normalize(im)
        im = im[np.newaxis, :]
        return [im, (ratio_h, ratio_w)]

def det_preprocess(img):
    preprocess_params = {'max_side_len': 960}
    preprocess_op = DBProcessTest(preprocess_params)
    im, ratio_list = preprocess_op(img)
    fetch = ["save_infer_model/scale_0.tmp_0"]
    if im is None:
        return None, 0
    return {
        "image": im[0]
    }, fetch, {
        "ratio_list": [ratio_list],
        "ori_im": img
    }

#检测部分的后处理过程，根据概率图
class DBPostProcess(object):
    """
    The post process for Differentiable Binarization (DB).
    """

    def __init__(self, params):
        self.thresh = params['thresh']
        self.box_thresh = params['box_thresh']
        self.max_candidates = params['max_candidates']
        self.unclip_ratio = params['unclip_ratio']
        self.min_size = 3
        self.dilation_kernel = np.array([[1, 1], [1, 1]])

    def boxes_from_bitmap(self, pred, _bitmap, dest_width, dest_height):
        '''
        _bitmap: single map with shape (1, H, W),
                whose values are binarized as {0, 1}
        '''

        bitmap = _bitmap
        height, width = bitmap.shape

        outs = cv2.findContours((bitmap * 255).astype(np.uint8), cv2.RETR_LIST,
                                cv2.CHAIN_APPROX_SIMPLE)
        if len(outs) == 3:
            img, contours, _ = outs[0], outs[1], outs[2]
        elif len(outs) == 2:
            contours, _ = outs[0], outs[1]

        num_contours = min(len(contours), self.max_candidates)
        boxes = np.zeros((num_contours, 4, 2), dtype=np.int16)
        scores = np.zeros((num_contours, ), dtype=np.float32)

        for index in range(num_contours):
            contour = contours[index]
            points, sside = self.get_mini_boxes(contour)
            if sside < self.min_size:
                continue
            points = np.array(points)
            score = self.box_score_fast(pred, points.reshape(-1, 2))
            if self.box_thresh > score:
                continue

            box = self.unclip(points).reshape(-1, 1, 2)
            box, sside = self.get_mini_boxes(box)
            if sside < self.min_size + 2:
                continue
            box = np.array(box)
            if not isinstance(dest_width, int):
                dest_width = dest_width.item()
                dest_height = dest_height.item()

            box[:, 0] = np.clip(
                np.round(box[:, 0] / width * dest_width), 0, dest_width)
            box[:, 1] = np.clip(
                np.round(box[:, 1] / height * dest_height), 0, dest_height)
            boxes[index, :, :] = box.astype(np.int16)
            scores[index] = score
        return boxes, scores

    def unclip(self, box):
        unclip_ratio = self.unclip_ratio
        poly = Polygon(box)
        distance = poly.area * unclip_ratio / poly.length
        offset = pyclipper.PyclipperOffset()
        offset.AddPath(box, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        expanded = np.array(offset.Execute(distance))
        return expanded

    def get_mini_boxes(self, contour):
        bounding_box = cv2.minAreaRect(contour)
        points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])

        index_1, index_2, index_3, index_4 = 0, 1, 2, 3
        if points[1][1] > points[0][1]:
            index_1 = 0
            index_4 = 1
        else:
            index_1 = 1
            index_4 = 0
        if points[3][1] > points[2][1]:
            index_2 = 2
            index_3 = 3
        else:
            index_2 = 3
            index_3 = 2

        box = [
            points[index_1], points[index_2], points[index_3], points[index_4]
        ]
        return box, min(bounding_box[1])

    def box_score_fast(self, bitmap, _box):
        h, w = bitmap.shape[:2]
        box = _box.copy()
        xmin = np.clip(np.floor(box[:, 0].min()).astype(np.int), 0, w - 1)
        xmax = np.clip(np.ceil(box[:, 0].max()).astype(np.int), 0, w - 1)
        ymin = np.clip(np.floor(box[:, 1].min()).astype(np.int), 0, h - 1)
        ymax = np.clip(np.ceil(box[:, 1].max()).astype(np.int), 0, h - 1)

        mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
        box[:, 0] = box[:, 0] - xmin
        box[:, 1] = box[:, 1] - ymin
        cv2.fillPoly(mask, box.reshape(1, -1, 2).astype(np.int32), 1)
        return cv2.mean(bitmap[ymin:ymax + 1, xmin:xmax + 1], mask)[0]

    def __call__(self, outs_dict, ratio_list):
        pred = outs_dict['maps']

        pred = pred[:, 0, :, :]
        segmentation = pred > self.thresh

        boxes_batch = []
        for batch_index in range(pred.shape[0]):
            height, width = pred.shape[-2:]

            mask = cv2.dilate(np.array(segmentation[batch_index]).astype(np.uint8), self.dilation_kernel)
            tmp_boxes, tmp_scores = self.boxes_from_bitmap(pred[batch_index], mask, width, height)

            boxes = []
            for k in range(len(tmp_boxes)):
                if tmp_scores[k] > self.box_thresh:
                    boxes.append(tmp_boxes[k])
            if len(boxes) > 0:
                boxes = np.array(boxes)

                ratio_h, ratio_w = ratio_list[batch_index]
                boxes[:, :, 0] = boxes[:, :, 0] / ratio_w
                boxes[:, :, 1] = boxes[:, :, 1] / ratio_h

            boxes_batch.append(boxes)
        return boxes_batch

det_postprocess_params = {
    "thresh": 0.3,
    "box_thresh": 0.5,
    "max_candidates": 1000,
    "unclip_ratio": 2.0,
    "min_size": 3
}

def order_points_clockwise(pts):
    """
    reference from: https://github.com/jrosebr1/imutils/blob/master/imutils/perspective.py
    # sort the points based on their x-coordinates
    """
    xSorted = pts[np.argsort(pts[:, 0]), :]

    # grab the left-most and right-most points from the sorted
    # x-roodinate points
    leftMost = xSorted[:2, :]
    rightMost = xSorted[2:, :]

    # now, sort the left-most coordinates according to their
    # y-coordinates so we can grab the top-left and bottom-left
    # points, respectively
    leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
    (tl, bl) = leftMost

    rightMost = rightMost[np.argsort(rightMost[:, 1]), :]
    (tr, br) = rightMost

    rect = np.array([tl, tr, br, bl], dtype="float32")
    return rect

def clip_det_res(points, img_height, img_width):
    for pno in range(points.shape[0]):
        points[pno, 0] = int(min(max(points[pno, 0], 0), img_width - 1))
        points[pno, 1] = int(min(max(points[pno, 1], 0), img_height - 1))
    return points

def filter_tag_det_res(dt_boxes, image_shape):
    img_height, img_width = image_shape[0:2]
    dt_boxes_new = []
    for box in dt_boxes:
        box = order_points_clockwise(box)
        box = clip_det_res(box, img_height, img_width)
        rect_width = int(np.linalg.norm(box[0] - box[1]))
        rect_height = int(np.linalg.norm(box[0] - box[3]))
        if rect_width <= 3 or rect_height <= 3:
            continue
        dt_boxes_new.append(box)
    dt_boxes = np.array(dt_boxes_new)
    return dt_boxes

#根据概率图获取boxes
def det_postprocess(outputs, args):
    outs_dict = {}
    outs_dict['maps'] = outputs[0]
    postprocess_op = DBPostProcess(det_postprocess_params)
    dt_boxes_list = postprocess_op(outs_dict, args["ratio_list"])
    dt_boxes = dt_boxes_list[0]
    dt_boxes = filter_tag_det_res(dt_boxes, args["ori_im"].shape)
    return dt_boxes

#由左上到右下排序boxes
def sorted_boxes(dt_boxes):
    """
    Sort text boxes in order from top to bottom, left to right
    args:
        dt_boxes(array):detected text boxes with shape [4, 2]
    return:
        sorted boxes(array) with shape [4, 2]
    """
    num_boxes = dt_boxes.shape[0]
    sorted_boxes = sorted(dt_boxes, key=lambda x: (x[0][1], x[0][0]))
    _boxes = list(sorted_boxes)

    for i in range(num_boxes - 1):
        if abs(_boxes[i + 1][0][1] - _boxes[i][0][1]) < 10 and \
                (_boxes[i + 1][0][0] < _boxes[i][0][0]):
            tmp = _boxes[i]
            _boxes[i] = _boxes[i + 1]
            _boxes[i + 1] = tmp
    return _boxes

#获取转正后的文本行
def get_rotate_crop_image(img, points):
    '''
    img_height, img_width = img.shape[0:2]
    left = int(np.min(points[:, 0]))
    right = int(np.max(points[:, 0]))
    top = int(np.min(points[:, 1]))
    bottom = int(np.max(points[:, 1]))
    img_crop = img[top:bottom, left:right, :].copy()
    points[:, 0] = points[:, 0] - left
    points[:, 1] = points[:, 1] - top
    '''
    img_crop_width = int(
        max(
            np.linalg.norm(points[0] - points[1]),
            np.linalg.norm(points[2] - points[3])))
    img_crop_height = int(
        max(
            np.linalg.norm(points[0] - points[3]),
            np.linalg.norm(points[1] - points[2])))
    pts_std = np.float32([[0, 0], [img_crop_width, 0],
                            [img_crop_width, img_crop_height],
                            [0, img_crop_height]])
    M = cv2.getPerspectiveTransform(points, pts_std)
    dst_img = cv2.warpPerspective(
        img,
        M, (img_crop_width, img_crop_height),
        borderMode=cv2.BORDER_REPLICATE,
        flags=cv2.INTER_CUBIC)
    dst_img_height, dst_img_width = dst_img.shape[0:2]
    if dst_img_height * 1.0 / dst_img_width >= 1.5:
        dst_img = np.rot90(dst_img)
    return dst_img

#保存文本行
def print_draw_crop_rec_res(img_crop_list):
    bbox_num = len(img_crop_list)
    for bno in range(bbox_num):
        cv2.imwrite("./rect_imgs/img_crop_%d.jpg" % bno, img_crop_list[bno])

##############################################
#                                            #
#               识别阶段代码                  #
#                                            #
##############################################

# 识别的前处理
# resize image to 3,32,W
def resize_norm_img(img, max_wh_ratio):
    rec_image_shape = "3, 32, 320"
    rec_image_shape = [int(v) for v in rec_image_shape.split(",")]
    imgC, imgH, imgW = rec_image_shape
    assert imgC == img.shape[2]
    wh_ratio = max(max_wh_ratio, imgW * 1.0 / imgH)
    character_type = 'ch'
    if character_type == "ch":
        imgW = int((32 * wh_ratio))
    h, w = img.shape[:2]
    ratio = w / float(h)
    if math.ceil(imgH * ratio) > imgW:
        resized_w = imgW
    else:
        resized_w = int(math.ceil(imgH * ratio))
    resized_image = cv2.resize(img, (resized_w, imgH))
    resized_image = resized_image.astype('float32')
    resized_image = resized_image.transpose((2, 0, 1)) / 255
    resized_image -= 0.5
    resized_image /= 0.5
    padding_im = np.zeros((imgC, imgH, imgW), dtype=np.float32)
    padding_im[:, :, 0:resized_w] = resized_image
    return padding_im

# 处理一个batch的图像
def rec_preprocess(img_list):
    img_num = len(img_list)
    args = {}
    # Calculate the aspect ratio of all text bars
    width_list = []
    for img in img_list:
        width_list.append(img.shape[1] / float(img.shape[0]))
    indices = np.argsort(np.array(width_list))
    args["indices"] = indices
    beg_img_no = 0
    end_img_no = img_num
    norm_img_batch = []
    max_wh_ratio = 0
    for ino in range(beg_img_no, end_img_no):
        h, w = img_list[indices[ino]].shape[0:2]
        wh_ratio = w * 1.0 / h
        max_wh_ratio = max(max_wh_ratio, wh_ratio)
    for ino in range(beg_img_no, end_img_no):
        # loss type is ctc
        norm_img = resize_norm_img(img_list[indices[ino]],
                                        max_wh_ratio)
        norm_img = norm_img[np.newaxis, :]
        norm_img_batch.append(norm_img)

    norm_img_batch = np.concatenate(norm_img_batch, axis=0)
    if img_num > 1:
        feed = [{
            "image": norm_img_batch[x]
        } for x in range(norm_img_batch.shape[0])]
    else:
        feed = {"image": norm_img_batch[0]}
    #fetch 
    fetch = ["ctc_greedy_decoder_0.tmp_0", "softmax_0.tmp_0"]
    return feed, fetch, args


# 识别的后处理
# 解码器参数
rec_char_ops_params = {
    "character_type": 'ch',
    "character_dict_path": "./general_ocr_config/ppocr_keys_v1.txt",
    "use_space_char": True,
    "max_text_length": 25,
    'loss_type' : 'ctc',
}

# 解码器
class CharacterOps(object):
    """
    Convert between text-label and text-index
    Args:
        config: config from yaml file
    """

    def __init__(self, config):
        self.character_type = config['character_type']
        self.loss_type = config['loss_type']
        self.max_text_len = config['max_text_length']
        # use the default dictionary(36 char)
        if self.character_type == "en":
            self.character_str = "0123456789abcdefghijklmnopqrstuvwxyz"
            dict_character = list(self.character_str)
        # use the custom dictionary
        elif self.character_type in [
                "ch", 'japan', 'korean', 'french', 'german'
        ]:
            character_dict_path = config['character_dict_path']
            add_space = False
            if 'use_space_char' in config:
                add_space = config['use_space_char']
            self.character_str = ""
            with open(character_dict_path, "rb") as fin:
                lines = fin.readlines()
                for line in lines:
                    line = line.decode('utf-8').strip("\n").strip("\r\n")
                    self.character_str += line
            if add_space:
                self.character_str += " "
            dict_character = list(self.character_str)
        elif self.character_type == "en_sensitive":
            # same with ASTER setting (use 94 char).
            self.character_str = string.printable[:-6]
            dict_character = list(self.character_str)
        else:
            self.character_str = None
        assert self.character_str is not None, \
            "Nonsupport type of the character: {}".format(self.character_str)
        self.beg_str = "sos"
        self.end_str = "eos"
        # add start and end str for attention
        if self.loss_type == "attention":
            dict_character = [self.beg_str, self.end_str] + dict_character
        elif self.loss_type == "srn":
            dict_character = dict_character + [self.beg_str, self.end_str]
        # create char dict
        self.dict = {}
        for i, char in enumerate(dict_character):
            self.dict[char] = i
        self.character = dict_character

    def encode(self, text):
        """
        convert text-label into text-index.
        Args:
            text: text labels of each image. [batch_size]
        Return:
            text: concatenated text index for CTCLoss.
                    [sum(text_lengths)] = [text_index_0 + text_index_1 + ... + text_index_(n - 1)]
        """
        # Ignore capital
        if self.character_type == "en":
            text = text.lower()

        text_list = []
        for char in text:
            if char not in self.dict:
                continue
            text_list.append(self.dict[char])
        text = np.array(text_list)
        return text

    def decode(self, text_index, is_remove_duplicate=False):
        """
        convert text-index into text-label.
        Args:
            text_index: text index for each image
            is_remove_duplicate: Whether to remove duplicate characters,
                                 The default is False
        Return:
            text: text label
        """
        char_list = []
        char_num = self.get_char_num()

        if self.loss_type == "attention":
            beg_idx = self.get_beg_end_flag_idx("beg")
            end_idx = self.get_beg_end_flag_idx("end")
            ignored_tokens = [beg_idx, end_idx]
        else:
            ignored_tokens = [char_num]

        for idx in range(len(text_index)):
            if text_index[idx] in ignored_tokens:
                continue
            if is_remove_duplicate:
                if idx > 0 and text_index[idx - 1] == text_index[idx]:
                    continue
            char_list.append(self.character[int(text_index[idx])])
        text = ''.join(char_list)
        return text

    def get_char_num(self):
        """
        Get character num
        """
        return len(self.character)

    def get_beg_end_flag_idx(self, beg_or_end):
        if self.loss_type == "attention":
            if beg_or_end == "beg":
                idx = np.array(self.dict[self.beg_str])
            elif beg_or_end == "end":
                idx = np.array(self.dict[self.end_str])
            else:
                assert False, "Unsupport type %s in get_beg_end_flag_idx"\
                    % beg_or_end
            return idx
        else:
            err = "error in get_beg_end_flag_idx when using the loss %s"\
                % (self.loss_type)
            assert False, err



# 获取识别结果
def rec_postprocess(outputs, args):
    loss_type = 'ctc'
    if loss_type == "ctc":
        rec_idx_batch = outputs[0]
        predict_batch = outputs[1]
        rec_idx_lod = args["ctc_greedy_decoder_0.tmp_0.lod"]
        predict_lod = args["softmax_0.tmp_0.lod"]
        indices = args["indices"]
        rec_res = [['', 0.0]] * (len(rec_idx_lod) - 1)
        for rno in range(len(rec_idx_lod) - 1):
            beg = rec_idx_lod[rno]
            end = rec_idx_lod[rno + 1]
            rec_idx_tmp = rec_idx_batch[beg:end, 0]
            char_ops = CharacterOps(rec_char_ops_params)
            preds_text = char_ops.decode(rec_idx_tmp)
            beg = predict_lod[rno]
            end = predict_lod[rno + 1]
            probs = predict_batch[beg:end, :]
            ind = np.argmax(probs, axis=1)
            blank = probs.shape[1]
            valid_ind = np.where(ind != (blank - 1))[0]
            if len(valid_ind) == 0:
                continue
            score = np.mean(probs[valid_ind, ind[valid_ind]])
            rec_res[indices[rno]] = [preds_text, score]
    return rec_res


##############################################
#                                            #
#               输出阶段代码                  #
#                                            #
##############################################
#将识别结果和原图打印到一张图上
def resize_img(img, input_size=600):
    """
    resize img and limit the longest side of the image to input_size
    """
    img = np.array(img)
    im_shape = img.shape
    im_size_max = np.max(im_shape[0:2])
    im_scale = float(input_size) / float(im_size_max)
    im = cv2.resize(img, None, None, fx=im_scale, fy=im_scale)
    return im

def str_count(s):
    """
    Count the number of Chinese characters,
    a single English character and a single number
    equal to half the length of Chinese characters.

    args:
        s(string): the input of string
    return(int):
        the number of Chinese characters
    """
    import string
    count_zh = count_pu = 0
    s_len = len(s)
    en_dg_count = 0
    for c in s:
        if c in string.ascii_letters or c.isdigit() or c.isspace():
            en_dg_count += 1
        elif c.isalpha():
            count_zh += 1
        else:
            count_pu += 1
    return s_len - math.ceil(en_dg_count / 2)

def text_visual(texts,
                scores,
                img_h=400,
                img_w=600,
                threshold=0.,
                font_path="./doc/simfang.ttf"):
    """
    create new blank img and draw txt on it
    args:
        texts(list): the text will be draw
        scores(list|None): corresponding score of each txt
        img_h(int): the height of blank img
        img_w(int): the width of blank img
        font_path: the path of font which is used to draw text
    return(array):

    """
    if scores is not None:
        assert len(texts) == len(
            scores), "The number of txts and corresponding scores must match"

    def create_blank_img():
        blank_img = np.ones(shape=[img_h, img_w], dtype=np.int8) * 255
        blank_img[:, img_w - 1:] = 0
        blank_img = Image.fromarray(blank_img).convert("RGB")
        draw_txt = ImageDraw.Draw(blank_img)
        return blank_img, draw_txt

    blank_img, draw_txt = create_blank_img()

    font_size = 20
    txt_color = (0, 0, 0)
    font = ImageFont.truetype(font_path, font_size, encoding="utf-8")

    gap = font_size + 5
    txt_img_list = []
    count, index = 1, 0
    for idx, txt in enumerate(texts):
        index += 1
        if scores[idx] < threshold or math.isnan(scores[idx]):
            index -= 1
            continue
        first_line = True
        while str_count(txt) >= img_w // font_size - 4:
            tmp = txt
            txt = tmp[:img_w // font_size - 4]
            if first_line:
                new_txt = str(index) + ': ' + txt
                first_line = False
            else:
                new_txt = '    ' + txt
            draw_txt.text((0, gap * count), new_txt, txt_color, font=font)
            txt = tmp[img_w // font_size - 4:]
            if count >= img_h // gap - 1:
                txt_img_list.append(np.array(blank_img))
                blank_img, draw_txt = create_blank_img()
                count = 0
            count += 1
        if first_line:
            new_txt = str(index) + ': ' + txt + '   ' + '%.3f' % (scores[idx])
        else:
            new_txt = "  " + txt + "  " + '%.3f' % (scores[idx])
        draw_txt.text((0, gap * count), new_txt, txt_color, font=font)
        # whether add new blank img or not
        if count >= img_h // gap - 1 and idx + 1 < len(texts):
            txt_img_list.append(np.array(blank_img))
            blank_img, draw_txt = create_blank_img()
            count = 0
        count += 1
    txt_img_list.append(np.array(blank_img))
    if len(txt_img_list) == 1:
        blank_img = np.array(txt_img_list[0])
    else:
        blank_img = np.concatenate(txt_img_list, axis=1)
    return np.array(blank_img)

def draw_ocr(image,
             boxes,
             txts=None,
             scores=None,
             drop_score=0.5,
             font_path="./general_ocr_config/simfang.ttf"):
    """
    Visualize the results of OCR detection and recognition
    args:
        image(Image|array): RGB image
        boxes(list): boxes with shape(N, 4, 2)
        txts(list): the texts
        scores(list): txxs corresponding scores
        drop_score(float): only scores greater than drop_threshold will be visualized
        font_path: the path of font which is used to draw text
    return(array):
        the visualized img
    """
    image_ori = image.copy()

    if scores is None:
        scores = [1] * len(boxes)
    box_num = len(boxes)
    for i in range(box_num):
        if scores is not None and (scores[i] < drop_score or
                                   math.isnan(scores[i])):
            continue
        box = np.reshape(np.array(boxes[i]), [-1, 1, 2]).astype(np.int64)
        print(box)
        image = cv2.polylines(np.array(image), [box], True, (255, 0, 0), 2)
    image = resize_img(image, input_size=600)

    # if txts is not None:
    #     img = np.array(resize_img(image, input_size=600))
    #     txt_img = text_visual(
    #         txts,
    #         scores,
    #         img_h=img.shape[0],
    #         img_w=600,
    #         threshold=drop_score,
    #         font_path=font_path)
    #     img_with_text = np.concatenate([np.array(img), np.array(txt_img)], axis=1)
    #     return img,img_with_text
    image_ori = resize_img(image_ori, input_size=600)
    return image,image_ori
    
#定义输出给主函数的接口函数

def check_general_ocr(input_pic_path,output_pic_600_path,output_pic_boxes_path,det_ip_port,rec_ip_port):
    """
    Convert between text-label and text-index
    Args:
        input_pic_path: Input image path
        output_pic_boxes_path：Output image path ,with red boxes on text area
        det_ip_port: The det serving port to listen
        rec_ip_port: The rec serving port to listen
    return:
        text_list: Boxes recognition result
        score_list: The confidence of boxes recognition result 
    """
    #初始化检测和识别
    det_client = Client()
    det_client.load_client_config("./general_ocr_config/det_infer_client/serving_client_conf.prototxt")
    det_client.connect(det_ip_port)

    #start rec Client
    rec_client = Client()
    rec_client.load_client_config("./general_ocr_config/rec_infer_client/serving_client_conf.prototxt")
    rec_client.connect(rec_ip_port)
    img = cv2.imread(input_pic_path)
    #保存一张用来存的图为RGB格式
    image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    #开始检测
    #前处理
    feed, fetch, tmp_args = det_preprocess(img)
    #推理
    fetch_map = det_client.predict(feed, fetch)
    outputs = [fetch_map[x] for x in fetch]

    #后处理
    dt_boxes = det_postprocess(outputs, tmp_args)
    print(dt_boxes.shape)
    #裁剪出框
    img_crop_list = []
    dt_boxes = sorted_boxes(dt_boxes)
    for bno in range(len(dt_boxes)):
        tmp_box = copy.deepcopy(dt_boxes[bno])
        img_crop = get_rotate_crop_image(img, tmp_box)
        img_crop_list.append(img_crop)

    #以batch为30开始识别
    batch_size = 12
    batch_num = len(img_crop_list) // batch_size + 1
    text_list = []
    score_list =[]
    if len(img_crop_list) != 0:
        for i in range(batch_num):
            if i == batch_num-1:
                img_batch = img_crop_list[i*batch_size:]
            else :
                img_batch = img_crop_list[i*batch_size:(i+1)*batch_size]
            if len(img_batch) == 0:
                continue
            feed, fetch, tmp_args = rec_preprocess(img_batch)

            #推理
            fetch_map = rec_client.predict(feed, fetch)
            outputs = [fetch_map[x] for x in fetch]
            for x in fetch_map.keys():
                if ".lod" in x:
                    tmp_args[x] = fetch_map[x]
            #后处理
            rec_res = rec_postprocess(outputs, tmp_args)
            for x in rec_res:
                text_list.append(x[0])
                score_list.append(x[1])
    draw_image,image_600 = draw_ocr(image,dt_boxes,text_list,score_list)
    cv2.imwrite(output_pic_boxes_path,draw_image[:, :, ::-1])
    cv2.imwrite(output_pic_600_path,image_600[:, :, ::-1])
    return text_list,score_list       

def general_ocr_client(img,det_client,rec_client):
    '''
    General_ocr_rec function
    Args:
        img: Input image,opencv BGR888
        det_ip_port: The det serving port to listen
        rec_ip_port: The rec serving port to listen
    return:
        dt_boxes:detection boxes list
        text_list: Boxes recognition result
        score_list: The confidence of boxes recognition result 
    '''
    #开始检测
    #前处理
    feed, fetch, tmp_args = det_preprocess(img)
    # print(feed, fetch, tmp_args)
    #推理
    fetch_map = det_client.predict(feed, fetch)
    # print('*'*100)
    # print(fetch_map)
    # print('*'*100)
    outputs = [fetch_map[x] for x in fetch]

    #后处理
    dt_boxes = det_postprocess(outputs, tmp_args)
    # print(dt_boxes.shape)
    #裁剪出框
    img_crop_list = []
    dt_boxes = sorted_boxes(dt_boxes)
    for bno in range(len(dt_boxes)):
        tmp_box = copy.deepcopy(dt_boxes[bno])
        img_crop = get_rotate_crop_image(img, tmp_box)
        img_crop_list.append(img_crop)

    #以batch为30开始识别
    batch_size = 30
    batch_num = len(img_crop_list) // batch_size + 1
    text_list = []
    score_list =[]
    for i in range(batch_num ):
        if i == (batch_num-1):
            img_batch = img_crop_list[i*batch_size:]
        else :
            img_batch = img_crop_list[i*batch_size:(i+1)*batch_size]
        if(len(img_batch)==0):
            continue
        feed, fetch, tmp_args = rec_preprocess(img_batch)

        #推理
        fetch_map = rec_client.predict(feed, fetch)
        # print(fetch_map)
        outputs = [fetch_map[x] for x in fetch]
        for x in fetch_map.keys():
            if ".lod" in x:
                # print(x),fetch_map[x]
                tmp_args[x] = fetch_map[x]
        #后处理
        rec_res = rec_postprocess(outputs, tmp_args)
        for x in rec_res:
            text_list.append(x[0])
            score_list.append(x[1])
    return dt_boxes,text_list,score_list

def general_ocr_port(img,det_ip_port,rec_ip_port):
    '''
    General_ocr_rec function
    Args:
        img: Input image,opencv BGR888
        det_ip_port: The det serving port to listen
        rec_ip_port: The rec serving port to listen
    return:
        dt_boxes:detection boxes list
        text_list: Boxes recognition result
        score_list: The confidence of boxes recognition result 
    '''
    #开始检测
    #前处理
    #初始化检测和识别
    det_client = Client()
    det_client.load_client_config("./general_ocr_config/det_infer_client/serving_client_conf.prototxt")
    det_client.connect(det_ip_port)

    feed, fetch, tmp_args = det_preprocess(img)
    #推理
    fetch_map = det_client.predict(feed, fetch)
    outputs = [fetch_map[x] for x in fetch]

    #后处理
    dt_boxes = det_postprocess(outputs, tmp_args)
    # print(dt_boxes.shape)
    #裁剪出框
    img_crop_list = []
    dt_boxes = sorted_boxes(dt_boxes)
    for bno in range(len(dt_boxes)):
        tmp_box = copy.deepcopy(dt_boxes[bno])
        img_crop = get_rotate_crop_image(img, tmp_box)
        img_crop_list.append(img_crop)

    #以batch为30开始识别
    batch_size = 8
    batch_num = len(img_crop_list) // batch_size + 1
    text_list = []
    score_list =[]
    if len(img_crop_list)==0:
        return text_list,score_list
    #start rec Client
    rec_client = Client()
    rec_client.load_client_config("./general_ocr_config/rec_infer_client/serving_client_conf.prototxt")
    rec_client.connect(rec_ip_port)
    feed, fetch, tmp_args = det_preprocess(img)
    
    for i in range(batch_num):
        if i == (batch_num-1):
            img_batch = img_crop_list[i*batch_size:]
        else :
            img_batch = img_crop_list[i*batch_size:(i+1)*batch_size]
        if(len(img_batch)==0):
            continue
        feed, fetch, tmp_args = rec_preprocess(img_batch)

        #推理
        fetch_map = rec_client.predict(feed, fetch)
        # print(fetch_map)
        outputs = [fetch_map[x] for x in fetch]
        for x in fetch_map.keys():
            if ".lod" in x:
                # print(x),fetch_map[x]
                tmp_args[x] = fetch_map[x]
        #后处理
        rec_res = rec_postprocess(outputs, tmp_args)
        for x in rec_res:
            text_list.append(x[0])
            score_list.append(x[1])
    rec_client.release()
    det_client.release()
    return dt_boxes,text_list,score_list

if __name__ == "__main__":


    det_ip_port = ['127.0.0.1:9293']
    rec_ip_port = ['127.0.0.1:9292']

    det_client = Client()
    det_client.load_client_config("./general_ocr_config/det_infer_client/serving_client_conf.prototxt")
    det_client.connect(det_ip_port)

    #start rec Client
    rec_client = Client()
    rec_client.load_client_config("./general_ocr_config/rec_infer_client/serving_client_conf.prototxt")
    rec_client.connect(rec_ip_port)
    input_pic_path = './02.jpg'
    img = cv2.imread(input_pic_path)
    dt_boxes,text_list,score_list = general_ocr_client(img,det_client,rec_client)
    print(text_list)
