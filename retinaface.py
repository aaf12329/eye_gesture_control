import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pkg_resources")
import colorsys
import os

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFont
from torch.autograd import Variable

from retinaface_arcface.nets.retinaface import RetinaFace
from retinaface_arcface.utils.anchors import Anchors
from retinaface_arcface.utils.box_utils import (decode, decode_landm, letterbox_image,
                             non_max_suppression, retinaface_correct_boxes)
from retinaface_arcface.utils.config import cfg_mnet, cfg_re50


def preprocess_input(image):
    """image input"""
    image -= np.array((104, 117, 123),np.float32)
    return image

#   注意主干网络与预训练权重的对应
#   注意修改model_path和backbone

class Retinaface(object):

    _defaults = {
        "model_path"        : 'model_data/Retinaface_mobilenet0.25.pth',
        "backbone"          : 'mobilenet',
        "confidence"        : 0.9,
        "nms_iou"           : 0.2,  # 调整NMS阈值平衡精度速度
        "cuda"              : torch.cuda.is_available(),
        #   是否需要进行图像大小限制。
        #   开启后，会将输入图像的大小限制为input_shape。否则使用原图进行预测。
        #   可根据输入图像的大小自行调整input_shape，注意为32的倍数，如[640, 640, 3]
        "input_shape"       : [960, 960, 3],  # input image size
        "letterbox_image"   : True  #图像处理方法
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    #   初始化Retinaface
    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        if self.backbone == "mobilenet":
            self.cfg = cfg_mnet
        else:
            self.cfg = cfg_re50
        self.generate()
        if self.letterbox_image:
            self.anchors = Anchors(self.cfg, image_size=[self.input_shape[0], self.input_shape[1]]).get_anchors()

    #   载入模型
    def generate(self):
        self.net = RetinaFace(cfg=self.cfg, mode='eval').eval()

        #   载入模型与权值
        print('Loading weights into state dict...')
        state_dict = torch.load(self.model_path, map_location='cpu', weights_only=True)
        self.net.load_state_dict(state_dict)
        if self.cuda:
            os.environ["CUDA_VISIBLE_DEVICES"] = '0'
            self.net = nn.DataParallel(self.net)
            self.net = self.net.cuda()
        print('Finished!')


    #   检测图片
    def detect_image(self, image):
        #   对输入图像进行一个备份，后面用于绘图
        old_image = image.copy()

        image = np.array(image,np.float32)
        #   计算scale，用于将获得的预测框转换成原图的高宽
        scale = [np.shape(image)[1], np.shape(image)[0], np.shape(image)[1], np.shape(image)[0]]
        scale_for_landmarks = [np.shape(image)[1], np.shape(image)[0], np.shape(image)[1], np.shape(image)[0],
                                            np.shape(image)[1], np.shape(image)[0], np.shape(image)[1], np.shape(image)[0],
                                            np.shape(image)[1], np.shape(image)[0]]

        im_height, im_width, _ = np.shape(image)
        #   letterbox_image可以给图像增加灰条，实现不失真的resize
        if self.letterbox_image:
            image = np.array(letterbox_image(image, [self.input_shape[1], self.input_shape[0]]), np.float32)
        else:
            self.anchors = Anchors(self.cfg, image_size=(im_height, im_width)).get_anchors()
            
        with torch.no_grad():
            #   图片预处理，归一化。
            image = torch.from_numpy(preprocess_input(image).transpose(2, 0, 1)).unsqueeze(0)

            if self.cuda:
                self.anchors = self.anchors.cuda()
                image = image.cuda()

            loc, conf, landms = self.net(image)

            #   将预测结果进行解码
            boxes = decode(loc.data.squeeze(0), self.anchors, self.cfg['variance'])
            boxes = boxes.cpu().numpy()

            conf = conf.data.squeeze(0)[:,1:2].cpu().numpy()
            
            landms = decode_landm(landms.data.squeeze(0), self.anchors, self.cfg['variance'])
            landms = landms.cpu().numpy()

            boxes_conf_landms = np.concatenate([boxes, conf, landms],-1)
            boxes_conf_landms = non_max_suppression(boxes_conf_landms, self.confidence)
            if len(boxes_conf_landms)<=0:
                return [], []

            #   如果使用了letterbox_image的话，要把灰条的部分去除掉。
            if self.letterbox_image:
                boxes_conf_landms = retinaface_correct_boxes(boxes_conf_landms, \
                    np.array([self.input_shape[0], self.input_shape[1]]), np.array([im_height, im_width]))
            
        boxes_conf_landms[:,:4] = boxes_conf_landms[:,:4]*scale
        boxes_conf_landms[:,5:] = boxes_conf_landms[:,5:]*scale_for_landmarks
        b1 = []
        b2 = []
        for b in boxes_conf_landms:
            if b[4] < 0.9:
                continue
            text = "{:.4f}".format(b[4])
            b = list(map(int, b))
            b1.append(b[0:4])
            b2.append(np.array([[b[5], b[6]], [b[7], b[8]],[b[9], b[10]],[b[11], b[12]],[b[13], b[14]]]))

        return b1, b2