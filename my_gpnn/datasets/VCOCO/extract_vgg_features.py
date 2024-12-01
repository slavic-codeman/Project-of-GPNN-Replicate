"""
Created on Oct 12, 2017

@author: Siyuan Qi

Description of the file.

"""

import os
import pickle
import warnings

import numpy as np
import cv2
import torch
import torchvision.models
import torchvision.transforms as transforms
from pycocotools.coco import COCO
from . import vcoco_config,roi_pooling


# 定义特征模式
feature_mode = 'roi_vgg'

class Vgg16(torch.nn.Module):
    def __init__(self, last_layer=0, requires_grad=False):
        super(Vgg16, self).__init__()
        pretrained_vgg = torchvision.models.vgg16(pretrained=True)
        self.features = torch.nn.Sequential()
        for x in range(len(pretrained_vgg.features)):
            self.features.add_module(str(x), pretrained_vgg.features[x])

        if feature_mode == 'roi_vgg':
            self.classifier = torch.nn.Sequential()
            self.classifier.add_module(str(0), pretrained_vgg.classifier[0])

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x):
        x = self.features(x)
        if feature_mode == 'roi_vgg':
            x = x.view(x.size(0), -1)
            x = self.classifier(x)
        return x


def get_model():
    vgg16 = Vgg16(last_layer=1).cuda()
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return vgg16, transform


def combine_box(box1, box2):
    return np.hstack((np.minimum(box1[:2], box2[:2]), np.maximum(box1[2:], box2[2:])))


def get_info(paths, imageset):
    vcoco_feature_path = paths.data_root
    vcoco_path = os.path.join(vcoco_feature_path, '../v-coco')

    prefix = 'instances' if 'test' not in imageset else 'image_info'
    coco = COCO(os.path.join(vcoco_path, 'coco', 'annotations', prefix + '_' + imageset + '2014.json'))
    image_list = coco.getImgIds()
    image_list = coco.loadImgs(image_list)

    det_res_path = os.path.join(
        '/home/siyuan/data/HICO/hico_20160224_det/Deformable-ConvNets/output/rfcn_dcn/vcoco/vcoco_detect2/{}2014'.format(imageset),
        'COCO_{}2014_detections.pkl'.format(imageset)
    )
    feature_path = os.path.join(vcoco_feature_path, 'features_{}'.format(feature_mode))

    classes = ['__background__',  # always index 0
               'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
               'traffic_light', 'fire_hydrant', 'stop_sign', 'parking_meter', 'bench', 'bird', 'cat', 'dog', 'horse',
               'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports_ball', 'kite', 'baseball_bat', 'baseball_glove',
               'skateboard', 'surfboard', 'tennis_racket', 'bottle', 'wine_glass', 'cup', 'fork', 'knife', 'spoon',
               'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot_dog', 'pizza', 'donut',
               'cake', 'chair', 'couch', 'potted_plant', 'bed', 'dining_table', 'toilet', 'tv', 'laptop', 'mouse',
               'remote', 'keyboard', 'cell_phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book',
               'clock', 'vase', 'scissors', 'teddy_bear', 'hair_drier', 'toothbrush']

    return vcoco_path, det_res_path, feature_path, classes, image_list


def extract_features(paths, imageset):
    input_h, input_w = 244, 244
    feature_size = (7, 7)
    adaptive_max_pool = roi_pooling.AdaptiveMaxPool2d(*feature_size)

    vcoco_path, det_res_path, feature_path, classes, image_list = get_info(paths, imageset)
    if not os.path.exists(feature_path):
        os.makedirs(feature_path)
    vgg16, transform = get_model()

    # 读取检测结果
    with open(det_res_path, 'rb') as f:  # 改为 'rb' 模式以兼容 Python 3
        det_res = pickle.load(f)

    for i_image, img_info in enumerate(image_list):
        img_name = img_info['file_name']
        print(img_name)

        # 提取边界框和类别
        det_boxes_all = np.empty((0, 4))
        det_classes_all = list()
        for c in range(1, len(classes)):
            for detection in det_res[c][i_image]:
                if detection[4] > 0.7:
                    det_boxes_all = np.vstack((det_boxes_all, np.array(detection[:4])[np.newaxis, ...]))
                    det_classes_all.append(c)
        if len(det_classes_all) == 0:
            continue

        edge_classes = list()
        for person_i, person_c in enumerate(det_classes_all):
            if person_c == 1:
                for obj_i, obj_c in enumerate(det_classes_all):
                    if obj_c == 1:
                        continue
                    combined_box = combine_box(det_boxes_all[person_i, :], det_boxes_all[obj_i, :])
                    det_boxes_all = np.vstack((det_boxes_all, combined_box))
                    edge_classes.append(0)
        det_classes_all.extend(edge_classes)

        # 读取图片
        image_path = os.path.join(vcoco_path, 'coco/images', '{}2014'.format(imageset), img_name)
        assert os.path.exists(image_path)
        original_img = cv2.imread(image_path)
        original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)  # cv2 默认读取为 BGR 格式，需要转换为 RGB

        # 提取特征
        roi_features = np.zeros((det_boxes_all.shape[0], 4096))
        for i_box in range(det_boxes_all.shape[0]):
            roi = det_boxes_all[i_box, :].astype(int)
            roi_image = original_img[roi[1]:roi[3] + 1, roi[0]:roi[2] + 1, :]
            roi_image = cv2.resize(roi_image, (input_h, input_w), interpolation=cv2.INTER_LINEAR)
            roi_image = transform(roi_image)
            roi_image = roi_image.unsqueeze(0).cuda()
            roi_features[i_box, ...] = vgg16(roi_image).detach().cpu().numpy()

        np.save(os.path.join(feature_path, '{}_classes.npy'.format(img_name)), det_classes_all)
        np.save(os.path.join(feature_path, '{}_boxes.npy'.format(img_name)), det_boxes_all)
        np.save(os.path.join(feature_path, '{}_features.npy'.format(img_name)), roi_features)


def main():
    paths = vcoco_config.Paths()
    imagesets = ['train', 'val', 'test']
    for imageset in imagesets:
        extract_features(paths, imageset)


if __name__ == '__main__':
    main()
