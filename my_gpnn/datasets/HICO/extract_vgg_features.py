"""
Created on Oct 12, 2017

@author: Siyuan Qi

Description of the file.

"""

import os
import pickle

import numpy as np
from imageio import imread  # 替代 scipy.misc.imread
import cv2
import torch
import torchvision.models
import torchvision.transforms as transforms  # 简化导入
import hico_config
import roi_pooling


class Vgg16(torch.nn.Module):
    def __init__(self, last_layer=0, requires_grad=False):
        super(Vgg16, self).__init__()
        pretrained_vgg = torchvision.models.vgg16(pretrained=True)
        self.features = torch.nn.Sequential()
        for x in range(len(pretrained_vgg.features)):
            self.features.add_module(str(x), pretrained_vgg.features[x])

        self.classifier = torch.nn.Sequential()
        self.classifier.add_module(str(0), pretrained_vgg.classifier[0])

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def get_info(paths):
    hico_path = paths.data_root
    hico_voc_path = os.path.join(hico_path, 'Deformable-ConvNets/data/hico/VOC2007')
    feature_path = os.path.join(hico_path, 'processed', 'features_roi_vgg')

    image_list_file = os.path.join(hico_voc_path, 'ImageSets/Main/trainvaltest.txt')
    det_res_path = os.path.join(hico_path, 'Deformable-ConvNets/output/rfcn_dcn/hico/hico_detect/2007_trainvaltest',
                                'hico_detect_trainvaltest_detections.pkl')

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

    return hico_path, hico_voc_path, det_res_path, feature_path, classes, image_list_file


def get_model():
    vgg16 = Vgg16(last_layer=1).cuda()
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    return vgg16, transform


def combine_box(box1, box2):
    return np.hstack((np.minimum(box1[:2], box2[:2]), np.maximum(box1[2:], box2[2:])))


def extract_features(paths):
    input_h, input_w = 224, 224  # 修改为标准的输入尺寸
    feature_size = (7, 7)
    adaptive_max_pool = roi_pooling.AdaptiveMaxPool2d(*feature_size)

    hico_path, hico_voc_path, det_res_path, feature_path, classes, image_list_file = get_info(paths)
    os.makedirs(feature_path, exist_ok=True)  # 确保目录存在

    image_list = []
    with open(image_list_file) as f:
        for line in f.readlines():
            image_list.append(line.strip())

    vgg16, transform = get_model()

    # 读取检测结果
    with open(det_res_path, 'rb') as f:  # 修改文件打开模式为 'rb'
        det_res = pickle.load(f)

    for i_image, img_name in enumerate(image_list):
        print(img_name)

        # 提取检测的边界框和类别
        det_boxes_all = np.empty((0, 4))
        det_classes_all = []
        for c in range(1, len(classes)):
            for detection in det_res[c][i_image]:
                if detection[4] > 0.7:
                    det_boxes_all = np.vstack((det_boxes_all, np.expand_dims(detection[:4], axis=0)))  # 修复形状问题
                    det_classes_all.append(c)
        if not det_classes_all:
            continue

        edge_classes = []
        for person_i, person_c in enumerate(det_classes_all):
            if person_c == 1:
                for obj_i, obj_c in enumerate(det_classes_all):
                    if obj_c == 1:
                        continue
                    combined_box = combine_box(det_boxes_all[person_i, :], det_boxes_all[obj_i, :])
                    det_boxes_all = np.vstack((det_boxes_all, combined_box))
                    edge_classes.append(0)
        det_classes_all.extend(edge_classes)

        # 应用 VGG 计算 ROI 特征
        image_path = os.path.join(hico_voc_path, 'JPEGImages', img_name + '.jpg')
        if not os.path.exists(image_path):  # 替换 assert
            raise FileNotFoundError(f"Image path does not exist: {image_path}")

        original_img = imread(image_path)  # 替代 scipy.misc.imread
        roi_features = np.empty((det_boxes_all.shape[0], 4096))  # 确保形状正确

        for i_box in range(det_boxes_all.shape[0]):
            roi = det_boxes_all[i_box, :].astype(int)
            roi_image = original_img[roi[1]:roi[3] + 1, roi[0]:roi[2] + 1, :]
            roi_image = transform(cv2.resize(roi_image, (input_h, input_w), interpolation=cv2.INTER_LINEAR))
            roi_image = roi_image.unsqueeze(0).cuda()
            roi_features[i_box, ...] = vgg16(roi_image).data.cpu().numpy()

        np.save(os.path.join(feature_path, f'{img_name}_classes'), det_classes_all)
        np.save(os.path.join(feature_path, f'{img_name}_boxes'), det_boxes_all)
        np.save(os.path.join(feature_path, f'{img_name}_features'), roi_features)


def main():
    paths = hico_config.Paths()
    extract_features(paths)
    import parse_features
    parse_features.main()


if __name__ == '__main__':
    main()
