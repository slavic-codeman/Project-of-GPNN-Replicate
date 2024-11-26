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
import torch.autograd
import torchvision.models
import matplotlib.pyplot as plt
import torchvision.transforms
from pycocotools.coco import COCO
import vsrl_utils as vu
import vcoco_config
import roi_pooling
import feature_model
import metadata


def get_model(paths, feature_type):
    """
    获取不同类型的特征提取模型。
    :param paths: 配置路径
    :param feature_type: 特征提取模型类型 ('vgg', 'resnet', 'densenet')
    :return: 特征提取模型
    """
    if feature_type == 'vgg':
        feature_network = feature_model.Vgg16(num_classes=len(metadata.action_classes))
    elif feature_type == 'resnet':
        feature_network = feature_model.Resnet152(num_classes=len(metadata.action_classes))
    elif feature_type == 'densenet':
        feature_network = feature_model.Densenet(num_classes=len(metadata.action_classes))
    else:
        raise ValueError('Feature type not recognized')

    # 根据不同的模型类型，选择不同的并行策略和 GPU 设置
    if feature_type.startswith('alexnet') or feature_type.startswith('vgg'):
        feature_network.features = torch.nn.DataParallel(feature_network.features)
        feature_network.cuda()
    else:
        feature_network = torch.nn.DataParallel(feature_network).cuda()

    checkpoint_dir = os.path.join(paths.tmp_root, 'checkpoints', 'vcoco', 'finetune_{}'.format(feature_type))
    best_model_file = os.path.join(checkpoint_dir, 'model_best.pth')

    checkpoint = torch.load(best_model_file)
    feature_network.load_state_dict(checkpoint['state_dict'])
    return feature_network


def combine_box(box1, box2):
    """
    合并两个框的坐标。
    :param box1: 第一个框的坐标 (x1, y1, x2, y2)
    :param box2: 第二个框的坐标 (x1, y1, x2, y2)
    :return: 合并后的框坐标
    """
    return np.hstack((np.minimum(box1[:2], box2[:2]), np.maximum(box1[2:], box2[2:])))


def get_info(paths, imageset, feature_type):
    """
    获取图像和检测信息。
    :param paths: 配置路径
    :param imageset: 数据集类型（train, val, test）
    :param feature_type: 特征提取模型类型
    :return: 各类信息，包括路径、类别、图像列表等
    """
    vcoco_feature_path = paths.data_root
    vcoco_path = os.path.join(vcoco_feature_path, '../v-coco')

    prefix = 'instances' if 'test' not in imageset else 'image_info'
    coco = COCO(os.path.join(vcoco_path, 'coco', 'annotations', prefix + '_' + imageset + '2014.json'))
    image_list = coco.getImgIds()
    image_list = coco.loadImgs(image_list)

    det_res_path = os.path.join('/home/siyuan/data/HICO/hico_20160224_det/Deformable-ConvNets/output/rfcn_dcn/vcoco/vcoco_detect2/{}2014'.format(imageset),
                                'COCO_{}2014_detections.pkl'.format(imageset))
    feature_path = os.path.join(vcoco_feature_path, 'features_{}'.format(feature_type))

    # 物体类别列表
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


def extract_features(paths, imageset, vcoco_imageset):
    """
    提取特征的主函数。
    :param paths: 配置路径
    :param imageset: 数据集类型（train, val, test）
    :param vcoco_imageset: vcoco 数据集类型（train, test, val）
    """
    feature_type = 'resnet'
    input_h, input_w = 244, 244
    feature_size = (7, 7)
    adaptive_max_pool = roi_pooling.AdaptiveMaxPool2d(*feature_size)

    # 特征存储路径
    det_feature_path = os.path.join(paths.data_root, 'features_deformable')

    vcoco_path, det_res_path, feature_path, classes, image_list = get_info(paths, imageset, feature_type)
    if not os.path.exists(feature_path):
        os.makedirs(feature_path)

    # 获取特征提取模型
    feature_network = get_model(paths, feature_type)
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225]),
    ])

    # 读取检测结果
    with open(det_res_path, 'rb') as f:
        det_res = pickle.load(f)

    coco_from_vcoco = vu.load_coco()
    vcoco_all = vu.load_vcoco('vcoco_{}'.format(vcoco_imageset))
    for x in vcoco_all:
        x = vu.attach_gt_boxes(x, coco_from_vcoco)

    vcoco_image_ids = vcoco_all[0]['image_id'][:, 0].astype(int)

    for i_image, img_info in enumerate(image_list):
        img_id = img_info['id']
        indices_in_vcoco = np.where(vcoco_image_ids == img_id)[0].tolist()
        if len(indices_in_vcoco) == 0:
            continue

        img_name = img_info['file_name']
        print(img_name)

        try:
            det_classes_all = np.load(os.path.join(det_feature_path, '{}_classes.npy'.format(img_name)))
            det_boxes_all = np.load(os.path.join(det_feature_path, '{}_boxes.npy'.format(img_name)))
        except IOError:
            continue

        # 读取图像
        image_path = os.path.join(vcoco_path, 'coco/images', '{}2014'.format(imageset), img_name)
        assert os.path.exists(image_path)
        original_img = cv2.imread(image_path)

        # 获取每个检测框的特征
        if feature_type == 'vgg':
            roi_features = np.zeros((det_boxes_all.shape[0], 4096))
        elif feature_type == 'resnet':
            roi_features = np.zeros((det_boxes_all.shape[0], 1000))
        elif feature_type == 'densenet':
            roi_features = np.zeros((det_boxes_all.shape[0], 1000))
        else:
            raise ValueError('Feature type not recognized')

        for i_box in range(det_boxes_all.shape[0]):
            roi = det_boxes_all[i_box, :].astype(int)
            roi_image = original_img[roi[1]:roi[3] + 1, roi[0]:roi[2] + 1, :]
            roi_image = transform(cv2.resize(roi_image, (input_h, input_w), interpolation=cv2.INTER_LINEAR))
            roi_image = torch.tensor(roi_image).unsqueeze(0).cuda()
            feature, _ = feature_network(roi_image)
            roi_features[i_box, ...] = feature.data.cpu().numpy()

        np.save(os.path.join(feature_path, '{}_classes'.format(img_name)), det_classes_all)
        np.save(os.path.join(feature_path, '{}_boxes'.format(img_name)), det_boxes_all)
        np.save(os.path.join(feature_path, '{}_features'.format(img_name)), roi_features)


def main():
    """
    主函数，遍历不同的数据集类型进行特征提取。
    """
    paths = vcoco_config.Paths()
    imagesets = [('val', 'test'), ('train', 'train'), ('train', 'val')]
    for imageset, vcoco_imageset in imagesets:
        extract_features(paths, imageset, vcoco_imageset)


if __name__ == '__main__':
    main()
