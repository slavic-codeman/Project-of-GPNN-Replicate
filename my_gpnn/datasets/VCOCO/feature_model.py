"""
Created on Feb 26, 2018

@author: Siyuan Qi

Description of the file.

"""

import os
import random

import numpy as np
import imageio  # 替换 scipy.misc
import torch
import torchvision
import cv2

import metadata


def parse_classes(det_classes):
    obj_nodes = False
    human_num = 0
    obj_num = 0
    for i in range(det_classes.shape[0]):
        if not obj_nodes:
            if det_classes[i] == 1:
                human_num += 1
            else:
                obj_nodes = True
                obj_num += 1
        else:
            if det_classes[i] > 1:
                obj_num += 1
            else:
                break

    node_num = human_num + obj_num
    edge_num = det_classes.shape[0] - node_num
    return human_num, obj_num, edge_num


def combine_box(box1, box2):
    return np.hstack((np.minimum(box1[:2], box2[:2]), np.maximum(box1[2:], box2[2:])))


def get_intersection(box1, box2):
    return np.hstack((np.maximum(box1[:2], box2[:2]), np.minimum(box1[2:], box2[2:])))


def compute_area(box):
    side1 = box[2] - box[0]
    side2 = box[3] - box[1]
    return max(side1, 0) * max(side2, 0)  # 防止负值


def compute_iou(box1, box2):
    intersection_area = compute_area(get_intersection(box1, box2))
    iou = intersection_area / (compute_area(box1) + compute_area(box2) - intersection_area)
    return iou


def perturb_box(box):
    side1 = box[2] - box[0]
    side2 = box[3] - box[1]
    return box + (np.random.rand(4) - 0.5) * np.array([side1, side2, side1, side2]) / 3


def perturb_gt_box(box):
    while True:
        new_box = perturb_box(box)
        if compute_iou(new_box, box) > 0.7:
            return new_box


def get_valid_roi(original_img, roi):
    roi[0] = min(original_img.shape[1] - 1, max(0, roi[0]))
    roi[1] = min(original_img.shape[0] - 1, max(0, roi[1]))
    roi[2] = min(original_img.shape[1] - 1, max(0, roi[2]))
    roi[3] = min(original_img.shape[0] - 1, max(0, roi[3]))
    return roi


class VCOCO(torch.utils.data.Dataset):
    def __init__(self, root, input_imsize, transform, imageset):
        self.imageset = imageset
        self.vcoco_imageset = 'val' if imageset == 'test' else 'train'
        self.vcoco_feature_path = os.path.join(root, 'features_deformable')
        self.vcoco_path = os.path.join(root, '..', 'v-coco')
        self.imsize = input_imsize
        self.transform = transform

        self.coco = metadata.load_coco()
        self.vcoco_all = metadata.load_vcoco(f'vcoco_{imageset}')
        self.hoi_list = []
        for i_action, vcoco in enumerate(self.vcoco_all):
            vcoco = metadata.attach_gt_boxes(vcoco, self.coco)
            positive_index = np.where(vcoco['label'] == 1)[0].tolist()
            self.hoi_list.extend([(i_action, image_index) for image_index in positive_index])
        self.positive_num = len(self.hoi_list)
        self.negative_num = 0 if imageset == 'test' else 200

    def __getitem__(self, index):
        if index < self.positive_num:
            action_i, image_i = self.hoi_list[index]
            vcoco = self.vcoco_all[action_i]
            label = vcoco['action_name']
            img_name = self.coco.loadImgs(ids=[image_i])[0]['file_name']

            role_bbox = vcoco['role_bbox'][image_i, :].reshape((-1, 4))
            roi = perturb_gt_box(role_bbox[0])
            for j in range(1, len(vcoco['role_name'])):
                if not np.isnan(role_bbox[j, 0]):
                    roi = combine_box(roi, perturb_gt_box(role_bbox[j]))
        else:
            label = 'none'
            return None  # 无效条目

        roi = roi.astype(int)  # 替换 .astype(np.int)
        image_path = os.path.join(self.vcoco_path, 'coco/images', f'{self.vcoco_imageset}2014', img_name)
        assert os.path.exists(image_path)
        original_img = imageio.imread(image_path)  # 替换 scipy.misc
        roi = get_valid_roi(original_img, roi)
        roi_image = original_img[roi[1]:roi[3] + 1, roi[0]:roi[2] + 1]
        roi_image = cv2.resize(roi_image, self.imsize, interpolation=cv2.INTER_LINEAR)
        if self.imageset != 'test' and random.random() > 0.5:
            roi_image = np.fliplr(roi_image).copy()
        roi_image = self.transform(roi_image)

        label = torch.LongTensor([metadata.action_index[label]])
        return roi_image, label

    def __len__(self):
        return self.positive_num + self.negative_num


class Vgg16(torch.nn.Module):
    def __init__(self, num_classes=1000):
        super(Vgg16, self).__init__()
        self.features = torchvision.models.vgg16(pretrained=True).features
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(512 * 7 * 7, 4096),
            torch.nn.ReLU(True),
            torch.nn.Dropout(),
            torch.nn.Linear(4096, 4096),
            # torch.nn.ReLU(True),
            # torch.nn.Dropout(),
            # torch.nn.Linear(4096, num_classes),
            # torch.nn.Sigmoid()
        )
        self.last_layer = torch.nn.Sequential(
            torch.nn.ReLU(True),
            torch.nn.Dropout(),
            torch.nn.Linear(4096, num_classes),
        )
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        output = self.last_layer(x)
        return x, output

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, torch.nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


class Resnet152(torch.nn.Module):
    def __init__(self, num_classes=1000):
        super(Resnet152, self).__init__()
        # self.learn_modules = torch.nn.Sequential()
        # pretrained_resnet = torchvision.models.resnet152(pretrained=True)
        # for i, m in enumerate(pretrained_resnet.modules()):
        #     if isinstance(m, torch.nn.Linear):
        #         break
        #     self.learn_modules.add_module(str(i), m)
        self.learn_modules = torchvision.models.resnet152(pretrained=True)
        self.fc = torch.nn.Linear(1000, num_classes)
        # self.fc = torch.nn.Sequential(
        #     torch.nn.ReLU(True),
        #     # torch.nn.Dropout(),
        #     torch.nn.Linear(1000, num_classes),
        # )
        # self._initialize_weights()

    def forward(self, x):
        x = self.learn_modules(x)
        x = x.view(x.size(0), -1)
        output = self.fc(x)
        return x, output

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


class Densenet(torch.nn.Module):
    def __init__(self, num_classes=1000):
        super(Densenet, self).__init__()
        self.learn_modules = torchvision.models.densenet161(pretrained=True)
        self.fc = torch.nn.Linear(1000, num_classes)
        self._initialize_weights()

    def forward(self, x):
        x = self.learn_modules(x)
        x = x.view(x.size(0), -1)
        output = self.fc(x)
        return x, output

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
# 简化主函数，避免复杂的构造逻辑
def main():
    print("Feature Model Loaded Successfully")


if __name__ == '__main__':
    main()
