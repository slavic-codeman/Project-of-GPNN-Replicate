"""
Created on Feb 24, 2018

@author: Siyuan Qi

Description of the file.

"""

import os
import time
import pickle
import argparse
import warnings

import torch.utils.data
import numpy as np
import vsrl_utils as vu

import vcoco_config


class VCOCO(torch.utils.data.Dataset):
    def __init__(self, root, imageset):
        self.root = root
        self.coco = vu.load_coco()
        vcoco_all = vu.load_vcoco(f'vcoco_{imageset}')  # 使用 f-string 替代字符串格式化
        self.image_ids = vcoco_all[0]['image_id'][:, 0].astype(int).tolist()
        self.unique_image_ids = list(set(self.image_ids))

    def __getitem__(self, index):
        img_name = self.coco.loadImgs(ids=[self.unique_image_ids[index]])[0]['file_name']
        try:
            # 明确指定 encoding='latin1'，支持 Python 2 pickle 格式
            data = pickle.load(open(os.path.join(self.root, f'{img_name}.p'), 'rb'), encoding='latin1')
            edge_features = np.load(os.path.join(self.root, f'{img_name}_edge_features.npy'), allow_pickle=True)
            node_features = np.load(os.path.join(self.root, f'{img_name}_node_features.npy'), allow_pickle=True)
        except (IOError, FileNotFoundError):
            # 改进错误处理逻辑：返回 None 或其他默认值，而不是递归调用
            warnings.warn(f'Data missing for {img_name}')
            return None

        img_id = data['img_id']
        adj_mat = data['adj_mat']
        node_labels = data['node_labels']
        node_roles = data['node_roles']
        boxes = data['boxes']
        human_num = data['human_num']
        obj_num = data['obj_num']
        classes = data['classes']
        return edge_features, node_features, adj_mat, node_labels, node_roles, boxes, img_id, img_name, human_num, obj_num, classes

    def __len__(self):
        return len(self.unique_image_ids)


def main(args):
    start_time = time.time()

    subset = ['train', 'val', 'test']
    training_set = VCOCO(os.path.join(args.data_root, 'processed'), subset[0])
    print(f'{len(training_set)} instances.')  # 替换为 f-string
    sample = training_set[0]

    if sample is None:
        print('No valid data in the first instance.')
    else:
        edge_features, node_features, adj_mat, node_labels, node_roles, boxes, img_id, img_name, human_num, obj_num, classes = sample
        print(f'Processed first sample: {img_name}')

    print(f'Time elapsed: {time.time() - start_time:.2f}s')  # 替换为 f-string


def parse_arguments():
    paths = vcoco_config.Paths()
    parser = argparse.ArgumentParser(description='V-COCO dataset')
    parser.add_argument('--data-root', default=paths.data_root, help='Dataset path')
    parser.add_argument('--tmp-root', default=paths.tmp_root, help='Intermediate result path')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    main(args)
