"""
Created on Mar 13, 2017

@author: Siyuan Qi

Description of the file: Parsing and feature extraction for HICO dataset.
"""

"""
读取anno_box.mat?
"""


import os
import time
import pickle
import numpy as np
import scipy.io
from . import hico_config
from . import metadata


def parse_classes(det_classes):
    """
    Parse detection classes to compute numbers of humans, objects, and edges.

    Args:
        det_classes (ndarray): Detected class IDs.

    Returns:
        tuple: (human_num, obj_num, edge_num)
    """
    obj_nodes = False
    human_num, obj_num = 0, 0

    for det_class in det_classes:
        if not obj_nodes:
            if det_class == 1:
                human_num += 1
            else:
                obj_nodes = True
                obj_num += 1
        else:
            if det_class > 1:
                obj_num += 1
            else:
                break

    node_num = human_num + obj_num
    edge_num = det_classes.shape[0] - node_num
    return human_num, obj_num, edge_num


def get_intersection(box1, box2):
    return np.hstack((np.maximum(box1[:2], box2[:2]), np.minimum(box1[2:], box2[2:])))


def compute_area(box):
    return max(0, box[2] - box[0]) * max(0, box[3] - box[1])


def get_node_index(classname, bbox, det_classes, det_boxes, node_num):
    """
    Find the node index with the highest IoU matching the given class and bbox.

    Args:
        classname (str): Target class name.
        bbox (list): Bounding box coordinates [x1, y1, x2, y2].
        det_classes (ndarray): Detected class IDs.
        det_boxes (ndarray): Detected bounding boxes.
        node_num (int): Number of nodes.

    Returns:
        int: Node index or -1 if no match found.
    """
    bbox = np.array(bbox, dtype=np.float32)
    max_iou = 0.5
    max_iou_index = -1

    for i_node in range(node_num):
        if classname == metadata.hico_classes[metadata.coco_to_hico[det_classes[i_node]]]:
            intersection_area = compute_area(get_intersection(bbox, det_boxes[i_node]))
            iou = intersection_area / (compute_area(bbox) + compute_area(det_boxes[i_node]) - intersection_area)
            if iou > max_iou:
                max_iou = iou
                max_iou_index = i_node

    return max_iou_index


def read_features(data_root, tmp_root, bbox, list_action):
    """
    Parse features from annotations and save processed data.

    Args:
        data_root (str): Root directory for data.
        tmp_root (str): Temporary directory for output.
        bbox (dict): Bounding box annotations.
        list_action (dict): HOI action annotations.
    """
    save_data_path = os.path.join(data_root, 'processed', 'hico_data_background_49')
    feature_path = os.path.join(data_root, 'processed', 'features_background_49')

    if not os.path.exists(save_data_path):
        os.makedirs(save_data_path)

    for i_image in range(bbox['filename'].shape[1]):
        filename = os.path.splitext(bbox['filename'][0, i_image][0])[0]
        print(f"Processing {filename}...")

        try:
            det_classes = np.load(os.path.join(feature_path, f"{filename}_classes.npy"))
            det_boxes = np.load(os.path.join(feature_path, f"{filename}_boxes.npy"))
            det_features = np.load(os.path.join(feature_path, f"{filename}_features.npy"))
        except IOError:
            print(f"Skipping {filename}, missing files.")
            continue

        # Parsing and saving features
        human_num, obj_num, edge_num = parse_classes(det_classes)
        node_num = human_num + obj_num
        assert edge_num == human_num * obj_num

        edge_features = np.zeros((node_num, node_num, 49))
        node_features = np.zeros((node_num, 98))
        adj_mat = np.zeros((node_num, node_num))
        node_labels = np.zeros((node_num, len(metadata.action_classes)))

        # Node and edge features
        for i_node in range(node_num):
            if i_node < human_num:
                node_features[i_node, :49] = det_features[i_node].flatten()
            else:
                node_features[i_node, 49:] = det_features[i_node].flatten()

        for i_human in range(human_num):
            for i_obj in range(obj_num):
                edge_features[i_human, human_num + i_obj] = det_features[node_num + i_human * obj_num + i_obj].flatten()

        # Save processed data
        instance = {
            'human_num': human_num,
            'obj_num': obj_num,
            'boxes': det_boxes,
            'classes': det_classes,
            'adj_mat': adj_mat,
            'node_labels': node_labels,
        }
        pickle.dump(instance, open(os.path.join(save_data_path, f"{filename}.p"), 'wb'))


def collect_data(paths):
    anno_bbox = scipy.io.loadmat(os.path.join(paths.data_root, 'anno_bbox.mat'))
    bbox_train = anno_bbox['bbox_train']
    bbox_test = anno_bbox['bbox_test']
    list_action = anno_bbox['list_action']

    read_features(paths.data_root, paths.tmp_root, bbox_train, list_action)
    read_features(paths.data_root, paths.tmp_root, bbox_test, list_action)


def main():
    paths = hico_config.Paths()
    start_time = time.time()
    collect_data(paths)
    print(f"Time elapsed: {time.time() - start_time:.2f}s")


if __name__ == '__main__':
    main()
