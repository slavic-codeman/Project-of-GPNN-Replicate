"""
Created on Oct 02, 2017

@author: Siyuan Qi

Description of the file.

"""

import os
import time
import pickle
import argparse

import numpy as np
import torch.utils.data

from . import hico_config


class HICO(torch.utils.data.Dataset):
    """
    PyTorch Dataset for the HICO dataset.

    Args:
        root (str): Path to the root directory containing the data.
        sequence_ids (list): List of sequence IDs to load.

    Attributes:
        root (str): Path to the data directory.
        sequence_ids (list): List of sequence IDs.
    """
    def __init__(self, root, sequence_ids):
        self.root = root
        self.sequence_ids = sequence_ids

    def __getitem__(self, index):
        """
        Retrieves a single data instance by index.

        Args:
            index (int): Index of the data instance.

        Returns:
            tuple: A tuple containing edge features, node features, adjacency matrix,
                   node labels, sequence ID, detected classes, bounding boxes,
                   number of humans, and number of objects.
        """
        sequence_id = self.sequence_ids[index]
        data_path = os.path.join(self.root, f"{sequence_id}.p")
        edge_features_path = os.path.join(self.root, f"{sequence_id}_edge_features.npy")
        node_features_path = os.path.join(self.root, f"{sequence_id}_node_features.npy")

        with open(data_path, "rb") as f:
            data = pickle.load(f)

        det_classes = data['classes']
        det_boxes = data['boxes']
        human_num = data['human_num']
        obj_num = data['obj_num']
        adj_mat = data['adj_mat']
        node_labels = data['node_labels']

        edge_features = np.load(edge_features_path)
        node_features = np.load(node_features_path)

        return edge_features, node_features, adj_mat, node_labels, sequence_id, det_classes, det_boxes, human_num, obj_num

    def __len__(self):
        """
        Returns the length of the dataset.

        Returns:
            int: Number of instances in the dataset.
        """
        return len(self.sequence_ids)


def main(args):
    """
    Main function for testing the HICO dataset class.
    """
    start_time = time.time()

    # Load subset filenames (train, val, test)
    subset = ['train', 'val', 'test']
    hico_voc_path = os.path.join(args.data_root, 'Deformable-ConvNets/data/hico/VOC2007')
    with open(os.path.join(hico_voc_path, 'ImageSets/Main', f"{subset[0]}.txt")) as f:
        filenames = [line.strip() for line in f.readlines()]

    # Test the dataset class with a small subset
    training_set = HICO(args.tmp_root, filenames[:5])
    print(f'{len(training_set)} instances in the dataset.')

    # Test data retrieval
    edge_features, node_features, adj_mat, node_labels, sequence_id, det_classes, det_boxes, human_num, obj_num = training_set[0]
    print(f"Sequence ID: {sequence_id}")
    print(f"Edge features shape: {edge_features.shape}")
    print(f"Node features shape: {node_features.shape}")
    print(f"Adjacency matrix shape: {adj_mat.shape}")
    print(f"Node labels shape: {node_labels.shape}")
    print(f"Number of humans: {human_num}, Number of objects: {obj_num}")

    print(f"Time elapsed: {time.time() - start_time:.2f}s")


def parse_arguments():
    """
    Parses command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    paths = hico_config.Paths()
    parser = argparse.ArgumentParser(description='HICO dataset')
    parser.add_argument('--data-root', default=paths.data_root, help='Dataset path')
    parser.add_argument('--tmp-root', default=paths.tmp_root, help='Intermediate result path')
    return parser.parse_args()


def unit_test():
    args = parse_arguments()
    main(args)
if __name__ == '__main__':
    args = parse_arguments()
    main(args)
