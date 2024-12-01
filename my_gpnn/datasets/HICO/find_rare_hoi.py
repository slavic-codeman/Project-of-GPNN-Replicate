"""
Created on Oct 22, 2017

@author: Siyuan Qi

Description of the file.

"""

import os
import numpy as np
import scipy.io
from . import hico_config


def collect_hoi_stats(bbox):
    """
    Collect statistics of HOI (Human-Object Interaction) occurrences in the dataset.

    Args:
        bbox: MATLAB structure containing bounding box and HOI information.

    Returns:
        stats: A numpy array where each index represents an HOI ID, and the value represents its occurrence count.
    """
    stats = np.zeros(600)  # Assuming 600 HOI classes
    for idx in range(bbox['filename'].shape[1]):
        for i_hoi in range(bbox['hoi'][0, idx]['id'].shape[1]):
            hoi_id = bbox['hoi'][0, idx]['id'][0, i_hoi][0, 0]
            stats[int(hoi_id) - 1] += 1  # Adjust index for 0-based indexing in Python

    return stats


def split_testing_set(paths, bbox, stats):
    """
    Split the testing set into rare and non-rare sets based on HOI occurrence statistics.

    Args:
        paths: Configuration paths containing data and temp root paths.
        bbox: MATLAB structure containing bounding box and HOI information for the test set.
        stats: Array of HOI occurrence statistics.

    Outputs:
        Writes two files: test_rare.txt and test_non_rare.txt, containing filenames of rare and non-rare samples.
    """
    feature_path = os.path.join(paths.data_root, 'processed', 'features_background_49')

    rare_set = []
    non_rare_set = []

    for idx in range(bbox['filename'].shape[1]):
        filename = str(bbox['filename'][0, idx][0])
        filename = os.path.splitext(filename)[0] + '\n'

        try:
            # Adjusted for Python 3 compatibility
            det_classes = np.load(os.path.join(feature_path, f"{filename.strip()}_classes.npy"))
        except IOError:
            continue

        rare = False
        for i_hoi in range(bbox['hoi'][0, idx]['id'].shape[1]):
            hoi_id = bbox['hoi'][0, idx]['id'][0, i_hoi][0, 0]
            if stats[int(hoi_id) - 1] < 10:  # HOI is rare if its count is less than 10
                rare_set.append(filename)
                rare = True
                break  # Skip remaining HOIs for this image if it's rare
        if not rare:
            non_rare_set.append(filename)

    os.makedirs(os.path.join(paths.tmp_root, 'hico'), exist_ok=True)  # Ensure output directory exists
    with open(os.path.join(paths.tmp_root, 'hico', 'test_rare.txt'), 'w') as f:
        f.writelines(rare_set)

    with open(os.path.join(paths.tmp_root, 'hico', 'test_non_rare.txt'), 'w') as f:
        f.writelines(non_rare_set)


def find_rare_hoi(paths):
    """
    Identify rare HOIs in the HICO dataset and split the testing set.

    Args:
        paths: Configuration paths containing data and temp root paths.
    """
    anno_bbox = scipy.io.loadmat(os.path.join(paths.data_root, 'anno_bbox.mat'))
    bbox_train = anno_bbox['bbox_train']
    bbox_test = anno_bbox['bbox_test']

    stats = collect_hoi_stats(bbox_train)
    split_testing_set(paths, bbox_test, stats)


def main():
    """
    Main function to find and process rare HOIs.
    """
    paths = hico_config.Paths()
    find_rare_hoi(paths)


if __name__ == '__main__':
    main()
