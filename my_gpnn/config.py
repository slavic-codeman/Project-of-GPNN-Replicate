"""
Created on Feb 17, 2017

@author: Siyuan Qi

Description of the file.

"""

import errno
import logging
import os


class Paths(object):
    def __init__(self):
        """
        Configuration of data paths
        member variables:
            data_root: The root folder of all the recorded data of events
            metadata_root: The root folder where the processed information (Skeleton and object features) is stored.
        """
        # self.project_root = '/home/tangjq/WORK/GPNN/gpnn-master/'
        # self.tmp_root = '/data1/tangjq/tmp'
        # self.log_root = os.path.join(self.project_root, 'log')

        # self.cad_data_root = ''
        # self.hico_data_root = os.path.join(self.project_root, 'tmp', 'hico')
        # self.vcoco_data_root = '/home/tangjq/WORK/GPNN/gpnn-master/'
        self.project_root = os.path.dirname(os.path.dirname(__file__))
        self.tmp_root = os.path.join(self.project_root, 'tmp')
        self.log_root = os.path.join(self.project_root, 'log')

        self.cad_data_root = ''
        self.hico_data_root = os.path.join(self.project_root, 'tmp', 'hico')
        self.vcoco_data_root = os.path.dirname(os.path.dirname(__file__))


def set_logger(name='learner.log'):
    if not os.path.exists(os.path.dirname(name)):
        try:
            os.makedirs(os.path.dirname(name))
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise

    logger = logging.getLogger(name)
    file_handler = logging.FileHandler(name, mode='w')
    file_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s: %(message)s',
                                                "%Y-%m-%d %H:%M:%S"))
    logger.addHandler(file_handler)
    logger.setLevel(logging.DEBUG)
    return logger
