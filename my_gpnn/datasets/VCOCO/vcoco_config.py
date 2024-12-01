"""
Created on Feb 24, 2018

@author: Siyuan Qi

Description of the file.

Compatible with Python 3.9
"""

import errno
import logging
import os

import config


class Paths(config.Paths):
    """
    Inherit configuration from the base `config.Paths` and extend for V-COCO.
    """
    def __init__(self):
        """
        Configuration of data paths
        member variables:
            - data_root: The root folder of all the recorded data of events.
            - metadata_root: The root folder where the processed information 
                             (Skeleton and object features) is stored.
        """
        super(Paths, self).__init__()
        # Set the data root for V-COCO specifically
        self.data_root = self.vcoco_data_root


def set_logger(name='learner.log'):
    """
    Set up the logger to write logs to a specified file.
    
    Args:
        name (str): The name of the log file.

    Returns:
        logging.Logger: Configured logger instance.
    """
    if not os.path.exists(os.path.dirname(name)):
        try:
            # Safely create directories
            print("SET UP\n\n",name)
            print(os.path.dirname(name))
            os.makedirs(os.path.dirname(name))
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise  # Re-raise if it's not a "file exists" error

    # Set up the logger
    logger = logging.getLogger(name)
    file_handler = logging.FileHandler(name, mode='w')  # Write mode
    file_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s: %(message)s',
                                                "%Y-%m-%d %H:%M:%S"))
    logger.addHandler(file_handler)
    logger.setLevel(logging.DEBUG)
    return logger


def main():
    """
    Simple unit test to verify functionality.
    """
    paths = Paths()
    print("V-COCO data root:", paths.data_root)

    logger = set_logger('./test_vcoco.log')
    logger.info("Logger test passed.")
    print("Logger and paths tested successfully!")


if __name__ == '__main__':
    main()
