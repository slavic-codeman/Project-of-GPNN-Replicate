"""
Created on Feb 17, 2017

@author: Siyuan Qi

Description of the file.

"""

import errno
import logging
import os

import config


class Paths(config.Paths):
    def __init__(self):
        """
        Configuration of data paths
        member variables:
            data_root: The root folder of all the recorded data of events
            metadata_root: The root folder where the processed information (Skeleton and object features) is stored.
        """
        super(Paths, self).__init__()
        self.data_root = self.cad_data_root


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

def main():
    """
    Unit test for cad120_config.py
    """
    print("Testing Paths class...")
    paths = Paths()

    print(f"Paths.data_root: {paths.data_root}")
    print("Paths class test passed!")

    print("Testing set_logger function...")
    logger = set_logger("./test_logs/learner.log")
    logger.info("This is a test log entry.")
    assert os.path.exists("test_logs/learner.log")
    with open("test_logs/learner.log", "r") as log_file:
        log_content = log_file.read()
        assert "This is a test log entry." in log_content
    print("set_logger function test passed!")


if __name__ == '__main__':

    main()
