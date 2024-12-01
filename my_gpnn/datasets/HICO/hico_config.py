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
        Configuration of data paths.

        Attributes:
            data_root (str): The root folder of all the recorded data of events.
        """
        super(Paths, self).__init__()
        self.data_root = self.hico_data_root


def set_logger(name='learner.log'):
    """
    Sets up a logger with the specified file name.

    Args:
        name (str): The name of the log file.

    Returns:
        logging.Logger: Configured logger instance.
    """
    log_dir = os.path.dirname(name)
    if log_dir and not os.path.exists(log_dir):
        try:
            os.makedirs(log_dir, exist_ok=True)
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # Prevent duplicate handlers if logger is reused
    if not logger.handlers:
        file_handler = logging.FileHandler(name, mode='w')
        file_handler.setFormatter(
            logging.Formatter('%(asctime)s %(levelname)s: %(message)s', "%Y-%m-%d %H:%M:%S")
        )
        logger.addHandler(file_handler)

    return logger


def main():
    """
    Unit test for hico_config.py.
    """
    print("Testing Paths class...")
    paths = Paths()
    print(f"Paths.hico_data_root: {paths.hico_data_root}")
    print(f"Paths.data_root: {paths.data_root}")
    assert hasattr(paths, "hico_data_root") and paths.data_root == paths.hico_data_root
    print("Paths class test passed!")

    print("Testing set_logger function...")
    log_file = "test_logs/learner.log"
    logger = set_logger(log_file)
    logger.info("This is a test log entry.")
    assert os.path.exists(log_file)

    # Check log content
    with open(log_file, "r") as f:
        content = f.read()
        assert "This is a test log entry." in content

    print("set_logger function test passed!")


if __name__ == '__main__':
    main()
