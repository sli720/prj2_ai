"""
Script for logging.
To disable logging the log level can be changed in the setLevel() call to logging.ERROR
"""
import logging
import sys


def get_logger(name):
    # Setup logging
    formatter = logging.Formatter(fmt='%(asctime)s %(levelname)-8s %(message)s', datefmt='%H:%M:%S')
    screen_handler = logging.StreamHandler(stream=sys.stdout)
    screen_handler.setFormatter(formatter)
    logger = logging.getLogger(name)
    logger.setLevel(logging.ERROR)  # set to logging.ERROR to disable logging messages; set logging.DEBUG to enable them
    logger.addHandler(screen_handler)
    return logger

