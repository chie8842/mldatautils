import logging
import datetime
import os
from mldatautils.utils import make_dirs

def logger_config(log_name, loglevel='INFO', log_file='log.txt'):
    '''logger_config
    Args:
        log_name(str):
        loglevel(str or int): loglevel('ERROR'/'WARN'/'INFO'/'DEBUG')
        log_file: logfile name

    Examples:
        >>> logger = logger_config('test', loglevel='WARN', log_file='log/log.txt')
        >>> logger.warning('This is the test logging')
        2019-04-26 05:56:13,128 test: WARNING This is the test logging
    '''

    log_dir = os.path.dirname(log_file)
    if log_dir != '' and not os.path.exists(log_dir):
        make_dirs(log_dir)
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(loglevel)
    formatter = logging.Formatter('%(asctime)s %(name)s: %(levelname)s %(message)s')
    stream_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(filename=log_file)
    file_handler.setLevel(loglevel)
    file_handler.setFormatter(logging.Formatter('%(asctime)s %(name)s %(levelname)s %(message)s'))

    logger = logging.getLogger(log_name)
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    logger.setLevel(loglevel)

    return logger
