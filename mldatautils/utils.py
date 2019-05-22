import os
import re
import numpy as np
import pandas as pd
from queuery_client import Client
from queuery_client import QueueryDataSource
from pytz import timezone
from datetime import date
from datetime import datetime
from datetime import timedelta


def get_jstdate_string():
    """
    e.g.
    >>> query = '''
    >>> select id, name from users where entered_at between \'{}\' and \'{}\'
    >>> '''.format(get_past_jstdate_string(), get_jstdate_string())
    >>> print(query)
    select id, name from users where entered_at between '2019-04-24' and '2019-04-25'
    """
    return datetime.strftime(
        datetime.now(timezone('Asia/Tokyo')), '%Y-%m-%d')

def get_past_jstdate_string(n=1):
    """
    e.g.
    >>> query = '''
    >>> select id, name from users where entered_at between \'{}\' and \'{}\'
    >>> '''.format(get_past_jstdate_string(), get_jstdate_string())
    >>> print(query)
    select id, name from users where entered_at between '2019-04-24' and '2019-04-25'
    """
    return datetime.strftime(
        datetime.now(timezone('Asia/Tokyo')) + timedelta(days=-n),
        '%Y-%m-%d')


def create_version():
    return datetime.strftime(
        datetime.now(timezone('Asia/Tokyo')), '%Y%m%d-%H%M')


def make_dirs(dirname):
    if not os.path.isdir(dirname):
        os.makedirs(dirname)


def isnotebook():
   """
   e.g.
   >>> if isnotebook():
   >>>     from tqdm import tqdm_notebook as tqdm
   >>> else:
   >>>     from tqdm import tqdm
   """
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False


def local_newest_version(model_dir='models'):
    """ Returns newest model version in local machine, e.g '20180525-1202'
    e.g.
    >>> model.load(os.path.join('models', local_newest_version(), 'model.pkl'))
    """
    if not os.path.isdir(model_dir):
        return None
    else:
        sorted_versions = sorted([v for v in os.listdir(model_dir)
                if os.path.isdir(os.path.join(model_dir, v)) and is_valid_version(v)])
        if len(sorted_versions) == 0:
            return None
        else:
            newest_version = sorted_versions[-1]
            return newest_version


def is_valid_version(v):
    """ Check if a string is a valid version
    >>> is_valid_version('20180525-1202')
    True
    >>> is_valid_version('20180525')
    False
    >>> is_valid_version('201805251221')
    False
    """
    if re.match('\A\d{8}-\d{4}\Z', v):
        return True
    return False

