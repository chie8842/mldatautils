# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import shutil
import boto3
import glob
import re
import sys
from boto3 import Session
from boto3.s3.transfer import TransferConfig
from pytz import timezone
from datetime import date
from datetime import datetime

from mldatautils.utils import isnotebook
from mldatautils.utils import make_dirs
from mldatautils.utils import get_jstdate_string
from mldatautils.utils import create_version
from mldatautils.utils import is_valid_version
from mldatautils.logger import logger_config

if isnotebook():
    from tqdm import tqdm_notebook as tqdm
else:
    from tqdm import tqdm


class MLDataBucketUtils(object):
    """ResearchBucketUtils
    This function is to access to s3 bucket for ml data management.
    Args:
        bucket_name(str): s3 bucket name
        loglevel(str or int): logging setting
    Examples:
        >>> mlbucket = MLDataBucketUtils('tmp_bucket', loglevel='INFO')
        >>> mlbucket.download_model(
                env_type='production',
                project_name='tmp_project')
    """

    def __init__(self, bucket_name, loglevel='WARNING'):
        self.s3 = boto3.resource('s3')
        self.logger = logger_config('s3_utils', loglevel)
        self.bucket_name = bucket_name
        self.bucket = self.s3.Bucket(self.bucket_name)

    def s3_newest_version(self, env_type, project_name, additional_prefix=''):
        """ Returns newest model version in s3 which has following structure.
        s3://bucket_name/env_type/project_name(/additional_prefix).

        Args:
            env_type(str): production/staging/development
            project_name(str): project name
            additional_prefix: If model prefix is 'production/tmp_project/models/yyyymmdd-HHMM',
                               `additional_prefix` is 'models'
        Return:
            newest_version: newest version in valid version(formatted by 'yyyymmdd-HHMM')
        """
        s3_prefix = os.path.join(env_type, project_name, additional_prefix)
        if s3_prefix[-1] != '/':
            s3_prefix = s3_prefix + '/'
        s3 = boto3.client('s3')
        objects = s3.list_objects_v2(Bucket=self.bucket_name, Prefix=s3_prefix, Delimiter='/')
        versions = [o.get('Prefix')[len(s3_prefix):-1] for o in objects.get('CommonPrefixes')]

        sorted_versions = sorted([v for v in versions if is_valid_version(v)])
        newest_version = sorted_versions[-1]
        return newest_version

    def download_model(self, env_type, project_name, additional_prefix='', model_version=None, model_dir='models'):
        """download model from s3 which has following structure
        s3://bucket_name/env_type/project_name(/additional_prefix)
        This method can recursively download all objects under the above prefix.

        Args:
            env_type(str): production/staging/development
            project_name(str): project name
            additional_prefix: If model prefix is 'production/tmp_project/models/yyyymmdd-HHMM',
                               `additional_prefix` is 'models'
            model_version(str): If you'd like to download specific version of model, you can use this argument. 
                                If this argument isn't set, latest version of model will be downloaded.
            model_dir(str): model directory.Default: 'models'.

        Return:
            model_version(str): version of downloaded data in s3
        """
        if model_version is None:
            model_version = self.s3_newest_version(env_type, project_name, additional_prefix)
        s3_prefix = os.path.join(env_type, project_name, model_version)
        self.download_data(s3_prefix, model_dir)
        return model_version

    def download_data(self, s3_prefix, data_dir):
        """ download data from s3 which has following structure.
            s3://bucket_name/s3_prefix
        This method can recursively download all objects under the above prefix.

        Args:
            s3_prefix(str): s3 prefix of data(directory)
            data_dir: local directory to put downloaded files
        """
        self.logger.info('download data from s3://{}/{} to '.format(self.bucket_name, s3_prefix, data_dir))
        for o in self.bucket.objects.filter(Prefix=s3_prefix).all():
            data_path = os.path.join(
                data_dir,
                re.sub(s3_prefix + '/', '', o.key))
            make_dirs(os.path.dirname(data_path))
            self.bucket.download_file(o.key, data_path)
            self.logger.info('downloaded {} to {}'.format(o.key, data_path))

    def upload_model(self, env_type, project_name, model_dir='models'):
       """upload model to s3
        This function upload all the data in the model directory to s3 according to following structure.
        s3://bucket_name/env_type/project_name/yyyymmdd-HHMM

       Args:
            env_type(str): production/staging/development
            project_name(str): part of s3 prefix
            model_dir(str): model directory.Default: 'models'.
       """
       s3_root_prefix = os.path.join(env_type, project_name, create_version())
       self.upload_data(s3_root_prefix, model_dir)

       for i in glob.glob(
           os.path.join(model_dir, '**'), recursive=True):
           pathname = re.sub(model_dir+'/', '', i)
           s3_prefix = os.path.join(s3_root_prefix, pathname)
           if os.path.isfile(i):
               self.bucket.upload_file(i, s3_prefix)
               self.logger.info(
                  f'uploaded: s3://{self.bucket_name}/{s3_prefix}')
       return s3_root_prefix

    def upload_data(self, s3_root_prefix, data_dir, filter_string='**'):
        """upload data to s3
        This function upload all the data in the data directory to s3 according to following structure.
        s3://bucket_name/env_type/project_name

        Args:
            project_name(str): project name
            data_dir(str): directory of upload data
            filter_string(str): filter string
        """
        files = glob.glob(os.path.join(data_dir, filter_string), recursive=True)
        for i in files:
           pathname = re.sub(data_dir+'/', '', i)
           s3_prefix = os.path.join(s3_root_prefix, pathname)
           if os.path.isfile(i):
               self.bucket.upload_file(i, s3_prefix)
               self.logger.info(
                  f'uploaded: s3://{self.bucket_name}/{s3_prefix}')
       return s3_root_prefix

