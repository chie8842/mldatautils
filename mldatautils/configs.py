from configparser import ConfigParser
import os
from pathlib import Path
import logging
from sqlalchemy import create_engine
import datetime

class GetConfigs(object):
    def __init__(self, config_file='config.ini'):
        self.config_file = config_file

    def get_config(self, config_name):
        configs = ConfigParser()
        configs.read(self.config_file)
        config = dict(configs.items(config_name))
        return config

    def get_awsconfig(self):
        return self.get_config('aws_default')

    def get_dwhconfig(self):
        return self.get_config('dwh_schema')

    def get_logconfig(self):
        return self.get_config('logging')

    def create_engine(self):
        dwh_config = self.get_dwhconfig()
        url = 'postgres://{user}:{password}@{host}:{port}/{database}'.format(**dwh_config)
        engine = create_engine(url)
        return engine


