import os
import sys
from configparser import ConfigParser
from sqlalchemy import create_engine

def _config_parse(config_file):
    if config_file is not None:
        try:
            configs = ConfigParse()
            configs.read(config_file)
            dwh_schema = dict(configs.items('dwh_schema'))
        except FileNotFoundError:
            logger.error(f'{config_file} does not exist.')
            sys.exit(1)
    else:
        dwh_schema = {
            'username': os.getenv('DB_USERNAME')
            'password': os.getenv('DB_PASSWORD')
            'hostname': os.getenv('DB_HOSTNAME')
            'port': os.getenv('DB_PORT')
            'database': os.getenv('DATABASE')
        }
    return config_file

def create_engine(config_file=None):
    dwh_schema = _config_parse(config_file)
    url = 'postgres://{user}:{password}@{host}:{port}/{database}'.format(**dwh_config)
    return create_engine(url)

