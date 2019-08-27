import os
import sys
from configparser import ConfigParser
from sqlalchemy import create_engine

def _config_parse(config_file):
    if config_file is not None:
        try:
            configs = ConfigParser()
            configs.read(config_file)
            dwh_schema = dict(configs.items('dwh_schema'))
        except FileNotFoundError:
            logger.error(f'{config_file} does not exist.')
            sys.exit(1)
    else:
        dwh_config = {
            'username': os.getenv('DB_USERNAME'),
            'password': os.getenv('DB_PASSWORD'),
            'hostname': os.getenv('DB_HOSTNAME'),
            'port': os.getenv('DB_PORT'),
            'database': os.getenv('DATABASE'),
        }
    return dwh_config

def engine(config_file=None):
    dwh_config = _config_parse(config_file)
    url = 'postgres://{username}:{password}@{hostname}:{port}/{database}'.format(**dwh_config)
    return create_engine(url)

