import os
import shutil
import pytest

@pytest.fixture(scope='session')
def remove_dirs():
    shutil.rmtree('../tmp')


