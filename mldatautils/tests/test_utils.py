import os
from mldatautils.utils import make_dirs, isnotebook

def test_make_dirs():
    dir_name = 'tmp/tmp'
    make_dirs(dir_name)
    assert os.path.isdir(dir_name)

def test_isnotebook():
  assert not isnotebook()
