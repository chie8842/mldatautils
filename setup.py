#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='mldatautils',
    version='0.0.2',
    description="mldatautils",
    long_description="mldatautils",
    download_url='https://pypi.org/project/mldatautils',
    author="Chie Hayashida",
    author_email='chie8842@gmail.com',
    url='https://github.com/chie8842/mldatautils',
    packages=find_packages(exclude=["*conftest*", "*tests*"]),
    include_package_data=True,
    install_requires=requirements,
    dependency_links=['https://pypi.python.org/pypi'],
    classifiers=[
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],
    test_suite='test'
)
