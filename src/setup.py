import setuptools
from setuptools import setup, find_packages

with open('README.md', 'r') as f:
    LONG_DESCRIPTION = f.read()

NAME = 'perlib'
VERSION = '1.0.0'
LONG_DESCRIPTION_CONTENT_TYPE = 'text/markdown'
URL = 'https://github.com/Ruzzg/perlib'
AUTHOR = 'Rüzgar Ersin Kanar'
AUTHOR_EMAIL = 'ruzgarknr@gmail.com'
LICENSE = 'Apache Software License'
KEYWORDS = 'perlib,tensorflow,machine learning,deep learning'

setup(
    name=NAME,
    version=VERSION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type=LONG_DESCRIPTION_CONTENT_TYPE,
    packages = find_packages(),
    #url=URL,
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    license=LICENSE,
    keywords=KEYWORDS,
    classifiers=[
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
)