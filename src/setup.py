import setuptools
from setuptools import setup, find_packages
import os

with open('README.md', 'r') as f:
    LONG_DESCRIPTION = f.read()

NAME = 'perlib'
VERSION = '1.0.6'
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
    install_requires =[ "openpyxl",
                        "joblib",
                        "keras",
                        "keras_tcn",
                        "lightgbm",
                        "loguru",
                        "matplotlib",
                        "numpy",
                        "pandas",
                        "python_dateutil",
                        "scikit_learn",
                        "scipy",
                        "statsmodels",
                        "tensorflow",
                        "tqdm",
                        "xgboost"],
    keywords=KEYWORDS,
    classifiers=[
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
)