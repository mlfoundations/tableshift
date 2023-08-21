#!/usr/bin/env python

from distutils.core import setup

setup(name='tableshift',
      version='0.1',
      url='https://tableshift.org',
      description='A tabular data benchmarking toolkit.',
      long_description='A benchmarking toolkit for tabular data under distirbution shift. '
                       'For more details, see the paper '
                       '"Benchmarking Distribution Shift in Tabular Data with TableShift", '
                       'Gardner, Popovic, and Schmidt, 2023.',
      author='Josh Gardner',
      author_email='jpgard@cs.washington.edu',
      packages=['tableshift'],
      data_files=[('tableshift/datasets',
                   ['tableshift/datasets/nhanes_data_sources.json',
                    'tableshift/datasets/icd9-codes.json'])],
      install_requires=[
          'numpy==1.23.5',
          'ray==2.2',
          'torch',
          'torchvision',
          'scikit-learn',
          'pandas',
          'fairlearn',
          'folktables',
          'frozendict',
          'rtdl',
          'xport',
          'tqdm',
          'hyperopt',
          'h5py',
          'tables',
          'category_encoders',
          'einops',
          'tab-transformer-pytorch',
          'openpyxl',
          'optuna',
          'kaggle',
          'datasets',
          'torchinfo'
      ]
      )
