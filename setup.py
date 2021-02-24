
from setuptools import setup

setup(name='tqpt',
      version='0.1.0',
      description='Automatically draw topological phase boundaries',
      author='Alexander Kerr',
      author_email='ajkerr0@gmail.com',
      packages=['tqpt'],
      install_requires=[
      'numpy',
      'scipy',
      'sklearn',
      ],
      )