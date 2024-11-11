from setuptools import setup, find_packages

setup(
   name='signature',
   version='1.0',
   description='Path signatures and tensor sequences',
   author='MrG1raffe',
   author_email='dimitri.sotnikov@gmail.com',
   packages=find_packages(),
   install_requires=[
      'numpy>=1.23.0',
      'typing',
      'matplotlib',
      'numba>=0.58.1',
      'iisignature'
   ], #external packages as dependencies
)