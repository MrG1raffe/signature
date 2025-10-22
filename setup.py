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
      'jax>=0.4.30',
      'typing',
      'matplotlib',
      'numba>=0.58.1',
      'iisignature',
      'simulation @ git+https://github.com/MrG1raffe/simulation.git',
      'pricing @ git+https://github.com/MrG1raffe/pricing.git'
   ], #external packages as dependencies
)