
from setuptools import setup

setup(name='pydlbaseball',
      version='0.1',
      description='Deep Learning Tools for training baseball models.',
      url='https://github.com/nchapman95/pydlbaseball/',
      author='Nick Chapman',
      author_email='nchapman958@gmail.com',
      license='MIT',
      install_requires=[
          'matplotlib','pandas'
      ],
      packages=['pydlbaseball'],
      zip_safe=False)

