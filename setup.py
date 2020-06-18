#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  2 21:22:31 2019

@author: ben
"""

import setuptools

setuptools.setup(name='vacuumworld-gym',
      version='0.0.1',
      description='',
      url='https://github.com/BenedictWilkinsAI/vacuumworld-gym',
      author='Benedict Wilkins',
      author_email='brjw@hotmail.co.uk',
      license='GNU General Public License v3 (GPLv3)T',
      packages=['vwgym'],
      include_package_data=True,
      install_requires=['vacuumworld', 'gym'],
      classifiers=[
        "Programming Language :: Python :: 3.7",
        "Development Status :: 2 - Pre-Alpha",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
      ])
