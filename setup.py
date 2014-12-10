#!/usr/bin/env python3
##
##      Project: SharedArray
## 
##         File: setup.py
## 
##  Description: Setup script
## 
##     Author/s: Mathieu Mirmont <mat@parad0x.org>
## 
##   Created on: 08/12/2014
## 
##

from distutils.core import setup, Extension
from glob import glob

setup(ext_modules = [ Extension("SharedArray", glob("src/*.c")) ])
