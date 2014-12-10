#!/usr/bin/env python
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
import os

setup(name         = 'SharedArray',
      description  = 'Share numpy arrays between processes',
      author       = 'Mathieu Mirmont',
      author_email = 'mat@parad0x.org',
      url          = 'http://parad0x.org/git/python/shared-array/',
      version      = '0.0',
      license      = "GPL 2",
      platforms    = "POSIX",
      ext_modules = [ Extension('SharedArray',
                                glob(os.path.join('src', '*.c')),
                                libraries = [ 'rt' ]) ])
