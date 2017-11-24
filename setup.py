#!/usr/bin/env python
#
# This file is part of SharedArray.
# Copyright (C) 2014-2016 Mathieu Mirmont <mat@parad0x.org>
#
# SharedArray is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 2 of the License, or
# (at your option) any later version.
#
# SharedArray is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with SharedArray.  If not, see <http://www.gnu.org/licenses/>.

from setuptools import setup
from setuptools.extension import Extension
from setuptools.command.build_ext import build_ext as _build_ext

from glob import glob
from os import path

here = path.abspath(path.dirname(__file__))

def read(fname):
    return open(path.join(here, fname)).read()

extensions = [Extension('SharedArray', 
                        glob(path.join(here, 'src', '*.c')),
                        libraries = [ 'rt' ])]

class build_ext(_build_ext):
    """ build_ext command for use when numpy headers are needed. """
    def run(self):

        # Import numpy here, only when headers are needed.
        import numpy

        # Add numpy headers to include_dirs.
        self.include_dirs.append(numpy.get_include())

        # Call original build_ext command
        _build_ext.run(self)

setup(name    = 'SharedArray',
      version = '1.0',

      # Description
      description      = 'Share numpy arrays between processes',
      long_description = read('README.md'),

      # Contact
      author       = 'Mathieu Mirmont',
      author_email = 'mat@parad0x.org',
      url          = 'http://parad0x.org/git/python/shared-array/about',

      # License
      license   = 'https://www.gnu.org/licenses/gpl-2.0.html',

      # Extras for pip
      keywords  = 'numpy array shared memory shm',
      classifiers  = [
          'Development Status :: 5 - Production/Stable',
          'Intended Audience :: Developers',
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: GNU General Public License v2 (GPLv2)',
          'Operating System :: POSIX',
          'Operating System :: Unix',
          'Programming Language :: C',
          'Topic :: Scientific/Engineering'
      ],
      
      install_requires=['numpy'],
      setup_requires=['numpy'],
      cmdclass={'build_ext':build_ext},

      # Compilation
      ext_modules  = extensions)
