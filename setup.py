#!/usr/bin/env python
#
# This file is part of SharedArray.
# Copyright (C) 2014 Mathieu Mirmont <mat@parad0x.org>
#
# SharedArray is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# SharedArray is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with SharedArray.  If not, see <http://www.gnu.org/licenses/>.

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
