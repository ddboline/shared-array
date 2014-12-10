SharedArray python/numpy extension
==================================

This is a simple python extension that lets you share numpy arrays
with other processes on the same computer.  It uses posix shared
memory internally and therefore should work on most operating systems.

Example
-------

	import numpy as np
	import SharedArray as sa
	
	a = sa.create("test1", 10)
	b = sa.attach("test1")
	
	a[0] = 42
	print(b[0])
	
	del a
	print(b[0])
	
	sa.delete("test1")
	print(b[0])

Functions
---------

### `SharedArray.create(name, shape, dtype=float)`

This function creates an array in shared memory identified by
`name`.  The `shape` and `dtype` arguments are the same as the numpy
function `numpy.zeros()`.  The returned array is initialized to zero.
The shared memory block holding the content of the array will not be
deleted when this array is destroyed, either implicitly or explicitly
by calling `del`, it will simply be detached from the current process.
To delete a shared array use the `SharedArray.delete()` function.

### `SharedArray.attach(name)`

This function attaches an array previously created in shared memory
and identified by `name`.  The shared memory block holding the content
of the array will not be deleted when this array is destroyed, either
implicitly or explicitly by calling `del`, it will simply be detached
from the current process.  To delete a shared array use the
`SharedArray.delete()` function.

### `SharedArray.delete(name)`

This function destroys an array previously created in shared memory
and identified by `name`.  After calling `delete`, the array will not
be attachable anymore, but currents attachments will not be affected.

Requirements
------------

* Python 2.7 or 3+
* Numpy 1.8
* Posix shared memory interface

SharedArray uses the posix shm interface (`shm_open` and `shm_unlink`)
and so should work on most operating systems that follow the posix
standards (Linux, *BSD, etc.).

Installation
------------

The extension uses the `distutils` python package that should be
familiar to most python users. To test the extension directly from the
source tree, without installing, type:

	python setup.py build_ext --inplace

To build and install the extension system-wide, type:

	python setup.py build
	sudo python setup.py install
