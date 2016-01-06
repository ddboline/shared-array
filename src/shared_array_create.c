/* 
 * This file is part of SharedArray.
 * Copyright (C) 2014 Mathieu Mirmont <mat@parad0x.org>
 * 
 * SharedArray is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 2 of the License, or
 * (at your option) any later version.
 * 
 * SharedArray is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with SharedArray.  If not, see <http://www.gnu.org/licenses/>.
 */

#define NPY_NO_DEPRECATED_API	NPY_1_8_API_VERSION
#define PY_ARRAY_UNIQUE_SYMBOL	SHARED_ARRAY_ARRAY_API
#define NO_IMPORT_ARRAY

#include <Python.h>
#include <numpy/arrayobject.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include "shared_array.h"

/*
 * Create a numpy array in shared memory
 */
static PyObject *do_create(const char *name, int ndims, npy_intp *dims, PyArray_Descr *dtype)
{
	struct array_meta *meta;
	size_t size;
	size_t map_size;
	void *map_addr;
	int i;
	int fd;
	PyObject *ret;
	PyLeonObject *leon;

	/* Check the number of dimensions */
	if (ndims > SHARED_ARRAY_NDIMS_MAX) {
		PyErr_SetString(PyExc_ValueError, "Too many dimensions, recompile SharedArray!");
		return NULL;
	}

	/* Calculate the memory size of the array */
	size = dtype->elsize;
	for (i = 0; i < ndims; i++)
		size *= dims[i];

	/* Calculate the size of the mmap'd area */
	map_size = size + sizeof (*meta);

	/* Create the shm block */
	if ((fd = shm_open(name, O_RDWR | O_CREAT | O_EXCL, 0666)) < 0)
		return PyErr_SetFromErrnoWithFilename(PyExc_OSError, name);

	/* Grow the file */
	if (ftruncate(fd, map_size) < 0) {
		close(fd);
		return PyErr_SetFromErrnoWithFilename(PyExc_OSError, name);
	}

	/* Map it */
	map_addr = mmap(NULL, map_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
	close(fd);
	if (map_addr == MAP_FAILED)
		return PyErr_SetFromErrnoWithFilename(PyExc_OSError, name);

	/* Append meta-data to the array in memory */
	meta = (struct array_meta *) (map_addr + size);
	strncpy(meta->magic, SHARED_ARRAY_MAGIC, sizeof (meta->magic));
	meta->size = size;
	meta->typenum = dtype->type_num;
	meta->ndims = ndims;
	for (i = 0; i < ndims; i++)
		meta->dims[i] = dims[i];

	/* Summon Leon to cleanup later */
	leon = PyObject_MALLOC(sizeof (*leon));
	PyObject_INIT((PyObject *) leon, &PyLeonObject_Type);
	leon->data = map_addr;
	leon->size = map_size;

	/* Create the array object */
	ret = PyArray_SimpleNewFromData(ndims, dims, dtype->type_num, map_addr);

	/* Attach Leon to the array */
	PyArray_SetBaseObject((PyArrayObject *) ret, (PyObject *) leon);
	return ret;
}

/*
 * Method: SharedArray.create()
 */
PyObject *shared_array_create(PyObject *self, PyObject *args, PyObject *kwds)
{
	static char *kwlist[] = { "name", "shape", "dtype", NULL };
	const char *name;
	PyArray_Dims shape = { NULL, 0 };
	PyArray_Descr *dtype = NULL;
	PyObject *ret = NULL;

	/* Parse the arguments */
	if (!PyArg_ParseTupleAndKeywords(args, kwds, "sO&|O&", kwlist,
					 &name,
					 PyArray_IntpConverter, &shape,
					 PyArray_DescrConverter, &dtype))
		goto out;

	/* Check the type */
	if (!dtype)
		dtype = PyArray_DescrFromType(NPY_DEFAULT_TYPE);

	/* Now do the real thing */
	ret = do_create(name, shape.len, shape.ptr, dtype);

out:	/* Clean-up on exit */
	if (shape.ptr)
		PyDimMem_FREE(shape.ptr);

	return ret;
}
