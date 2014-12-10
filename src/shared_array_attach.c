/*
 *      Project: SharedArray
 * 
 *         File: src/shared_array_attach.c
 * 
 *  Description: Attach a numpy array from shared memory
 * 
 *     Author/s: Mathieu Mirmont <mat@parad0x.org>
 * 
 *   Created on: 10/12/2014
 * 
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
#include "shared_array_leon.h"

/*
 * Attach a numpy array from shared memory
 */
static PyObject *do_attach(const char *name)
{
	struct array_meta meta;
	void *data;
	int fd;
	PyObject *ret;
	PyLeonObject *leon;

	/* Open the shm block */
	if ((fd = shm_open(name, O_RDWR, 0)) < 0)
		return PyErr_SetFromErrno(PyExc_RuntimeError);

	/* Read the meta data structure */
	if (read(fd, &meta, sizeof (meta)) != sizeof (meta)) {
		close(fd);
		return PyErr_SetFromErrno(PyExc_RuntimeError);
	}

	/* Check the meta data */
	if (strncmp(meta.magic, SHARED_ARRAY_MAGIC, sizeof (meta.magic))) {
		close(fd);
		PyErr_SetString(PyExc_RuntimeError, "No SharedArray at this address");
		return NULL;
	}

	/* Check the number of dimensions */
	if (meta.ndims > SHARED_ARRAY_NDIMS_MAX) {
		close(fd);
		PyErr_SetString(PyExc_RuntimeError, "Too many dimensions, recompile SharedArray!");
		return NULL;
	}
	
	/* Map the array data */
	data = mmap(NULL, meta.size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
	close(fd);
	if (data == MAP_FAILED)
		return PyErr_SetFromErrno(PyExc_RuntimeError);

	/* Summon Leon */
	leon = PyObject_MALLOC(sizeof (*leon));
	PyObject_INIT((PyObject *) leon, &PyLeonObject_Type);
	leon->data = data;
	leon->size = meta.size;

	/* Create the array object */
	ret = PyArray_SimpleNewFromData(meta.ndims, meta.dims, meta.typenum, data + sizeof (meta));

	/* Attach Leon to the array */
	PyArray_SetBaseObject((PyArrayObject *) ret, (PyObject *) leon);
	return ret;
}

/*
 * Method: SharedArray.attach()
 */
PyObject *shared_array_attach(PyObject *self, PyObject *args)
{
	const char *name;

	/* Parse the arguments */
	if (!PyArg_ParseTuple(args, "s", &name))
		return NULL;

	/* Now do the real thing */
	return do_attach(name);
}
