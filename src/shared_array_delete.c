/*
 *      Project: SharedArray
 * 
 *         File: shared_array_delete.c
 * 
 *  Description: Delete a numpy array from shared memory
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
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include "shared_array.h"

/*
 * Delete a numpy array from shared memory
 */
static PyObject *do_delete(const char *name)
{
	struct array_meta meta;
	int fd;

	/* Open the shm block */
	if ((fd = shm_open(name, O_RDWR, 0)) < 0)
		return PyErr_SetFromErrno(PyExc_RuntimeError);

	/* Read the meta data structure */
	if (read(fd, &meta, sizeof (meta)) != sizeof (meta)) {
		close(fd);
		return PyErr_SetFromErrno(PyExc_RuntimeError);
	}

	/* Close the shm block */
	close(fd);

	/* Check the meta data */
	if (strncmp(meta.magic, SHARED_ARRAY_MAGIC, sizeof (meta.magic))) {
		PyErr_SetString(PyExc_RuntimeError, "No SharedArray at this address");
		return NULL;
	}

	/* Unlink the shm block */
	if (shm_unlink(name) < 0)
		return PyErr_SetFromErrno(PyExc_RuntimeError);

	Py_RETURN_NONE;
}

/*
 * Method: SharedArray.delete()
 */
PyObject *shared_array_delete(PyObject *self, PyObject *args)
{
	const char *name;

	/* Parse the arguments */
	if (!PyArg_ParseTuple(args, "s", &name))
		return NULL;

	/* Now do the real thing */
	return do_delete(name);
}
