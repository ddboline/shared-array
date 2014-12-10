/*
 *      Project: SharedArray
 * 
 *         File: src/shared_array.h
 * 
 *  Description: SharedArray module definition
 * 
 *     Author/s: Mathieu Mirmont <mat@parad0x.org>
 * 
 *   Created on: 08/12/2014
 * 
 */

#ifndef __SHARED_ARRAY_H__
#define __SHARED_ARRAY_H__

#include <Python.h>
#include <numpy/arrayobject.h>

/* Magic header */
#define SHARED_ARRAY_MAGIC	"[SharedArray]"

/* Maximum number of dimensions */
#define SHARED_ARRAY_NDIMS_MAX	16

/* Array metadata */
struct array_meta {
	char magic[16];
	uint32_t ndims;
	uint32_t dims[SHARED_ARRAY_NDIMS_MAX];
} __attribute__ ((packed));

/* Module functions */
extern PyObject *shared_array_create(PyObject *self, PyObject *args, PyObject *kw);

#endif /* !__SHARED_ARRAY_H__ */
