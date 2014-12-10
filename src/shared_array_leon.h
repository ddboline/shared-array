/*
 *      Project: SharedArray
 * 
 *         File: src/shared_array_leon.h
 * 
 *  Description: PyLeonObject definition
 * 
 *     Author/s: Mathieu Mirmont <mat@parad0x.org>
 * 
 *   Created on: 09/12/2014
 * 
 */

#ifndef __SHARED_ARRAY_LEON_H__
#define __SHARED_ARRAY_LEON_H__

/*
 * Object definition
 */
typedef struct {
	PyObject_HEAD

	/* Address and size of the mapped memory region */
	void *data;
	size_t size;
} PyLeonObject;

/*
 * Object type
 */
extern PyTypeObject PyLeonObject_Type;

#endif /* !__SHARED_ARRAY_LEON_H__ */
