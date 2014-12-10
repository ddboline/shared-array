/*
 *      Project: SharedArray
 * 
 *         File: src/shared_array_leon.c
 * 
 *  Description: PyLeonObject definition
 * 
 *     Author/s: Mathieu Mirmont <mat@parad0x.org>
 * 
 *   Created on: 09/12/2014
 * 
 */

#define NPY_NO_DEPRECATED_API	NPY_1_8_API_VERSION
#include <Python.h>
#include <sys/mman.h>
#include "shared_array_leon.h"

/*
 * Deallocation function
 */
static void leon_dealloc(PyLeonObject *op)
{
	/* Unmap the data */
	if (munmap(op->data, op->size) < 0)
		PyErr_SetFromErrno(PyExc_RuntimeError);
}

/*
 * SharedArrayObject type definition
 */
PyTypeObject PyLeonObject_Type = {
	PyVarObject_HEAD_INIT(NULL, 0)
	"shared_array.leon",			/* tp_name		*/
	sizeof (PyLeonObject),			/* tp_basicsize		*/
	0,					/* tp_itemsize		*/
	(destructor) leon_dealloc,		/* tp_dealloc		*/
	0,					/* tp_print		*/
	0,					/* tp_getattr		*/
	0,					/* tp_setattr		*/
	0,					/* tp_reserved		*/
	0,					/* tp_repr		*/
	0,					/* tp_as_number		*/
	0,					/* tp_as_sequence	*/
	0,					/* tp_as_mapping	*/
	0,					/* tp_hash		*/
	0,					/* tp_call		*/
	0,					/* tp_str		*/
	0,					/* tp_getattro		*/
	0,					/* tp_setattro		*/
	0,					/* tp_as_buffer		*/
	Py_TPFLAGS_DEFAULT,			/* tp_flags		*/
	0,					/* tp_doc		*/
	0,					/* tp_traverse		*/
	0,					/* tp_clear		*/
	0,					/* tp_richcompare	*/
	0,					/* tp_weaklistoffset	*/
	0,					/* tp_iter		*/
	0,					/* tp_iternext		*/
	0,					/* tp_methods		*/
	0,					/* tp_members		*/
	0,					/* tp_getset		*/
	0,					/* tp_base		*/
	0,					/* tp_dict		*/
	0,					/* tp_descr_get		*/
	0,					/* tp_descr_set		*/
	0,					/* tp_dictoffset	*/
	0,					/* tp_init		*/
	0,					/* tp_alloc		*/
	0,					/* tp_new		*/
	0,					/* tp_free		*/
	0,					/* tp_is_gc		*/
	0,					/* tp_bases		*/
	0,					/* tp_mro		*/
	0,					/* tp_cache		*/
	0,					/* tp_subclasses	*/
	0,					/* tp_weaklist		*/
	0,					/* tp_del		*/
	0,					/* tp_version_tag	*/
};
