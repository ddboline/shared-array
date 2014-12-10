/*
 *      Project: SharedArray
 * 
 *         File: src/shared_array.c
 * 
 *  Description: SharedArray module definition
 * 
 *     Author/s: Mathieu Mirmont <mat@parad0x.org>
 * 
 *   Created on: 08/12/2014
 * 
 */

#define NPY_NO_DEPRECATED_API	NPY_1_8_API_VERSION
#define PY_ARRAY_UNIQUE_SYMBOL	SHARED_ARRAY_ARRAY_API

#include <Python.h>
#include <numpy/arrayobject.h>
#include "shared_array.h"
#include "shared_array_leon.h"

/*
 * Module functions
 */
static PyMethodDef module_functions[] = {
	{ "create", (PyCFunction) shared_array_create,
	  METH_VARARGS | METH_KEYWORDS, "Create a numpy array in shared memory" },
	{ "attach", (PyCFunction) shared_array_attach,
	  METH_VARARGS, "Attach a numpy array from shared memory" },
	{ "delete", (PyCFunction) shared_array_delete,
	  METH_VARARGS, "Delete a numpy array from shared memory" },
	{ NULL, NULL, 0, NULL }
};

/*
 * Module definition
 */
static struct PyModuleDef module_def = {
        PyModuleDef_HEAD_INIT,
	"SharedArray",
	"This module lets you share numpy arrays between several python interpreters",
        -1,
        module_functions,
        NULL,
        NULL,
        NULL,
        NULL,
};

/*
 * Module initialisation
 */
PyMODINIT_FUNC PyInit_SharedArray(void)
{
	PyObject *m;

	/* Ready our type */
	if (PyType_Ready(&PyLeonObject_Type) < 0)
		return NULL;

	/* Register our module */
	if (!(m = PyModule_Create(&module_def)))
		return NULL;

	/* Register our type */
	Py_INCREF(&PyLeonObject_Type);
	PyModule_AddObject(m, "SharedArray", (PyObject *) &PyLeonObject_Type);
	
	/* Import numpy arrays */
	import_array();
	return m;
}
