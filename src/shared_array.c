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

/* Module name */
static const char module_name[] = "SharedArray";

/* Module documentation string */
static const char module_docstring[] =
	"This module lets you share numpy arrays "
	"between several python interpreters";

/*
 * Module functions
 */
static PyMethodDef module_functions[] = {
	{ "create", (PyCFunction) shared_array_create,
	  METH_VARARGS | METH_KEYWORDS,
	  "Create a numpy array in shared memory" },

	{ "attach", (PyCFunction) shared_array_attach,
	  METH_VARARGS,
	  "Attach an existing numpy array from shared memory" },

	{ "delete", (PyCFunction) shared_array_delete,
	  METH_VARARGS,
	  "Delete an existing numpy array from shared memory" },

	{ NULL, NULL, 0, NULL }
};

#if PY_MAJOR_VERSION >= 3

/*
 * Module definition
 */
static struct PyModuleDef module_def = {
        PyModuleDef_HEAD_INIT,
	module_name,		/* m_name	*/
	module_docstring,	/* m_doc	*/
        -1,			/* m_size	*/
        module_functions,	/* m_methods	*/
        NULL,			/* m_reload	*/
        NULL,			/* m_traverse	*/
        NULL,			/* m_clear	*/
        NULL,			/* m_free	*/
};

/* Module creation function for python 3 */
#define CREATE_MODULE(NAME, FUNCTIONS, DOCSTRING)	\
	PyModule_Create(&module_def)
#else
/* Module creation function for python 2 */
#define CREATE_MODULE(NAME, FUNCTIONS, DOCSTRING)	\
	Py_InitModule3(NAME, FUNCTIONS, DOCSTRING)
#endif

/*
 * Module initialisation
 */
static PyObject *module_init(void)
{
	PyObject *m;

	/* Import numpy arrays */
	import_array();

	/* Register the module */
	if (!(m = CREATE_MODULE(module_name, module_functions, module_docstring)))
		return NULL;

	/* Register the Leon type */
	PyType_Ready(&PyLeonObject_Type);
	Py_INCREF(&PyLeonObject_Type);
	PyModule_AddObject(m, module_name, (PyObject *) &PyLeonObject_Type);

	return m;
}

/*
 * Python 2.7 compatibility blob
 */
#if PY_MAJOR_VERSION >= 3
PyMODINIT_FUNC PyInit_SharedArray(void)
{
	return module_init();
}
#else
PyMODINIT_FUNC initSharedArray(void)
{
	module_init();
}
#endif
