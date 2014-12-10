/* 
 * This file is part of SharedArray.
 * Copyright (C) 2014 Mathieu Mirmont <mat@parad0x.org>
 * 
 * SharedArray is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
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
