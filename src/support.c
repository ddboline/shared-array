/* 
 * This file is part of SharedArray.
 * Copyright (C) 2014-2016 Mathieu Mirmont <mat@parad0x.org>
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

#include <sys/types.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <string.h>
#include <errno.h>
#include "shared_array.h"

#define PREFIX_FILE	"file://"
#define PREFIX_SHM	"shm://"

/*
 * Open a file or shm to be used as data storage
 */
int open_file(const char *name, int flags, mode_t mode)
{
	/* Files */
	if (!strncmp(name, PREFIX_FILE, strlen(PREFIX_FILE)))
		return open(name + strlen(PREFIX_FILE), flags, mode);

	/* POSIX SHM */
	if (!strncmp(name, PREFIX_SHM, strlen(PREFIX_SHM)))
		return shm_open(name + strlen(PREFIX_SHM), flags, mode);

	/* For backward compatibility, assume POSIX SHM by default */
	if (!strstr(name, "://"))
		return shm_open(name, flags, mode);
	
	errno = EINVAL;
	return -1;
}

/*
 * Unlink a file or shm
 */
int unlink_file(const char *name)
{
	/* Files */
	if (!strncmp(name, PREFIX_FILE, strlen(PREFIX_FILE)))
		return unlink(name + strlen(PREFIX_FILE));

	/* POSIX SHM */
	if (!strncmp(name, PREFIX_SHM, strlen(PREFIX_SHM)))
		return shm_unlink(name + strlen(PREFIX_SHM));

	/* For backward compatibility, assume POSIX SHM by default */
	if (!strstr(name, "://"))
		return shm_unlink(name);
	
	errno = EINVAL;
	return -1;
}
