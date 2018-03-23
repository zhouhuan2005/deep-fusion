/*******************************************************************************
 * This file is part of the JITInfer (https://github.com/tensor-tang/jitinfer).
 * Copyright (c) 2018 Tensor Tang.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <http://www.gnu.org/licenses/>.
 ******************************************************************************/
#include "util.h"

namespace jitinfer {

void *malloc(size_t size, int alignment) {
  void *ptr;

#ifdef _WIN32
  ptr = _aligned_malloc(size, alignment);
  int rc = ptr ? 0 : -1;
#else
  int rc = ::posix_memalign(&ptr, alignment, size);
#endif

  return (rc == 0) ? ptr : 0;
}

void free(void *p) {
#ifdef _WIN32
  _aligned_free(p);
#else
  ::free(p);
#endif
}

// TODO: optimize jit dump code and getenv
int _getenv(char *value, const char *name, int length) {
  int result = 0;
  int last_idx = 0;
  if (length > 1) {
    int value_length = 0;
#ifdef _WIN32
    value_length = GetEnvironmentVariable(name, value, length);
    if (value_length >= length) {
      result = -value_length;
    } else {
      last_idx = value_length;
      result = value_length;
    }
#else
    char *buffer = getenv(name);
    if (buffer != NULL) {
      value_length = strlen(buffer);
      if (value_length >= length) {
        result = -value_length;
      } else {
        strncpy(value, buffer, value_length);
        last_idx = value_length;
        result = value_length;
      }
    }
#endif
  }
  value[last_idx] = '\0';
  return result;
}

static bool dump_jit_code;

bool jit_dump_code() {
  static bool initialized = false;
  if (!initialized) {
    const int len = 2;
    char env_dump[len] = {0};
    dump_jit_code =
        _getenv(env_dump, "MKLDNN_JIT_DUMP", len) == 1 && atoi(env_dump) == 1;
    initialized = true;
  }
  return dump_jit_code;
}
}
