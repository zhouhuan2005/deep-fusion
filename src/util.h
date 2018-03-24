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

#pragma once

#include <assert.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef WIN32
#include <malloc.h>
#include <windows.h>
#endif

namespace jitinfer {

namespace util {
// TODO: use forward template
template <typename T>
size_t array_product(const T *p, size_t num) {
  size_t out = 1;
  for (size_t i = 0; i < num; ++i) {
    out *= size_t(p[i]);
  }
  return out;
}
}

void *malloc(size_t size, int alignment);

void free(void *p);

template <typename T, typename P>
inline bool one_of(T val, P item) {
  return val == item;
}
template <typename T, typename P, typename... Args>
inline bool one_of(T val, P item, Args... item_others) {
  return val == item || one_of(val, item_others...);
}

// TODO: optimize jit dump code and getenv
int _getenv(char *value, const char *name, int length);
bool jit_dump_code();
}
