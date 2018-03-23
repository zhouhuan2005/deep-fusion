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

namespace jitinfer {

typedef float f32;
typedef int32_t s32;
typedef int8_t s8;
typedef uint8_t u8;

// Disable the copy and assignment operator for a class.
#ifndef DISABLE_COPY_AND_ASSIGN
#define DISABLE_COPY_AND_ASSIGN(classname)          \
private:                                            \
  classname(const classname &) = delete;            \
  classname(const classname &&) = delete;           \
  classname &operator=(const classname &) = delete; \
  classname &operator=(const classname &&) = delete
#endif

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

template <typename T, typename P>
inline bool one_of(T val, P item) {
  return val == item;
}
template <typename T, typename P, typename... Args>
inline bool one_of(T val, P item, Args... item_others) {
  return val == item || one_of(val, item_others...);
}

// TODO: getenv, dump jit bin

// TODO: add glog?
}
