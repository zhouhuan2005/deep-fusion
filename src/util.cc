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
#include "jitinfer_thread.h"
#include "log.h"

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
namespace util {

#ifdef WITH_COLD_CACHE
dummy_memory::dummy_memory(size_t num_bytes) {
  int max_nthr = omp_get_max_threads();
  debug("Max OMP threads: %d", max_nthr);
  size_ = num_bytes * max_nthr;
  p_ = (unsigned char *)malloc(size_);
}

dummy_memory::~dummy_memory() { free(p_); }

void dummy_memory::clear_cache() {
#pragma omp parallel for
  for (size_t i = 0; i < size_; ++i) {
    // disable gcc optimize
    volatile unsigned char write = 3, read = 4;
    *(p_ + i) = write;
    read = p_[i];
  }
}

// skx, L3: 1.375MB * n
//      L2: 1MB
//      L1: 32KB
constexpr size_t PAGE_2MB = 2 * 1024 * 1024;
static dummy_memory dummy_mem(PAGE_2MB);
void clear_cache() { dummy_mem.clear_cache(); }
#else
// hot cache, do nothing
void clear_cache() { ; }
#endif

namespace env {
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

static bool profiling = false;
// when need profiling
// 1. cmake -DWITH_VERBOSE=ON
// 2. export JITINFER_VERBOSE=1
bool profiling_time() {
  static bool initialized = false;
  if (!initialized) {
    const int len = 2;
    char env_dump[len] = {0};
    profiling =
        _getenv(env_dump, "JITINFER_VERBOSE", len) == 1 && atoi(env_dump) == 1;
    initialized = true;
  }
  return profiling;
}

static bool dump_jit_code = false;
// when need dump jit code
// 1. cmake -DCMAKE_BUILD_TYPE=DEBUG
// 2. export JITINFER_DUMP_CODE
bool jit_dump_code() {
  static bool initialized = false;
  if (!initialized) {
    const int len = 2;
    char env_dump[len] = {0};
    dump_jit_code = _getenv(env_dump, "JITINFER_DUMP_CODE", len) == 1 &&
                    atoi(env_dump) == 1;
    initialized = true;
  }
  return dump_jit_code;
}
}
}
}
