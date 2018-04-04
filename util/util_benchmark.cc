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
#include "util_benchmark.h"
#include "log.h"
#include "omp_thread.h"

namespace jitinfer {
namespace util {

#ifdef WITH_COLD_CACHE
dummy_memory::dummy_memory(size_t num_bytes) {
  int max_nthr = omp_get_max_threads();
  debug("Max OMP threads: %d", max_nthr);
  size_ = num_bytes * max_nthr;
  p_ = (unsigned char*)aligned_malloc(size_, 64);
}
dummy_memory::~dummy_memory() { free(p_); }
void dummy_memory::clear_cache() {
#pragma omp parallel for schedule(static)
  for (size_t i = 0; i < size_; ++i) {
    p_[i] = p_[i] * 2;
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

const char* dtype2str(memory::dtype dt) {
  using dtype = memory::dtype;
  switch (dt) {
    case dtype::f32:
      return "f32";
    case dtype::s32:
      return "s32";
    case dtype::s8:
      return "s8";
    case dtype::u8:
      return "u8";
    default:
      error_and_exit("Unknow data type");
      return NULL;
  }
}
}
}
