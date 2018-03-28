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

#include "bench_common.h"
#include "src/jitinfer_thread.h"
#include "src/log.h"

namespace jitinfer {
namespace util {

#ifdef WITH_COLD_CACHE
dummy_memory::dummy_memory(size_t num_bytes) {
  int max_nthr = omp_get_max_threads();
  debug("Max OMP threads: %d", max_nthr);
  size_ = num_bytes * max_nthr;
  p_ = (unsigned char*)malloc(size_);
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

memory::dtype mkldnn2jitinfer(mkldnn::memory::data_type dt) {
  switch (dt) {
#define CASE(tp)                      \
  case mkldnn::memory::data_type::tp: \
    return memory::dtype::tp
    CASE(f32);
    CASE(s32);
    CASE(s8);
    CASE(u8);
#undef CASE
    default:
      error("Unkown type %d", dt);
  }
}

mkldnn::memory::data_type jitinfer2mkldnn(memory::dtype dt) {
  switch (dt) {
#define CASE(tp)                    \
  case jitinfer::memory::dtype::tp: \
    return mkldnn::memory::data_type::tp
    CASE(f32);
    CASE(s32);
    CASE(s8);
    CASE(u8);
#undef CASE
    default:
      error("Unkown type %d", dt);
  }
}

mkldnn::memory::dims jitinferDims2mkldnn(const memory::nchw_dims& nchwdims) {
  mkldnn::memory::dims out(nchwdims.size());
  for (size_t i = 0; i < out.size(); ++i) {
    out[i] = nchwdims[i];
  }
  return out;
}

std::unique_ptr<mkldnn::eltwise_forward::primitive_desc> get_mkldnn_relu_pd(
    const mkldnn::memory::desc md, const mkldnn::engine& eng) {
  using namespace mkldnn;
  auto relu_desc = eltwise_forward::desc(
      prop_kind::forward_inference, algorithm::eltwise_relu, md, 0.f, 0.f);
  return std::unique_ptr<eltwise_forward::primitive_desc>(
      new eltwise_forward::primitive_desc(relu_desc, eng));
}
}
}
