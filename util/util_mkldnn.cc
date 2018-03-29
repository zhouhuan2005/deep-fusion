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
#include "util_mkldnn.h"
#include "log.h"

namespace jitinfer {
namespace util {
// TODO: get concat pd
std::unique_ptr<mkldnn::eltwise_forward::primitive_desc> get_mkldnn_relu_pd(
    const mkldnn::memory::desc md, const mkldnn::engine& eng) {
  using namespace mkldnn;
  auto relu_desc = eltwise_forward::desc(
      prop_kind::forward_inference, algorithm::eltwise_relu, md, 0.f, 0.f);
  return std::unique_ptr<eltwise_forward::primitive_desc>(
      new eltwise_forward::primitive_desc(relu_desc, eng));
}

namespace exchange {
memory::dtype dtype(mkldnn::memory::data_type dt) {
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
      error_and_exit("Unkown type %d", dt);
  }
}

mkldnn::memory::data_type dtype(memory::dtype dt) {
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
      error_and_exit("Unkown type %d", dt);
  }
}

mkldnn::memory::dims dims(const memory::nchw_dims& nchwdims) {
  mkldnn::memory::dims out(nchwdims.size());
  for (size_t i = 0; i < out.size(); ++i) {
    out[i] = nchwdims[i];
  }
  return out;
}
}
}
}
