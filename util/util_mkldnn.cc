/*******************************************************************************
* Copyright 2017-2018 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#include "util_mkldnn.h"
#include "log.h"

namespace deepfusion {
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
  case deepfusion::memory::dtype::tp: \
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
