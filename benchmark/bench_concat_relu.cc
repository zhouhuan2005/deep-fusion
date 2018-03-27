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
#include <iostream>
#include <numeric>
#include <string>
#include "jitinfer.h"
#include "mkldnn.hpp"

#include <assert.h>
#include <stdint.h>
#include "common.h"
#include "src/jitinfer_common.h"
#include "src/util.h"

static int burning_iter = 50;
static int iter = 100;
static mkldnn::engine eng = mkldnn::engine(mkldnn::engine::cpu, 0);

std::unique_ptr<mkldnn::eltwise_forward::primitive_desc> get_relu_pd(
    const mkldnn::memory::desc md) {
  using namespace mkldnn;
  auto relu_desc = eltwise_forward::desc(
      prop_kind::forward_inference, algorithm::eltwise_relu, md, 0.f, 0.f);
  return std::unique_ptr<eltwise_forward::primitive_desc>(
      new eltwise_forward::primitive_desc(relu_desc, eng));
}

template <typename dtype>  // should be one of s32, s8, u8
void bench_mkldnn_concat(bool with_relu = false) {
  using namespace mkldnn;
  std::unique_ptr<primitive> fwd_concat, fwd_relu;
  std::unique_ptr<concat::primitive_desc> concat_pd;
  std::unique_ptr<eltwise_forward::primitive_desc> relu_pd;
  std::vector<primitive> pp_concat, pp_relu;  // pipeline

  // below is input
  int concat_dimension = 1;
  memory::format fmt = memory::format::nhwc;
  // note: src dims always is nchw format, only data layout can be nhwc
  std::vector<memory::dims> src_dims = {{4, 128, 224, 224}, {4, 256, 224, 224}};
  // std::vector<memory::dims> src_dims = {
  //  {4, 32, 4, 4},
  //  {4, 64, 4, 4}};

  // cal dst dims
  int oc = src_dims[0][concat_dimension];
  assert(src_dims[0].size() == 4);
  for (size_t i = 1; i < src_dims.size(); i++) {
    assert(src_dims[0].size() == src_dims[i].size());
    for (size_t dim = 0; dim < src_dims[i].size(); ++dim) {
      if (dim == (size_t)concat_dimension) {
        oc += src_dims[i][dim];
      } else {
        assert(src_dims[i][dim] == src_dims[0][dim]);
      }
    }
  }
  memory::dims dst_dims = {src_dims[0][0], oc, src_dims[0][2], src_dims[0][3]};
  memory::data_type data_type =
      jitinfer::jitinfer2mkldnn(jitinfer::data_traits<dtype>::dtype);

  // allocate srcs memory
  std::vector<memory::primitive_desc> srcs_pd;
  std::vector<memory> srcs;
  for (size_t i = 0; i < src_dims.size(); ++i) {
    auto desc = memory::desc(src_dims[i], data_type, fmt);
    auto mpd = memory::primitive_desc(desc, eng);
    auto src_memory = memory(mpd);
    srcs_pd.push_back(mpd);
    srcs.push_back(src_memory);
  }

  // dst memory
  auto dst_desc = memory::desc(dst_dims, data_type, fmt);
  concat_pd.reset(
      new concat::primitive_desc(dst_desc, concat_dimension, srcs_pd));
  auto dst = memory(concat_pd->dst_primitive_desc());

  // concat
  std::vector<primitive::at> inputs;
  for (size_t i = 0; i < srcs.size(); i++) {
    inputs.push_back(srcs[i]);
  }
  fwd_concat.reset(new concat(*concat_pd, inputs, dst));
  pp_concat.clear();
  pp_concat.push_back(*fwd_concat);

  if (with_relu) {
    // add relu
    relu_pd = get_relu_pd(dst_desc);
    fwd_relu.reset(new eltwise_forward(*relu_pd, dst, dst));
    pp_relu.clear();
    pp_relu.push_back(*fwd_relu);
  }

  for (auto i = 0; i < burning_iter; ++i) {
    jitinfer::util::clear_cache();
    stream(stream::kind::eager).submit(pp_concat).wait();

    if (with_relu) {
      stream(stream::kind::eager).submit(pp_relu).wait();
    }

    jitinfer::util::clear_cache();
  }

  // cal time
  double sum_concat = 0;
  double sum_relu = 0;
  for (auto i = 0; i < iter; ++i) {
    jitinfer::util::clear_cache();

    auto s1 = jitinfer::util::timer::get_current_ms();
    stream(stream::kind::eager).submit(pp_concat).wait();
    auto s2 = jitinfer::util::timer::get_current_ms();
    sum_concat += (s2 - s1);
    if (with_relu) {
      stream(stream::kind::eager).submit(pp_relu).wait();
      auto s3 = jitinfer::util::timer::get_current_ms();
      sum_relu += (s3 - s2);
    }

    jitinfer::util::clear_cache();
  }

  std::cout << "In";
  for (size_t i = 0; i < src_dims.size(); i++) {
    auto& dims = src_dims[i];
    printf("(%d, %d, %d, %d) ", dims[0], dims[1], dims[2], dims[3]);
  }
  printf("==> Out(%d, %d, %d, %d)\n",
         dst_dims[0],
         dst_dims[1],
         dst_dims[2],
         dst_dims[3]);

  auto avg_concat = sum_concat / (double)iter;
  auto avg_relu = sum_relu / (double)iter;
  std::cout << "MKL-DNN Concat" << (with_relu ? " + ReLU" : "");
  std::cout << " avg time (" << avg_concat;
  if (with_relu) {
    std::cout << " + " << avg_relu;
  }
  std::cout << ") " << avg_concat + avg_relu << "ms" << std::endl;
}

int main(int argc, char** argv) {
  // TODO: make test cases here
  try {
    bench_mkldnn_concat<jitinfer::s8>(true);
  } catch (mkldnn::error& e) {
    std::cerr << "status: " << e.status << std::endl;
    std::cerr << "message: " << e.message << std::endl;
  }
  return 0;
}
