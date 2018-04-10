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

#include <gflags/gflags.h>
#include <mkldnn.hpp>
#include <sstream>
#include "jitinfer.h"
#include "log.h"
#include "util_benchmark.h"
#include "util_mkldnn.h"

DEFINE_int32(burning_iter, 50, "Burning iterations");
DEFINE_int32(iter, 100, "Iterations for average");
DEFINE_int32(n, 4, "Number of images (==batch size), 'n' in 'nchw'.");
DEFINE_int32(h, 100, "Height of images, 'h' in 'nchw'");
DEFINE_int32(w, 100, "Width of images, 'w' in 'nchw'");
DEFINE_string(c, "", "Input channels, for example, -c=64,64,32");
DEFINE_string(dtype, "s8", "Data type");
DEFINE_bool(post_relu, true, "Post ReLU after Concat");

static mkldnn::engine eng = mkldnn::engine(mkldnn::engine::cpu, 0);

struct bench_params {
  // @note: dims always write as nchw, but acutal run format is nhwc
  std::vector<jitinfer::memory::nchw_dims> srcs_dims;
};

void bench_mkldnn_concat(const std::vector<mkldnn::memory::dims>& srcs_dims,
                         const mkldnn::memory::dims& dst_dims,
                         mkldnn::memory::data_type dt,
                         bool post_relu) {
  using namespace mkldnn;
  std::unique_ptr<primitive> fwd_concat, fwd_relu;
  std::unique_ptr<concat::primitive_desc> concat_pd;
  std::unique_ptr<eltwise_forward::primitive_desc> relu_pd;
  std::vector<primitive> pp_concat, pp_relu;  // pipeline

  // below is input
  int concat_dimension = 1;
  memory::format fmt = memory::format::nhwc;
  // allocate srcs memory
  std::vector<memory::primitive_desc> srcs_pd;
  std::vector<memory> srcs;
  for (size_t i = 0; i < srcs_dims.size(); ++i) {
    auto desc = memory::desc(srcs_dims[i], dt, fmt);
    auto mpd = memory::primitive_desc(desc, eng);
    auto src_memory = memory(mpd);
    srcs_pd.push_back(mpd);
    srcs.push_back(src_memory);
  }
  // dst memory
  auto dst_desc = memory::desc(dst_dims, dt, fmt);
  concat_pd.reset(
      new concat::primitive_desc(dst_desc, concat_dimension, srcs_pd));
  auto dst = memory(concat_pd->dst_primitive_desc());

  // concat
  std::vector<primitive::at> inputs;
  for (size_t i = 0; i < srcs.size(); i++) {
    inputs.push_back(srcs[i]);
  }
  fwd_concat.reset(new mkldnn::concat(*concat_pd, inputs, dst));
  pp_concat.clear();
  pp_concat.push_back(*fwd_concat);

  if (post_relu) {
    // add relu
    relu_pd = jitinfer::util::get_mkldnn_relu_pd(dst_desc, eng);
    fwd_relu.reset(new eltwise_forward(*relu_pd, dst, dst));
    pp_relu.clear();
    pp_relu.push_back(*fwd_relu);
  }

  for (auto i = 0; i < FLAGS_burning_iter; ++i) {
    jitinfer::util::clear_cache();
    stream(stream::kind::eager).submit(pp_concat).wait();
    if (post_relu) {
      stream(stream::kind::eager).submit(pp_relu).wait();
    }
    jitinfer::util::clear_cache();
  }

  // cal time
  double sum_concat = 0;
  double sum_relu = 0;
  for (auto i = 0; i < FLAGS_iter; ++i) {
    jitinfer::util::clear_cache();
    auto s1 = jitinfer::util::timer::get_current_ms();
    stream(stream::kind::eager).submit(pp_concat).wait();
    auto s2 = jitinfer::util::timer::get_current_ms();
    sum_concat += (s2 - s1);
    if (post_relu) {
      stream(stream::kind::eager).submit(pp_relu).wait();
      auto s3 = jitinfer::util::timer::get_current_ms();
      sum_relu += (s3 - s2);
    }
    jitinfer::util::clear_cache();
  }

  auto avg_concat = sum_concat / (double)FLAGS_iter;
  auto avg_relu = sum_relu / (double)FLAGS_iter;
  std::ostringstream oss;
  oss << "MKL-DNN Concat" << (post_relu ? " + ReLU" : "") << " avg time ("
      << avg_concat;
  if (post_relu) {
    oss << " + " << avg_relu;
  }
  oss << ") " << avg_concat + avg_relu << " ms";
  info("%s", oss.str().c_str());
}

void bench_jitinfer_concat(
    const std::vector<jitinfer::memory::nchw_dims>& srcs_dims,
    const jitinfer::memory::nchw_dims& dst_dims,
    jitinfer::memory::dtype dt,
    bool post_relu) {
  using namespace jitinfer;

  std::vector<std::unique_ptr<memory>> srcs(srcs_dims.size());
  std::unique_ptr<memory> dst;
  memory::format fmt = memory::format::nhwc;
  for (size_t i = 0; i < srcs.size(); ++i) {
    srcs[i].reset(new memory(srcs_dims[i], fmt, dt));
  }
  dst.reset(new memory(dst_dims, fmt, dt));

  auto c = concat(srcs, dst, post_relu);

  for (auto i = 0; i < FLAGS_burning_iter; ++i) {
    jitinfer::util::clear_cache();
    c->submit();
    jitinfer::util::clear_cache();
  }

  double sum_concat = 0;
  for (auto i = 0; i < FLAGS_iter; ++i) {
    jitinfer::util::clear_cache();
    auto s1 = jitinfer::util::timer::get_current_ms();
    c->submit();
    auto s2 = jitinfer::util::timer::get_current_ms();
    sum_concat += (s2 - s1);
    jitinfer::util::clear_cache();
  }

  std::ostringstream oss;
  oss << "JitInfer Concat" << (post_relu ? "_ReLU" : "")
      << " avg time: " << sum_concat / (double)FLAGS_iter << " ms";
  info("%s", oss.str().c_str());
}

void bench_both(const bench_params& p,
                jitinfer::memory::dtype dt,
                bool post_relu) {
  auto srcs_dims = p.srcs_dims;
  jitinfer::memory::nchw_dims dst_dims;  // given as nchw
  std::vector<mkldnn::memory::dims> mkldnn_srcs_dims(srcs_dims.size());
  mkldnn::memory::dims mkldnn_dst_dims;
  dst_dims[0] = srcs_dims[0][0];
  dst_dims[1] = 0;
  dst_dims[2] = srcs_dims[0][2];
  dst_dims[3] = srcs_dims[0][3];
  std::ostringstream oss;
  info("==========================================");
  oss << "Benchmark with data type " << jitinfer::util::dtype2str(dt)
      << (post_relu ? ", with ReLU" : " without ReLU");
  oss << "\nData sizes: In";
  for (size_t i = 0; i < srcs_dims.size(); i++) {
    const auto& dims = srcs_dims[i];
    check_eq(dims.size(), 4);
    for (size_t dim = 0; dim < dims.size(); ++dim) {
      if (dim == 1) {
        dst_dims[1] += dims[dim];  // oc
      } else {
        check_eq(dst_dims[dim], dims[dim]);
      }
    }
    oss << "(" << dims[0] << ", " << dims[1] << ", " << dims[2] << ", "
        << dims[3] << ")@NCHW, ";
    mkldnn_srcs_dims[i] = jitinfer::util::exchange::dims(dims);
  }
  oss << "==> Out(" << dst_dims[0] << ", " << dst_dims[1] << ", " << dst_dims[2]
      << ", " << dst_dims[3] << ")@NCHW";
  info("%s", oss.str().c_str());
  mkldnn_dst_dims = jitinfer::util::exchange::dims(dst_dims);
  bench_mkldnn_concat(mkldnn_srcs_dims,
                      mkldnn_dst_dims,
                      jitinfer::util::exchange::dtype(dt),
                      post_relu);
  bench_jitinfer_concat(srcs_dims, dst_dims, dt, post_relu);
}

int main(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  // only run if given some input channels
  // for example:
  // bench_concat -n 3 -c 16,16,64 -h 4 -w 6 -dtype s8 -post_relu
  if (!FLAGS_c.empty()) {
    auto ics = jitinfer::util::split(FLAGS_c);
    bench_params test_case;
    test_case.srcs_dims.resize(ics.size());
    for (size_t i = 0; i < ics.size(); ++i) {
      auto& dims = test_case.srcs_dims[i];
      // always load input dims as nchw format
      dims[0] = FLAGS_n;
      dims[1] = std::stoi(ics[i]);
      dims[2] = FLAGS_h;
      dims[3] = FLAGS_w;
    }
    bench_both(
        test_case, jitinfer::util::str2dtype(FLAGS_dtype), FLAGS_post_relu);
    return 0;
  }

  // nothing input, then run some default cases
  bench_params default_cases[] = {
      {{{4, 128, 244, 244}, {4, 256, 244, 244}}},  // 64x
      {{{4, 64, 64, 64}, {4, 96, 64, 64}}},        // 32x
      {{{4, 16, 9, 9}, {4, 64, 9, 9}}}             // 16x
  };
  jitinfer::memory::dtype dtypes[] = {jitinfer::memory::dtype::s8,
                                      jitinfer::memory::dtype::s32,
                                      jitinfer::memory::dtype::f32};
  size_t param_sz = sizeof(default_cases) / sizeof(bench_params);
  size_t dt_sz = sizeof(dtypes) / sizeof(jitinfer::memory::dtype);
  for (size_t i = 0; i < param_sz; ++i) {
    for (auto post_relu : {true, false}) {
      for (size_t j = 0; j < dt_sz; ++j) {
        bench_both(default_cases[i], dtypes[j], post_relu);
      }
    }
  }

  return 0;
}
