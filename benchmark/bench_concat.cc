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
#include <mkldnn.hpp>
#include <sstream>
#include "jitinfer.h"
#include "log.h"
#include "util_benchmark.h"
#include "util_mkldnn.h"

static int burning_iter = 50;
static int iter = 100;
static mkldnn::engine eng = mkldnn::engine(mkldnn::engine::cpu, 0);

struct bench_params {
  // @note: dims always write as nchw, but acutal run format is nhwc
  std::vector<jitinfer::memory::nchw_dims> srcs_dims;
};

static bench_params default_cases[] = {
    {{{4, 128, 244, 244}, {4, 256, 244, 244}}},  // 64x
    {{{4, 64, 64, 64}, {4, 96, 64, 64}}},        // 32x
    {{{4, 16, 9, 9}, {4, 64, 9, 9}}}             // 16x
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

  for (auto i = 0; i < burning_iter; ++i) {
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
  for (auto i = 0; i < iter; ++i) {
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

  auto avg_concat = sum_concat / (double)iter;
  auto avg_relu = sum_relu / (double)iter;
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

  for (auto i = 0; i < burning_iter; ++i) {
    jitinfer::util::clear_cache();
    c->submit();
    jitinfer::util::clear_cache();
  }

  double sum_concat = 0;
  for (auto i = 0; i < iter; ++i) {
    jitinfer::util::clear_cache();

    auto s1 = jitinfer::util::timer::get_current_ms();
    c->submit();
    auto s2 = jitinfer::util::timer::get_current_ms();
    sum_concat += (s2 - s1);
    jitinfer::util::clear_cache();
  }

  std::ostringstream oss;
  oss << "JitInfer Concat" << (post_relu ? "_ReLU" : "")
      << " avg time: " << sum_concat / (double)iter << " ms";
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
  oss << "Benchmark with data type: " << dt
      << (post_relu ? " with ReLU" : " without ReLU");
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
  size_t param_sz = sizeof(default_cases) / sizeof(bench_params);
  bench_params* pm = default_cases;
  jitinfer::memory::dtype dtypes[] = {jitinfer::memory::dtype::s32,
                                      jitinfer::memory::dtype::s8};
  size_t dt_sz = sizeof(dtypes) / sizeof(jitinfer::memory::dtype);
  if (argc > 1) {
    // TODO: enable get user param and dtype from outside
    // pm = ;
    // dtypes[0] = ;
    param_sz = 1;
    dt_sz = 1;
  }

  for (size_t i = 0; i < param_sz; ++i) {
    for (size_t j = 0; j < dt_sz; ++j) {
      for (auto post_relu : {true, false}) {
        bench_both(pm[i], dtypes[j], post_relu);
      }
    }
  }

  return 0;
}
