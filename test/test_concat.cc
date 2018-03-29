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
#include "util_jitinfer.h"
#include "util_mkldnn.h"
#include "util_test.h"

namespace jitinfer {

using memory = jitinfer::memory;
using format = jitinfer::memory::format;

struct test_concat_params {
  std::vector<memory::nchw_dims> srcs_dims;
  memory::nchw_dims dst_dims;
};

template <typename dtype>
class test_concat : public ::testing::TestWithParam<test_concat_params> {
  void check_result(const test_concat_params& pm,
                    const std::vector<std::unique_ptr<memory>>& srcs,
                    const std::unique_ptr<memory>& dst,
                    bool post_relu) {
    mkldnn::engine eng = mkldnn::engine(mkldnn::engine::cpu, 0);

    std::unique_ptr<mkldnn::primitive> fwd_concat, fwd_relu;
    std::unique_ptr<mkldnn::concat::primitive_desc> concat_pd;
    std::unique_ptr<mkldnn::eltwise_forward::primitive_desc> relu_pd;
    std::vector<mkldnn::primitive> pp_concat;

    // below is input
    int concat_dimension = 1;
    mkldnn::memory::format fmt = mkldnn::memory::format::nhwc;
    auto mkldnn_dt = util::exchange::dtype(dst->data_type());
    // allocate srcs memory
    std::vector<mkldnn::memory::primitive_desc> srcs_pd;
    std::vector<mkldnn::memory> mkldnn_srcs;
    for (size_t i = 0; i < srcs.size(); ++i) {
      auto mkldnn_dims = util::exchange::dims(pm.srcs_dims[i]);
      auto desc = mkldnn::memory::desc(mkldnn_dims, mkldnn_dt, fmt);
      auto mpd = mkldnn::memory::primitive_desc(desc, eng);
      auto src_memory = mkldnn::memory(mpd);
      assert(srcs[i]->size() == src_memory.get_size() / sizeof(dtype));
      util::copy_array<dtype>((dtype*)(src_memory.get_data_handle()),
                              (dtype*)(srcs[i]->data()),
                              srcs[i]->size());
      srcs_pd.push_back(mpd);
      mkldnn_srcs.push_back(src_memory);
    }
    // dst memory
    auto dst_desc =
        mkldnn::memory::desc(util::exchange::dims(pm.dst_dims), mkldnn_dt, fmt);
    concat_pd.reset(new mkldnn::concat::primitive_desc(
        dst_desc, concat_dimension, srcs_pd));
    auto mkldnn_dst = mkldnn::memory(concat_pd->dst_primitive_desc());

    // concat
    std::vector<mkldnn::primitive::at> inputs;
    for (size_t i = 0; i < mkldnn_srcs.size(); i++) {
      inputs.push_back(mkldnn_srcs[i]);
    }
    fwd_concat.reset(new mkldnn::concat(*concat_pd, inputs, mkldnn_dst));
    pp_concat.clear();
    pp_concat.push_back(*fwd_concat);

    if (post_relu) {
      // add relu
      relu_pd = jitinfer::util::get_mkldnn_relu_pd(dst_desc, eng);
      fwd_relu.reset(
          new mkldnn::eltwise_forward(*relu_pd, mkldnn_dst, mkldnn_dst));
      pp_concat.push_back(*fwd_relu);
    }

    mkldnn::stream(mkldnn::stream::kind::eager).submit(pp_concat).wait();
    dtype* ref_data = (dtype*)(mkldnn_dst.get_data_handle());
    dtype* jit_data = (dtype*)(dst->data());
    util::compare_array<dtype>(jit_data, ref_data, dst->size());
  }

protected:
  virtual void SetUp() {
    test_concat_params p =
        ::testing::TestWithParam<test_concat_params>::GetParam();
    auto dt = util::type2dtype<dtype>::dtype;
    std::vector<std::unique_ptr<memory>> srcs(p.srcs_dims.size());
    std::unique_ptr<memory> dst;
    memory::format fmt = format::nhwc;
    for (size_t i = 0; i < p.srcs_dims.size(); ++i) {
      srcs[i].reset(new memory(p.srcs_dims[i], fmt, dt));
      util::fill_data<dtype>(static_cast<dtype*>(srcs[i]->data()),
                             srcs[i]->size());
    }
    dst.reset(new memory(p.dst_dims, fmt, dt));

    for (bool post_relu : {true, false}) {
      auto c = concat(srcs, dst, post_relu);
      c->submit();
      check_result(p, srcs, dst, post_relu);
    }
  }
};

using test_concat_f32 = test_concat<f32>;
using test_concat_s32 = test_concat<s32>;
using test_concat_s8 = test_concat<s8>;
using test_concat_u8 = test_concat<u8>;

TEST_P(test_concat_f32, TestsConcat) {}
TEST_P(test_concat_s32, TestsConcat) {}
TEST_P(test_concat_s8, TestsConcat) {}
TEST_P(test_concat_u8, TestsConcat) {}

// @note: the srcs and dst are always given as nchw
INSTANTIATE_TEST_CASE_P(
    TestConcat,
    test_concat_f32,
    ::testing::Values(
        test_concat_params{{{2, 64, 1, 1}, {2, 96, 1, 1}}, {2, 160, 1, 1}},
        test_concat_params{{{2, 64, 4, 4}, {2, 32, 4, 4}}, {2, 96, 4, 4}},
        test_concat_params{{{2, 256, 16, 16}, {2, 256, 16, 16}},
                           {2, 512, 16, 16}}));

INSTANTIATE_TEST_CASE_P(
    TestConcat,
    test_concat_s32,
    ::testing::Values(
        test_concat_params{{{2, 64, 1, 1}, {2, 96, 1, 1}}, {2, 160, 1, 1}},
        test_concat_params{{{2, 64, 4, 4}, {2, 32, 4, 4}}, {2, 96, 4, 4}},
        test_concat_params{{{2, 256, 16, 16}, {2, 256, 16, 16}},
                           {2, 512, 16, 16}}));

INSTANTIATE_TEST_CASE_P(
    TestConcat,
    test_concat_s8,
    ::testing::Values(
        test_concat_params{{{2, 64, 1, 1}, {2, 96, 1, 1}}, {2, 160, 1, 1}},
        test_concat_params{{{2, 64, 4, 4}, {2, 32, 4, 4}}, {2, 96, 4, 4}},
        test_concat_params{{{2, 256, 16, 16}, {2, 256, 16, 16}},
                           {2, 512, 16, 16}}));

INSTANTIATE_TEST_CASE_P(
    TestConcat,
    test_concat_u8,
    ::testing::Values(
        test_concat_params{{{2, 64, 1, 1}, {2, 96, 1, 1}}, {2, 160, 1, 1}},
        test_concat_params{{{2, 64, 4, 4}, {2, 32, 4, 4}}, {2, 96, 4, 4}},
        test_concat_params{{{2, 256, 16, 16}, {2, 256, 16, 16}},
                           {2, 512, 16, 16}}));
}
