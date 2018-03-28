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
#include "gtest/gtest.h"
#include "jitinfer.h"
#include "test_common.h"

namespace jitinfer {

using memory = jitinfer::memory;
using format = jitinfer::memory::format;

struct test_concat_params {
  std::vector<memory::nchw_dims> srcs_dims;
  memory::nchw_dims dst_dims;
};

template <typename dtype>
class test_concat : public ::testing::TestWithParam<test_concat_params> {
  void check_data(const std::vector<std::unique_ptr<memory>> &srcs,
                  const std::unique_ptr<memory> &dst,
                  bool post_relu) {
    // TODO: add mkldnn concat
    ;
  }

protected:
  virtual void SetUp() {
    test_concat_params p =
        ::testing::TestWithParam<test_concat_params>::GetParam();
    auto dt = data_traits<dtype>::dtype;
    std::vector<std::unique_ptr<memory>> srcs(p.srcs_dims.size());
    std::unique_ptr<memory> dst;
    memory::format fmt = format::nhwc;
    for (size_t i = 0; i < p.srcs_dims.size(); ++i) {
      srcs[i].reset(new memory(p.srcs_dims[i], fmt, dt));
      util::fill_data<dtype>(static_cast<dtype *>(srcs[i]->data()),
                             srcs[i]->size());
    }
    dst.reset(new memory(p.dst_dims, fmt, dt));

    for (bool post_relu : {true, false}) {
      auto c = concat(srcs, dst, post_relu);
      c->submit();
      check_data(srcs, dst, post_relu);
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
        test_concat_params{{{2, 64, 1, 1}, {2, 96, 1, 1}}, {2, 160, 1, 1}}));

INSTANTIATE_TEST_CASE_P(
    TestConcat,
    test_concat_s32,
    ::testing::Values(
        test_concat_params{{{2, 64, 1, 1}, {2, 96, 1, 1}}, {2, 160, 1, 1}},
        test_concat_params{{{2, 64, 1, 1}, {2, 96, 1, 1}}, {2, 160, 1, 1}}));

INSTANTIATE_TEST_CASE_P(
    TestConcat,
    test_concat_s8,
    ::testing::Values(
        test_concat_params{{{2, 64, 1, 1}, {2, 96, 1, 1}}, {2, 160, 1, 1}},
        test_concat_params{{{2, 64, 1, 1}, {2, 96, 1, 1}}, {2, 160, 1, 1}}));

INSTANTIATE_TEST_CASE_P(
    TestConcat,
    test_concat_u8,
    ::testing::Values(
        test_concat_params{{{2, 64, 1, 1}, {2, 96, 1, 1}}, {2, 160, 1, 1}},
        test_concat_params{{{2, 64, 1, 1}, {2, 96, 1, 1}}, {2, 160, 1, 1}}));
}
