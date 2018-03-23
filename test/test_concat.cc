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
#include "src/util.h"

namespace jitinfer {

struct test_concat_params {
  // TODO: dims
  int tmp;
};

template <typename dtype>
class test_concat : public ::testing::TestWithParam<test_concat_params> {
protected:
  virtual void SetUp() {
    test_concat_params p =
        ::testing::TestWithParam<test_concat_params>::GetParam();
    ASSERT_TRUE(p.tmp == 1);
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

INSTANTIATE_TEST_CASE_P(TestConcat,
                        test_concat_f32,
                        ::testing::Values(test_concat_params{1}));

INSTANTIATE_TEST_CASE_P(TestConcat,
                        test_concat_s32,
                        ::testing::Values(test_concat_params{1}));

INSTANTIATE_TEST_CASE_P(TestConcat,
                        test_concat_s8,
                        ::testing::Values(test_concat_params{1}));

INSTANTIATE_TEST_CASE_P(TestConcat,
                        test_concat_u8,
                        ::testing::Values(test_concat_params{1}));
}
