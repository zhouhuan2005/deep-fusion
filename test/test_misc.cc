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

#include "gtest/gtest.h"
#include "test_utils.h"

namespace deepfusion {

TEST(TestMisc, test_misc) {
  using namespace utils;

  EXPECT_EQ(dividable_of(12, 5, 4, 3, 2), 4);
  EXPECT_EQ(dividable_of(12, 3, 4, 5, 2), 3);
  EXPECT_EQ(dividable_of(5, 3, 2), 1);

  EXPECT_EQ(find_dividable(14, 8), 7);
  EXPECT_EQ(find_dividable(12, 5), 4);
  EXPECT_EQ(find_dividable(12, 3), 3);
  EXPECT_EQ(find_dividable(5, 4), 1);
  EXPECT_EQ(find_dividable(5, 5), 5);
  EXPECT_EQ(find_dividable(5, 8), 5);

  EXPECT_TRUE(all_true(true));
  EXPECT_TRUE(all_true(1, 1, 1, true));
  EXPECT_TRUE(all_true(true, 1, 1, 1, true));

  EXPECT_FALSE(all_true(false));
  EXPECT_FALSE(all_true(1, 0, true));
  EXPECT_FALSE(all_true(1, 1, 0, false));
  EXPECT_FALSE(all_true(false, 1, 0, true));
}

}
