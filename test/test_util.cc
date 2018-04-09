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
#include "util.h"

namespace jitinfer {

TEST(TestUtil, test_util) {
  using namespace util;

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
