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
#include "src/log.h"

#include <stdio.h>

namespace jitinfer {

TEST(TestLog, log) {
  log(stdout, test, "hello test with param %d", 1);
  log(stdout, test, "hello test");
  info("hello test with param %d", 1);
  info("hello test with param");

  debug("hello test with param %d", 1);
  debug("hello test with param");

  warning("hello test with param %d", 1);
  warning("hello test with param");

  check(true);
  check(1 == 1);
  check(true && (1 == 1));
  check_eq(1, 1);

  check_lt(1, 2);
  check_le(1, 1);
  check_le(1, 2);

  check_gt(3, 2);
  check_ge(3, 1);
  check_ge(3, 3);
  /*
    // add string
    check(1 == 1, "test check with param %d", 1);
    check(1 == 1, "test check");

    check_eq(1, 1, "test check_eq with param %d", 1);
    check_eq(1, 1, "test check_eq");

    check_lt(1, 2, "test check_lt with param %d", 1);
    check_le(1, 1, "test check_le");

    check_gt(2, 1, "test check_gt with param %d", 1);
    check_ge(1, 1, "test check_ge");
  */
}
}
