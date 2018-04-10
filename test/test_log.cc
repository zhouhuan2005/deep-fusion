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

#include <stdio.h>
#include "gtest/gtest.h"
#include "log.h"

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
