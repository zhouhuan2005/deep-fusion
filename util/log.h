/*******************************************************************************
 * Copyright 2018 Tensor Tang. All Rights Reserved
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
#pragma once
#include <stdio.h>

namespace jitinfer {

#define __FILENAME__                                                       \
  (__builtin_strrchr(__FILE__, '/') ? __builtin_strrchr(__FILE__, '/') + 1 \
                                    : __FILE__)

#define log(ter, type, fmt, ...)               \
  fprintf(ter,                                 \
          "[" #type " %s %s:%d] >> " fmt "\n", \
          __TIME__,                            \
          __FILENAME__,                        \
          __LINE__,                            \
          ##__VA_ARGS__)  //; fflush(stdout)

#define info(fmt, ...) log(stdout, INFO, fmt, ##__VA_ARGS__)

#define warning(fmt, ...) log(stdout, WARNING, fmt, ##__VA_ARGS__)

#define error_and_exit(fmt, ...)            \
  {                                         \
    log(stderr, ERROR, fmt, ##__VA_ARGS__); \
    exit(EXIT_FAILURE);                     \
  }

#ifdef NDEBUG
#define debug(fmt, ...)
#else
#define debug(fmt, ...) log(stdout, DEBUG, fmt, ##__VA_ARGS__)
#endif

#define check(x) \
  if (!(x)) error_and_exit("Check Failed!");

#define check_compare(val0, val1, cmp) \
  if (!((val0)cmp(val1)))              \
    error_and_exit("Check " #val0 " " #cmp " " #val1 " Failed!");

#define check_eq(val0, val1) check_compare(val0, val1, ==)

#define check_gt(val0, val1) check_compare(val0, val1, >)

#define check_ge(val0, val1) check_compare(val0, val1, >=)

#define check_lt(val0, val1) check_compare(val0, val1, <)

#define check_le(val0, val1) check_compare(val0, val1, <=)
}
