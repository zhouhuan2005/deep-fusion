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

#pragma once

#include <assert.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

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

#define error(fmt, ...)                     \
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
  if (!(x)) error("Check Failed!");

#define check_compare(val0, val1, cmp) \
  if (!((val0)cmp(val1))) error("Check " #val0 " " #cmp " " #val1 " Failed!");

#define check_eq(val0, val1, ...) check_compare(val0, val1, ==)

#define check_gt(val0, val1, ...) check_compare(val0, val1, >)

#define check_ge(val0, val1, ...) check_compare(val0, val1, >=)

#define check_lt(val0, val1, ...) check_compare(val0, val1, <)

#define check_le(val0, val1, ...) check_compare(val0, val1, <=)
}
