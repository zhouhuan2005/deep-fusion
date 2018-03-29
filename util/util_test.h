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
/**
 * This file defines some utilities that used in gtest
 */

#pragma once
#include <cmath>
#include "gtest/gtest.h"
#include "jitinfer.h"
#include "omp_thread.h"
#include "util.h"
#include "util_jitinfer.h"

namespace jitinfer {
namespace util {

template <typename data_t>
static inline data_t set_value(size_t index) {
  using data_type = jitinfer::memory::dtype;

  if (type2dtype<data_t>::dtype == data_type::f32) {
    double mean = 1., deviation = 1e-2;
    return static_cast<data_t>(mean + deviation * sinf(float(index % 37)));
  } else if (one_of(type2dtype<data_t>::dtype, data_type::s8, data_type::s32)) {
    return data_t(rand() % 21 - 10);
  } else if (type2dtype<data_t>::dtype == data_type::u8) {
    return data_t(rand() % 17);
  } else {
    return data_t(0);
  }
}

template <typename T>
void fill_data(T* p, size_t sz) {
#pragma omp parallel for schedule(static)
  for (size_t i = 0; i < sz; i++) {
    p[i] = set_value<T>(i);
  }
}

template <typename T>
void compare_array(T* dst, T* ref, size_t sz) {
#pragma omp parallel for schedule(static)
  for (size_t i = 0; i < sz; ++i) {
    if (std::is_same<T, f32>::value) {
      f32 diff = dst[i] - ref[i];
      f32 e = (std::fabs(ref[i]) > (f32)1e-4) ? diff / ref[i] : diff;
      EXPECT_NEAR(e, 0.f, (f32)1e-4) << "Index: " << i << " Total: " << sz;
    } else {
      EXPECT_EQ(dst[i], ref[i]) << "Index: " << i << " Total: " << sz;
    }
  }
}
}
}
