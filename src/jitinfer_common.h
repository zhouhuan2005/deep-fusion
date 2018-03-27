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

#include "jitinfer.h"
#include "util.h"

namespace jitinfer {

typedef float f32;
typedef int32_t s32;
typedef int8_t s8;
typedef uint8_t u8;

template <typename data_t>
struct data_traits {};
template <>
struct data_traits<f32> {
  static const auto dtype = memory::dtype::f32;
};
template <>
struct data_traits<u8> {
  static const auto dtype = memory::dtype::u8;
};
template <>
struct data_traits<s8> {
  static const auto dtype = memory::dtype::s8;
};
template <>
struct data_traits<s32> {
  static const auto dtype = memory::dtype::s32;
};

// TODO: change name of this dtype traits functions

template <memory::dtype>
struct prec_traits;
template <>
struct prec_traits<memory::dtype::f32> {
  typedef f32 type;
};
template <>
struct prec_traits<memory::dtype::s32> {
  typedef s32 type;
};
template <>
struct prec_traits<memory::dtype::s8> {
  typedef s8 type;
};
template <>
struct prec_traits<memory::dtype::u8> {
  typedef u8 type;
};

inline size_t dtype_size(memory::dtype dt) {
  switch (dt) {
#define CASE(tp) \
  case tp:       \
    return sizeof(typename prec_traits<tp>::type)
    CASE(memory::dtype::f32);
    CASE(memory::dtype::s32);
    CASE(memory::dtype::s8);
    CASE(memory::dtype::u8);
#undef CASE
    default:
      assert(!"bad data_type");
  }
}
}
