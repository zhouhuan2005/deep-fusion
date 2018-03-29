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
 * This file is used on jitinfer library
 */
#pragma once

#include "jitinfer.h"
#include "util.h"

namespace jitinfer {
namespace util {

// type(float, int8...) to jitinfer::memory::dtype
template <typename T>
struct type2dtype {};
template <>
struct type2dtype<f32> {
  static const auto dtype = memory::dtype::f32;
};
template <>
struct type2dtype<u8> {
  static const auto dtype = memory::dtype::u8;
};
template <>
struct type2dtype<s8> {
  static const auto dtype = memory::dtype::s8;
};
template <>
struct type2dtype<s32> {
  static const auto dtype = memory::dtype::s32;
};

// jitinfer::memory::dtype to type(float, int8...)
template <memory::dtype>
struct dtype2type {};
template <>
struct dtype2type<memory::dtype::f32> {
  typedef f32 type;
};
template <>
struct dtype2type<memory::dtype::s32> {
  typedef s32 type;
};
template <>
struct dtype2type<memory::dtype::s8> {
  typedef s8 type;
};
template <>
struct dtype2type<memory::dtype::u8> {
  typedef u8 type;
};

size_t dtype_size(memory::dtype dt);
}
}
