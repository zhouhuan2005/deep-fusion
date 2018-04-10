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

#pragma once

#include "deepfusion.h"
#include "util.h"

namespace deepfusion {
namespace util {

// type(float, int8...) to deepfusion::memory::dtype
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

// deepfusion::memory::dtype to type(float, int8...)
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

int conv_output_size(int image, int kernel, int stride, int padding);
int pool_output_size(int image, int kernel, int stride, int padding);
}
}
