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
#include "util_jitinfer.h"

namespace jitinfer {
namespace util {
size_t dtype_size(memory::dtype dt) {
  switch (dt) {
#define CASE(tp) \
  case tp:       \
    return sizeof(typename dtype2type<tp>::type)
    CASE(memory::dtype::f32);
    CASE(memory::dtype::s32);
    CASE(memory::dtype::s8);
    CASE(memory::dtype::u8);
#undef CASE
    default:
      assert(!"Unkown data type");
      return 0;
  }
}

int conv_output_size(int image, int kernel, int stride, int padding) {
  return (image + 2 * padding - kernel) / stride + 1;
}
int pool_output_size(int image, int kernel, int stride, int padding) {
  return (image + 2 * padding - kernel + stride - 1) / stride + 1;
}
}
}
