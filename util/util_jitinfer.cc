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
