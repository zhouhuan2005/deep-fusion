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
#include "deepfusion_utils.h"

namespace deepfusion {
namespace utils {

void* aligned_malloc(size_t size, int alignment) {
  void *ptr;

#ifdef _WIN32
  ptr = _aligned_malloc(size, alignment);
  int rc = ptr ? 0 : -1;
#else
  int rc = ::posix_memalign(&ptr, alignment, size);
#endif

  return (rc == 0) ? ptr : 0;
}

void aligned_free(void *p) {
#ifdef _WIN32
  _aligned_free(p);
#else
  ::free(p);
#endif
}

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

}
}
