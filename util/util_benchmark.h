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

#include "jitinfer.h"
#include "util.h"

namespace jitinfer {
namespace util {
void clear_cache();

#ifdef WITH_COLD_CACHE
struct dummy_memory {
public:
  void clear_cache();
  explicit dummy_memory(size_t n);
  ~dummy_memory();

private:
  unsigned char* p_;
  size_t size_;
  DISABLE_COPY_AND_ASSIGN(dummy_memory);
};
#endif

const char* dtype2str(memory::dtype dt);
memory::dtype str2dtype(const std::string& str);
memory::dtype str2dtype(const char* str);
std::vector<std::string> split(const std::string& s, char delimiter = ',');
}
}
