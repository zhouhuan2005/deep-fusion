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

int _getenv(char *value, const char *name, int length) {
  int result = 0;
  int last_idx = 0;
  if (length > 1) {
    int value_length = 0;
#ifdef _WIN32
    value_length = GetEnvironmentVariable(name, value, length);
    if (value_length >= length) {
      result = -value_length;
    } else {
      last_idx = value_length;
      result = value_length;
    }
#else
    char *buffer = getenv(name);
    if (buffer != NULL) {
      value_length = strlen(buffer);
      if (value_length >= length) {
        result = -value_length;
      } else {
        strncpy(value, buffer, value_length);
        last_idx = value_length;
        result = value_length;
      }
    }
#endif
  }
  value[last_idx] = '\0';
  return result;
}

// If need profiling
// 1. cmake -DWITH_PROFILE=ON
// 2. export DEEPFUSION_PROFILE=1
bool is_profiling() {
  static bool profiling = false;
  static bool initialized = false;
  if (!initialized) {
    const int len = 2;
    char env_dump[len] = {0};
    profiling =_getenv(env_dump, "DEEPFUSION_PROFILE", len) == 1 && atoi(env_dump) == 1;
    initialized = true;
  }
  return profiling;
}

// If need dump jit code
// 1. cmake -DCMAKE_BUILD_TYPE=DEBUG
// 2. export DEEPFUSION_DUMP_CODE=1
bool jit_dump_code() {
  static bool initialized = false;
  static bool dump_jit_code = false;
  if (!initialized) {
    const int len = 2;
    char env_dump[len] = {0};
    dump_jit_code = _getenv(env_dump, "DEEPFUSION_DUMP_CODE", len) == 1 &&
                    atoi(env_dump) == 1;
    initialized = true;
  }
  return dump_jit_code;
}

}
}
