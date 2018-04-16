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

#include <cmath>
#include "gtest/gtest.h"
#include "deepfusion.h"
#include "omp_thread.h"
#include "deepfusion_utils.h"
#include "mkldnn.hpp"

namespace deepfusion {
namespace testutils {

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

template <typename data_t>
static inline data_t set_value(size_t index) {
  using data_type = deepfusion::memory::dtype;

  if (utils::type2dtype<data_t>::dtype == data_type::f32) {
    double mean = 1., deviation = 1e-2;
    return static_cast<data_t>(mean + deviation * sinf(float(index % 37)));
  } else if (utils::one_of(utils::type2dtype<data_t>::dtype, data_type::s8, data_type::s32)) {
    return data_t(rand() % 21 - 10);
  } else if (utils::type2dtype<data_t>::dtype == data_type::u8) {
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

// mkld-dnn related
std::unique_ptr<mkldnn::eltwise_forward::primitive_desc> get_mkldnn_relu_pd(
    const mkldnn::memory::desc md, const mkldnn::engine& eng);

// exchange btw mkldnn and deepfusion
mkldnn::memory::dims to_mkldnn_dims(const memory::nchw_dims& nchwdims);
memory::dtype from_mkldnn_dtype(mkldnn::memory::data_type dt);
mkldnn::memory::data_type to_mkldnn_dtype(memory::dtype dt);

}
}
