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

#include "op_concat.h"
#include "util_deepfusion.h"

namespace deepfusion {

template <typename dtype>
void op_concat<dtype>::infer() {
  using namespace util;
  const auto &jcp = kernel_->jcp_;

  const int work_amount = jcp.bs * jcp.h * jcp.w;
  const int max = omp_get_max_threads();

  if (work_amount < max) {
#pragma omp parallel for schedule(static) collapse(1)
    for (int iwork = 0; iwork < max; ++iwork) {
      int n{0}, h{0}, w{0};
      nd_iterator_init(iwork, n, jcp.bs, h, jcp.h, w, jcp.w);
      int nhw = n * (jcp.h * jcp.w) + h * (jcp.w) + w;
      auto srcs = src_with_offset_ + iwork * jcp.n_inputs;
      for (int i = 0; i < jcp.n_inputs; ++i) {
        srcs[i] = srcs_data_[i] + (nhw * ic_[i]);
      }
      jit::jit_concat_call_s p = {0};
      p.src = reinterpret_cast<const void **>(srcs);
      p.nb_ic = reinterpret_cast<const int *>(nb_ic_);
      p.dst = reinterpret_cast<void *>(dst_data_ + nhw * jcp.oc);
      kernel_->jit_ker_(&p);
    }
  } else {
// if work amount > max omp threads, need balance
#pragma omp parallel
    {
      int ithr = omp_get_thread_num(), nthr = omp_get_num_threads();
      int start{0}, end{0};
      balance211(work_amount, nthr, ithr, start, end);
      int n{0}, h{0}, w{0};
      nd_iterator_init(start, n, jcp.bs, h, jcp.h, w, jcp.w);
      auto srcs = src_with_offset_ + ithr * jcp.n_inputs;
      jit::jit_concat_call_s p = {0};
      for (int iwork = start; iwork < end; ++iwork) {
        int nhw = n * (jcp.h * jcp.w) + h * (jcp.w) + w;
        for (int i = 0; i < jcp.n_inputs; ++i) {
          srcs[i] = srcs_data_[i] + (nhw * ic_[i]);
        }
        p.src = reinterpret_cast<const void **>(srcs);
        p.nb_ic = reinterpret_cast<const int *>(nb_ic_);
        p.dst = reinterpret_cast<void *>(dst_data_ + nhw * jcp.oc);
        // one kernel move one dst oc from all srcs
        kernel_->jit_ker_(&p);
        nd_iterator_step(n, jcp.bs, h, jcp.h, w, jcp.w);
      }
    }
  }
}

template class op_concat<f32>;
template class op_concat<s32>;
template class op_concat<s8>;
template class op_concat<u8>;
}
