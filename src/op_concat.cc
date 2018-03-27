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

#include "op_concat.h"
#include "jitinfer_common.h"
#include "util.h"

namespace jitinfer {

template <typename dtype>
void op_concat<dtype>::infer() {
  using namespace util;
  const auto &jcp = kernel_->jcp;

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
      kernel_->jit_ker(&p);
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
        kernel_->jit_ker(&p);
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
