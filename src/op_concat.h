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

#pragma once

#include <jitinfer.h>
#include "jit_concat_kernel.h"
#include "log.h"
#include "omp_thread.h"

namespace jitinfer {

template <typename dtype>
class op_concat : public op {
public:
  explicit op_concat(const std::vector<std::unique_ptr<memory>> &srcs,
                     std::unique_ptr<memory> &dst,
                     bool post_relu = false)
      : op() {
    jit::jit_concat_conf_t conf;
    if (!init_conf(conf, srcs, dst, post_relu)) {
      error_and_exit("Init Concat op failed!");
    }

    kernel_ = new jit::jit_concat_kernel(conf);

    const auto &jcp = kernel_->jcp_;
    const int num_srcs = jcp.n_inputs;
    assert(num_srcs == srcs.size());

    srcs_data_ = (const dtype **)malloc(num_srcs * sizeof(dtype *), 64);
    ic_ = (int *)malloc(num_srcs * sizeof(int), 64);
    nb_ic_ = (int *)malloc(num_srcs * sizeof(int), 64);

    for (int i = 0; i < num_srcs; ++i) {
      auto dim = srcs[i]->actual_dims();
      assert(srcs[i]->dim_format() == memory::format::nhwc);
      ic_[i] = dim[3];
      nb_ic_[i] = ic_[i] / jcp.block;
      check_eq(nb_ic_[i] * jcp.block, ic_[i]);
      // the src data is load here, if need update whe infer should change API
      srcs_data_[i] = reinterpret_cast<const dtype *>(srcs[i]->data());
    }
    dst_data_ = (dtype *)dst->data();

    const int nthreads = omp_get_max_threads();
    debug("Concat: Max OMP threads: %d", nthreads);
    src_with_offset_ =
        (const dtype **)malloc(nthreads * num_srcs * sizeof(dtype *), 64);
  }

  ~op_concat() {
    free(ic_);
    free(nb_ic_);
    free(srcs_data_);
    free(src_with_offset_);
    delete kernel_;
  }

protected:
  bool init_conf(jit::jit_concat_conf_t &conf,
                 const std::vector<std::unique_ptr<memory>> &srcs,
                 const std::unique_ptr<memory> &dst,
                 bool post_relu = false) {
    // TODO: can add more init of op_concat itself
    // before run into kernel init_conf
    return jit::jit_concat_kernel::init_conf(conf, srcs, dst, post_relu);
  }
  void infer() override;
  const char *name() { return "concat"; }

private:
  jit::jit_concat_kernel *kernel_;
  dtype *dst_data_;
  const dtype **srcs_data_;
  const dtype **src_with_offset_;
  int *ic_;
  int *nb_ic_;
};
}
