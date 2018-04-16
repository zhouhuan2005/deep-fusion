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
#include "jit_concat_kernel.h"
#include "log.h"
#include "omp_thread.h"

namespace deepfusion {

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

    srcs_data_ = (const dtype **)utils::aligned_malloc(num_srcs * sizeof(dtype *), 64);
    ic_ = (int *)utils::aligned_malloc(num_srcs * sizeof(int), 64);
    nb_ic_ = (int *)utils::aligned_malloc(num_srcs * sizeof(int), 64);

    for (int i = 0; i < num_srcs; ++i) {
      auto dim = srcs[i]->actual_dims();
      assert(srcs[i]->dim_format() == memory::format::nhwc);
      ic_[i] = dim[3];
      nb_ic_[i] = ic_[i] / jcp.block;
      check_eq(nb_ic_[i] * jcp.block, ic_[i]);
      // XXX the src data pointer is determined here, if need update when infer should change API
      srcs_data_[i] = reinterpret_cast<const dtype *>(srcs[i]->data());
    }
    dst_data_ = (dtype *)dst->data();

    const int nthreads = omp_get_max_threads();
    debug("Concat: Max OMP threads: %d", nthreads);
    src_with_offset_ = (const dtype **)utils::aligned_malloc(nthreads * num_srcs * sizeof(dtype *), 4096);
  }

  ~op_concat() {
    utils::aligned_free(ic_);
    utils::aligned_free(nb_ic_);
    utils::aligned_free(srcs_data_);
    utils::aligned_free(src_with_offset_);
    delete kernel_;
  }

protected:
  bool init_conf(jit::jit_concat_conf_t &conf,
                 const std::vector<std::unique_ptr<memory>> &srcs,
                 const std::unique_ptr<memory> &dst,
                 bool post_relu = false) {
    /* TODO can add more init of op_concat itself
            before run into kernel init_conf
    */
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
