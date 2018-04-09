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
#include "jit_conv_kernel.h"
#include "log.h"
#include "omp_thread.h"

namespace jitinfer {

template <typename dst_data_t>
class op_conv : public op {
  typedef u8 src_data_t;
  typedef s8 wei_data_t;
  typedef s32 bia_data_t;
  typedef s32 acc_data_t;

public:
  explicit op_conv(const std::unique_ptr<memory> &src,
                   const std::unique_ptr<memory> &wei,
                   const std::unique_ptr<memory> &bia,
                   std::array<int, 2> sz_kernel,
                   std::array<int, 2> sz_stride,
                   std::array<int, 2> sz_padding,
                   std::unique_ptr<memory> &dst,
                   const std::unique_ptr<memory> &wei1x1 = nullptr,
                   const std::unique_ptr<memory> &bia1x1 = nullptr,
                   bool conv0_relu = false,
                   bool conv1_relu = false)
      : op(), fuse_conv1x1_(wei1x1 != nullptr) {
    jit::jit_conv_conf_t conf;
    if (!init_conf(conf,
                   src,
                   wei,
                   bia,
                   sz_kernel,
                   sz_stride,
                   sz_padding,
                   dst,
                   wei1x1,
                   bia1x1,
                   conv0_relu,
                   conv1_relu)) {
      error_and_exit("Init Conv op failed!");
    }

    kernel_ = new jit::jit_conv_kernel(conf);
    const auto &jcp = kernel_->jcp_;
    const int nthreads = omp_get_max_threads();
    ws_per_thread_ = jcp.oh * jcp.ow * jcp.oc_block * jcp.nb_oc_blocking;
    ws_ = (acc_data_t *)aligned_malloc(
        nthreads * ws_per_thread_ * sizeof(acc_data_t), 4096);  // 64??

    // acc format (h, oc/16, ow, 16o)
    ws1x1_per_thread_ = jcp.oh * jcp.ow * jcp.oc1x1;
    ws1x1_ = (acc_data_t *)aligned_malloc(
        nthreads * ws1x1_per_thread_ * sizeof(acc_data_t), 4096);  // 64??
  }

  ~op_conv() {
    free(ws_);
    free(ws1x1_);
    delete kernel_;
  }

protected:
  bool init_conf(jit::jit_conv_conf_t &conf,
                 const std::unique_ptr<memory> &src,
                 const std::unique_ptr<memory> &wei,
                 const std::unique_ptr<memory> &bia,
                 std::array<int, 2> sz_kernel,
                 std::array<int, 2> sz_stride,
                 std::array<int, 2> sz_padding,
                 std::unique_ptr<memory> &dst,
                 const std::unique_ptr<memory> &wei1x1,
                 const std::unique_ptr<memory> &bia1x1,
                 bool conv0_relu,
                 bool conv1_relu);
  void infer() override;
  const char *name() { return "conv"; }

private:
  bool fuse_conv1x1_;
  jit::jit_conv_kernel *kernel_;
  size_t ws_per_thread_;
  size_t ws1x1_per_thread_;
  acc_data_t *ws_;
  acc_data_t *ws1x1_;
};
}
