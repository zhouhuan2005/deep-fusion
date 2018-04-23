/*******************************************************************************
* Copyright 2016-2018 Intel Corporation
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

#ifndef OP_CONV_POOL_H
#define OP_CONV_POOL_H

#include "deepfusion.h"
#include "log.h"
#include "omp_thread.h"
#include "jit_call_conf.h"
#include "jit_pool_kernel.h"
#include "jit_conv_kernel_4_conv_pool.h"

namespace deepfusion {

template <typename dst_data_t>
class op_conv_pool_t : public op {
    typedef u8 src_data_t;
    typedef s8 wei_data_t;
    // typedef s32 bia_data_t;
    typedef s32 acc_data_t;
    typedef s32 tmp_data_t;
public:
    explicit op_conv_pool_t
                  (const std::unique_ptr<memory> &conv_src,
                   const std::unique_ptr<memory> &conv_wei,
                   const std::unique_ptr<memory> &conv_bia,
                   std::unique_ptr<memory> &conv_dst,
                   std::array<int, 2> conv_stride,
                   std::array<int, 2> conv_padding,
                   std::vector<float> conv_scales,
                   const std::unique_ptr<memory>  &pool_src,
                   std::unique_ptr<memory>  &pool_dst,
                   std::array<int, 2> pool_stride,
                   std::array<int, 2> pool_padding,
                   std::array<int, 2> pool_kernel,
                   alg_kind_t pool_alg,
                   round_mode conv_round_mode = round_mode::nearest,
                   round_mode pool_round_mode = round_mode::nearest
                   )
    : op() {
    if (!init_conf(conv_conf_,
                   pool_conf_,
                   conv_src,
                   conv_wei,
                   conv_bia,
                   conv_dst,
                   1,
                   conv_stride,
                   conv_padding,
                   conv_scales,
                   pool_src,
                   pool_dst,
                   pool_stride,
                   pool_padding,
                   pool_kernel,
                   pool_alg)) {
          error_and_exit("Init Conv relu pool op failed!");
    }

    conv_kernel_ = new jit::jit_conv_kernel_4_conv_pool(conv_conf_);
    pool_kernel_ = new jit::jit_pool_kernel(pool_conf_);
    const auto &jcp = conv_kernel_->jcp;
    const int nthreads = omp_get_max_threads();
    ws_per_thread_ = jcp.oh * jcp.ow * jcp.oc_block * jcp.nb_oc_blocking;
    ws_ = (acc_data_t *)utils::aligned_malloc(
        nthreads * ws_per_thread_ * sizeof(acc_data_t), 4096);  // page align here
    // acc format (h, oc/16, ow, 16o)

    // TODO enable update data handle from outside
    src_data_ = reinterpret_cast<const src_data_t *>(conv_src->data());
    wei_data_ = reinterpret_cast<const wei_data_t *>(conv_wei->data());
    bia_data_ =
        conv_bia != nullptr ? reinterpret_cast<const void *>(conv_bia->data()) : NULL;
    // TODO: do not save pointor, save buffer instead
    conv_scales_data_ = reinterpret_cast<const float *>(conv_scales.data());
    tmp_data_ = reinterpret_cast<tmp_data_t *>(conv_dst->data());
    pool_src_data_ = reinterpret_cast<tmp_data_t *>(pool_src->data());
    dst_data_ = reinterpret_cast<dst_data_t *>(pool_dst->data());
  }

  ~op_conv_pool_t() {
    utils::aligned_free(ws_);
    delete conv_kernel_;
    delete pool_kernel_;
  }

protected:
  bool init_conf(jit::jit_conv_conf_t &conv_conf,
                 jit::jit_pool_conf_t &pool_conf,
                 const std::unique_ptr<memory> &conv_src,
                 const std::unique_ptr<memory> &conv_wei,
                 const std::unique_ptr<memory> &conv_bia,
                 std::unique_ptr<memory> &conv_dst,
                 int ngroups,  // only enabled on conv0
                 std::array<int, 2> conv_stride,
                 std::array<int, 2> conv_padding,
                 std::vector<float> conv_scales,
                 const std::unique_ptr<memory>  &pool_src,
                 std::unique_ptr<memory>  &pool_dst,
                 std::array<int, 2> pool_stride,
                 std::array<int, 2> pool_padding,
                 std::array<int, 2> pool_kernel,
                 alg_kind_t pool_alg);
  
  void infer() override;
  inline void infer_conv();
  inline void infer_pool();

  const char *name() { return "conv_relu_pool"; }

private:
  const src_data_t *src_data_;
  const wei_data_t *wei_data_;
  const void *bia_data_;
  const float *conv_scales_data_;
  size_t ws_per_thread_;
  dst_data_t *dst_data_;
  tmp_data_t *tmp_data_, *pool_src_data_;
  acc_data_t *ws_;
  jit::jit_conv_kernel_4_conv_pool *conv_kernel_; 
  jit::jit_conv_conf_t conv_conf_;
  jit::jit_pool_kernel *pool_kernel_;
  jit::jit_pool_conf_t pool_conf_;
};
}
#endif

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
