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

#include <stdint.h>
#include "deepfusion.h"

namespace deepfusion {

enum conv_loop_order_t { loop_cgn, loop_gnc, loop_ngc };
/** Kinds of algorithms. */
typedef enum {
    deepfusion_alg_kind_undef,
    /** Direct convolution */
    deepfusion_convolution_direct = 1,
    /** Winograd convolution */
    deepfusion_convolution_winograd = 2,
    /** Eltwise: ReLU */
    deepfusion_eltwise_relu = 8,
    /** Eltwise: hyperbolic tangent non-linearity (tanh) */
    deepfusion_eltwise_tanh = 9,
    /** Eltwise: parametric exponential linear unit (elu) */
    deepfusion_eltwise_elu = 10,
    /** Eltwise: square */
    deepfusion_eltwise_square = 11,
    /** Eltwise: abs */
    deepfusion_eltwise_abs = 12,
    /** Eltwise: square root */
    deepfusion_eltwise_sqrt = 13,
    /** Eltwise: linear */
    deepfusion_eltwise_linear = 14,
    /** Eltwise: bounded_relu */
    deepfusion_eltwise_bounded_relu = 15,
    /** Eltwise: soft_relu */
    deepfusion_eltwise_soft_relu = 16,
    /** Eltwise: logistic */
    deepfusion_eltwise_logistic = 17,
    /** Max pooling */
    deepfusion_pooling_max = 34,
    /** Average pooling include padding */
    deepfusion_pooling_avg_include_padding = 40,
    /** Average pooling exclude padding */
    deepfusion_pooling_avg_exclude_padding = 41,
    deepfusion_pooling_avg = deepfusion_pooling_avg_exclude_padding,
    /** Local response normalization (LRN) across multiple channels */
    deepfusion_lrn_across_channels = 65,
    /** LRN within a single channel */
    deepfusion_lrn_within_channel = 66,
    /** Direct deconvolution */
    deepfusion_deconvolution_direct = 71,
    /** Winograd deconvolution */
    deepfusion_deconvolution_winograd = 72,
    /** RNN cell */
    deepfusion_vanilla_rnn = 80,
    /** LSTM cell */
    deepfusion_vanilla_lstm = 81,
    /** GRU cell */
    deepfusion_vanilla_gru = 82,
} deepfusion_alg_kind_t;

using alg_kind_t = deepfusion_alg_kind_t;
namespace alg_kind {
    const alg_kind_t undef = deepfusion_alg_kind_undef;
    const alg_kind_t convolution_direct = deepfusion_convolution_direct;
    const alg_kind_t convolution_winograd = deepfusion_convolution_winograd;
    const alg_kind_t deconvolution_direct = deepfusion_deconvolution_direct;
    const alg_kind_t deconvolution_winograd = deepfusion_deconvolution_winograd;
    const alg_kind_t eltwise_relu = deepfusion_eltwise_relu;
    const alg_kind_t eltwise_tanh = deepfusion_eltwise_tanh;
    const alg_kind_t eltwise_elu = deepfusion_eltwise_elu;
    const alg_kind_t eltwise_square = deepfusion_eltwise_square;
    const alg_kind_t eltwise_abs = deepfusion_eltwise_abs;
    const alg_kind_t eltwise_sqrt = deepfusion_eltwise_sqrt;
    const alg_kind_t eltwise_linear = deepfusion_eltwise_linear;
    const alg_kind_t eltwise_bounded_relu = deepfusion_eltwise_bounded_relu;
    const alg_kind_t eltwise_soft_relu = deepfusion_eltwise_soft_relu;
    const alg_kind_t eltwise_logistic = deepfusion_eltwise_logistic;
    const alg_kind_t pooling_max = deepfusion_pooling_max;
    const alg_kind_t pooling_avg = deepfusion_pooling_avg;
    const alg_kind_t pooling_avg_include_padding = deepfusion_pooling_avg_include_padding;
    const alg_kind_t pooling_avg_exclude_padding = deepfusion_pooling_avg_exclude_padding;
    const alg_kind_t lrn_across_channels = deepfusion_lrn_across_channels;
    const alg_kind_t lrn_within_channel = deepfusion_lrn_within_channel;
    const alg_kind_t vanilla_rnn = deepfusion_vanilla_rnn;
    const alg_kind_t vanilla_lstm = deepfusion_vanilla_lstm;
    const alg_kind_t vanilla_gru = deepfusion_vanilla_gru;
}
namespace jit {

// concat with optional relu fusion
struct jit_concat_call_t {
  const void **src;
  const int  *nb_ic;
  const void *dst;
};

struct jit_concat_conf_t {
  int           bs;
  int           h, w;
  int           oc;
  int           n_inputs;
  memory::dtype dt;
  int           typesize;
  int           block;      // u8: 64, s32: 16
  int           bits_size;  // 128, 256, 512 : xmm, ymm, zmm
  bool          with_relu;
};

// convreluconv1x1relu
struct jit_conv_call_t {
  const void *src;
  const void *dst; /* hack, non-const for forward */
  const void *wei;
  const void *bia;
  const void *scales;
  const void *acc_s32;

  const void *wei1x1;
  const void *bia1x1;
  const void *acc1x1;
  // const void *out1x1; == dst
  const void *scales1x1;
  size_t ocb3x3;

  size_t kh_padding;
  size_t channel;
};

struct jit_conv_conf_t {
  int bs;
  int gp, ic, oc;
  int ih, iw, oh, ow;
  int kh, kw;
  int sh, sw;
  int l_pad, t_pad;  // left, top padding
  int ic_block, oc_block;
  int nb_ic, nb_oc;
  // @note: nc_ic==(nb_ic_blocking * ic_chunk)
  int nb_ic_blocking, nb_oc_blocking;
  int ur_w, ur_w_tail;
  int typesize_in;
  int typesize_out;
  int typesize_acc;
  int typesize_conv0_bia;
  int typesize_conv1_bia;
  memory::dtype dst_dt, conv0_bias_dt, conv1_bias_dt;
  round_mode conv0_round_mode, conv1_round_mode;
  conv_loop_order_t loop_order;
  /* conv 1x1*/
  int oc1x1;
  int oc1x1_block;
  int nb_oc1x1;
  bool use_vnni;
  bool fuse_conv1x1;
  bool conv0_with_relu;
  bool conv1_with_relu;
  bool conv0_with_bias;
  bool conv1_with_bias;
  bool conv0_multi_oc_scale;  // whether use multi channel to scale oc
  bool conv1_multi_oc_scale;
  /*pool*/
  int pool_kw;
  alg_kind_t pool_alg;
};

/*pool*/
struct jit_pool_conf_t {
  int mb, c;
  int ih, iw, oh, ow;
  int stride_h, stride_w;
  int kh, kw;
  int t_pad, l_pad;
  alg_kind_t alg;
  bool is_training;
  bool pad_w_is_null;
  bool is_backward;
  memory::dtype ind_dt;

  int c_block, c_tail, nb_c;
  int ur_c, ur_c_tail;
  int ur_w;
  int ur_w_tail;
  size_t tail[4];
  memory::dtype src_dt;
  memory::dtype dst_dt;
};

struct jit_pool_call_t {
  const char *src_i8;
  const char *dst_i8;
  size_t kw_range;
  size_t kh_range;
  float idivider;
  size_t move_bits;
};

}
}
