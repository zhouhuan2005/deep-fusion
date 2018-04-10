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

namespace jit {
struct jit_concat_call_s {
  const void **src;
  const int *nb_ic;
  const void *dst;
};

struct jit_concat_conf_t {
  int bs;
  int h, w;
  int oc;
  int n_inputs;
  memory::dtype dt;
  int typesize;
  int block;      // u8: 64, s32: 16
  int bits_size;  // 128, 256, 512 : xmm, ymm, zmm
  bool with_relu;
};

struct jit_conv_call_s {
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
};
}
}
