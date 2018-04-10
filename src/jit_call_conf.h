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

#include <stdint.h>
#include "jitinfer.h"

namespace jitinfer {
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

struct jit_conv_call_s {
  const void *src;
  const void *dst; /* hack, non-const for forward */
  const void *filt;
  const void *bias;
  const void *scales;
  const void *acc_s32;

  const void *wei1x1;
  const void *bia1x1;
  const void *acc1x1;
  const void *out1x1;
  const void *scales1x1;
  size_t ocb3x3;

  size_t kh_padding;
  size_t kw_padding;
  size_t channel;
  size_t oc_blocks;
};
}
}
