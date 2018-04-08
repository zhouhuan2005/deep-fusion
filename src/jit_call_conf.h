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
  int l_pad, t_pad;
  int r_pad, b_pad;
  int kh, kw;
  int sh, sw;
  int oc_block;
  int nb_oc_blocking;
  /* conv 1x1*/
  int oc1x1;
  int oc1x1_block;
  int nb_oc1x1;
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
