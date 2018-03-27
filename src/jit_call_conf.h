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
  int block;  // u8: 64, s32: 16
  bool with_relu;
};
}
}
