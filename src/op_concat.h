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

namespace jitinfer {

class op_concat : public op {
public:
  explicit op_concat(bool post_relu) : op() {
    kernel_ = new jit::jit_concat_kernel(/*jcp*/);
    // acc memory
  }

  ~op_concat() { delete kernel_; }

private:
  void infer() override;
  // pd_t conf_;
  jit::jit_concat_kernel *kernel_;

  // const data_t **src_;
  // const data_t **src_with_offset_;
  // int *ic_;
  // int *nb_ic_;
};
}
