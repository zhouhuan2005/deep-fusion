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
#include "op_conv.h"
#include "util_jitinfer.h"

namespace jitinfer {

template <typename dst_data_t>
void op_conv<dst_data_t>::infer() {
  using namespace util;
  const auto &jcp = kernel_->jcp_;
  if (post_conv1x1_) {
    ;
  } else {
    ;
  }
}

template class op_conv<f32>;
template class op_conv<s32>;
template class op_conv<s8>;
template class op_conv<u8>;
}
