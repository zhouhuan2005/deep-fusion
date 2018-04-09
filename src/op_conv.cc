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
bool op_conv<dst_data_t>::init_conf(jit::jit_conv_conf_t &conf,
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
                                    bool conv1_relu) {
  // TODO: check size reasonable
  using namespace util;
  if (!all_true(
          src->data_type() == memory::dtype::u8,
          wei->data_type() == memory::dtype::s8,
          dst->data_type() == type2dtype<dst_data_t>::dtype,
          bia == nullptr || bia->data_type() == memory::dtype::s32,
          wei1x1 == nullptr || wei1x1->data_type() == memory::dtype::s8,
          bia1x1 == nullptr || bia1x1->data_type() == memory::dtype::s32)) {
    info("Data type do not match");
    return false;
  }

  return jit::jit_conv_kernel::init_conf(conf,
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
                                         conv1_relu);
}

template <typename dst_data_t>
void op_conv<dst_data_t>::infer() {
  using namespace util;
  const auto &jcp = kernel_->jcp_;
  if (fuse_conv1x1_) {
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
