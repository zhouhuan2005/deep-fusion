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
#include "jit_conv_kernel.h"
#include "util_jitinfer.h"

#define GET_OFF(field) offsetof(jit_conv_call_s, field)

namespace jitinfer {
namespace jit {

using namespace Xbyak;

void jit_conv_kernel::generate() {
  preamble();

  postamble();
}
bool jit_conv_kernel::init_conf(jit_conv_conf_t &jcp,
                                const std::unique_ptr<memory> &src,
                                const std::unique_ptr<memory> &wei,
                                const std::unique_ptr<memory> &bia,
                                std::array<int, 2> sz_kernel,
                                std::array<int, 2> sz_stride,
                                std::array<int, 2> sz_padding,
                                std::unique_ptr<memory> &dst,
                                const std::unique_ptr<memory> &wei1x1,
                                const std::unique_ptr<memory> &bia1x1,
                                bool relu_conv0,
                                bool relu_conv1) {
  jcp = jitinfer::util::zero<decltype(jcp)>();

  return true;
}
}
}
