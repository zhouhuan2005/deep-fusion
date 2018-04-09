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

#include "jit_call_conf.h"
#include "jit_generator.h"

namespace jitinfer {

namespace jit {

struct jit_conv_kernel : public jit_generator {
  DECLARE_JIT_KERNEL(jit_conv_kernel);

  jit_conv_kernel(jit_conv_conf_t ajcp) : jcp_(ajcp) {
    generate();
    jit_ker_ = (void (*)(jit_conv_call_s *))getCode();
  }

  static bool init_conf(jit_conv_conf_t &jcp,
                        const std::unique_ptr<memory> &src,
                        const std::unique_ptr<memory> &wei,
                        const std::unique_ptr<memory> &bia,
                        std::array<int, 2> sz_stride,
                        std::array<int, 2> sz_padding,
                        std::unique_ptr<memory> &dst,
                        const std::unique_ptr<memory> &wei1x1,
                        const std::unique_ptr<memory> &bia1x1,
                        bool conv0_relu,
                        bool conv1_relu);

  jit_conv_conf_t jcp_;
  void (*jit_ker_)(jit_conv_call_s *);

private:
  using reg64_t = const Xbyak::Reg64;
  using reg32_t = const Xbyak::Reg32;
  using zmm_t = const Xbyak::Zmm;
  using ymm_t = const Xbyak::Ymm;
  using xmm_t = const Xbyak::Xmm;

  void generate();
};
}
}
