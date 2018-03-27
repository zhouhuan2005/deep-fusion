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
#include "log.h"

namespace jitinfer {

namespace jit {

struct jit_concat_kernel : public jit_generator {
  DECLARE_JIT_KERNEL(jit_concat_kernel);

  jit_concat_kernel(const std::vector<std::unique_ptr<memory>>& srcs,
                    const std::unique_ptr<memory>& dst,
                    bool post_relu) {
    if (!init_conf(srcs, dst, post_relu)) {
      error("Init jit_concat kernel failed!");
    }
    generate();
    jit_ker = (void (*)(jit_concat_call_s*))getCode();
  }

  jit_concat_conf_t jcp;
  void (*jit_ker)(jit_concat_call_s*);

private:
  using reg64_t = const Xbyak::Reg64;
  using reg32_t = const Xbyak::Reg32;
  using zmm_t = const Xbyak::Zmm;
  using xmm_t = const Xbyak::Xmm;

  reg64_t param = abi_param1;
  reg64_t reg_ptr_src = r8;
  reg64_t reg_ptr_nb_ic = r9;
  reg64_t reg_ptr_dst = r10;
  reg64_t reg_ptr_src_i = r11;
  reg64_t reg_ninputs = r12;
  reg32_t reg_nb = r15d;

  xmm_t xmm_src = xmm_t(30);
  zmm_t zmm_src = zmm_t(30);
  zmm_t zmm_zero = zmm_t(31);
  xmm_t xmm_zero = xmm_t(31);

  void compute_one_input();
  void generate();
  bool init_conf(const std::vector<std::unique_ptr<memory>>& srcs,
                 const std::unique_ptr<memory>& dst,
                 bool post_relu);
};
}
}
