/*******************************************************************************
 * Copyright 2018 Tensor Tang. All Rights Reserved
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

#include "jit_call_conf.h"
#include "jit_generator.h"

namespace jitinfer {

namespace jit {

struct jit_concat_kernel : public jit_generator {
  DECLARE_JIT_KERNEL(jit_concat_kernel);

  jit_concat_kernel(jit_concat_conf_t ajcp) : jcp_(ajcp) {
    generate();
    jit_ker_ = (void (*)(jit_concat_call_s*))getCode();
  }

  static bool init_conf(jit_concat_conf_t& jcp,
                        const std::vector<std::unique_ptr<memory>>& srcs,
                        const std::unique_ptr<memory>& dst,
                        bool post_relu);

  jit_concat_conf_t jcp_;
  void (*jit_ker_)(jit_concat_call_s*);

private:
  enum {
    USE_ZMM = 512,
    USE_YMM = 256,
    USE_XMM = 128,
  };
  using reg64_t = const Xbyak::Reg64;
  using reg32_t = const Xbyak::Reg32;
  using zmm_t = const Xbyak::Zmm;
  using ymm_t = const Xbyak::Ymm;
  using xmm_t = const Xbyak::Xmm;

  reg64_t param = abi_param1;
  reg64_t reg_ptr_src = r8;
  reg64_t reg_ptr_nb_ic = r9;
  reg64_t reg_ptr_dst = r10;
  reg64_t reg_ptr_src_i = r11;
  reg64_t reg_ninputs = r12;
  reg32_t reg_nb = r15d;

  xmm_t xmm_src = xmm_t(30);
  ymm_t ymm_src = ymm_t(30);
  zmm_t zmm_src = zmm_t(30);
  xmm_t xmm_zero = xmm_t(31);
  ymm_t ymm_zero = ymm_t(31);
  zmm_t zmm_zero = zmm_t(31);

  void compute_one_input();
  void generate();
};
}
}
