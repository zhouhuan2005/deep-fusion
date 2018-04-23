/*******************************************************************************
* Copyright 2017-2018 Intel Corporation
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

namespace deepfusion {
namespace jit {

struct jit_conv_kernel_4_conv_pool : public jit_generator {
  DECLARE_JIT_KERNEL(jit_conv_kernel);

  jit_conv_kernel_4_conv_pool(jit_conv_conf_t ajcp) : jcp(ajcp) {
    generate();
    jit_ker_ = (void (*)(jit_conv_call_t *))getCode();
  }

  static bool init_conf(jit_conv_conf_t &jcp,
                        const std::unique_ptr<memory> &src,
                        const std::unique_ptr<memory> &wei,
                        const std::unique_ptr<memory> &bia,
                        int ngroups,  // only enabled on conv0
                        std::array<int, 2> sz_stride,
                        std::array<int, 2> sz_padding,
                        std::unique_ptr<memory> &dst,
                        std::vector<float> conv_scales,
                        bool conv_relu);

  jit_conv_conf_t jcp;
  void (*jit_ker_)(jit_conv_call_t *);

private:
  enum {
    ker_reg_base_idx = 28,
  };

  using reg64_t = const Xbyak::Reg64;
  using reg32_t = const Xbyak::Reg32;
  using zmm_t = const Xbyak::Zmm;
  using ymm_t = const Xbyak::Ymm;
  using xmm_t = const Xbyak::Xmm;
  using op_mask = const Xbyak::Opmask;

  reg64_t reg_inp = r8;
  reg64_t reg_ker = r9;
  reg64_t reg_out = r10;  // when fuse 1x1, do not need 3x3 out
  reg64_t aux_reg_inp = r11;
  reg64_t aux_reg_ker = r12;
  reg64_t reg_acc_s32 = r13;
  reg64_t reg_scratch_3x3 = r14;
  reg64_t reg_kj = rax;
  reg64_t reg_ptr_scales = rax;
  reg64_t reg_oi = rbx;  // can not use rbx any more
  reg64_t reg_bias = rdx;
  reg64_t reg_kh = abi_not_param1;
  reg64_t param = abi_param1;
  reg64_t reg_channel = r15;
  reg64_t reg_tmp = rbp;
  reg64_t imm_addr64 = r15;

  op_mask pool_k_cmp_mask = op_mask(7);

  op_mask mask(int idx) {
    return op_mask(6 - idx);
  }
  zmm_t zmm_tmp = zmm_t(28);
  zmm_t zmm_one = zmm_t(29);
  zmm_t zmm_scales = zmm_t(30);
  zmm_t zmm_bcast = zmm_t(30);
  zmm_t zmm_zero = zmm_t(31);
  zmm_t zmm_wei = zmm_t(31);

  zmm_t zmm_out(int i_ur, int i_oc) {
    int idx = i_ur + i_oc * jcp.ur_w;
    assert(idx < ker_reg_base_idx);
    return zmm_t(idx);
  }

  xmm_t xmm_out(int i_ur, int i_oc) {
    int idx = i_ur + i_oc * jcp.ur_w;
    assert(idx < ker_reg_base_idx);
    return xmm_t(idx);
  }

  zmm_t zmm_inp(int i_ic, int nb_x_blocking) {
    int idx = i_ic + nb_x_blocking * jcp.ur_w;
    assert(idx < 31);
    return zmm_t(idx);
  }

  int get_ow_start(int ki, int pad_l) {
    return std::max(0, (pad_l - ki + jcp.sw - 1) / jcp.sw);
  }

  int get_ow_end(int ur_w, int ki, int pad_r) {
    return ur_w -
           std::max(0, (ki + pad_r - (jcp.kw - 1) + jcp.sw - 1) / jcp.sw);
  }

  bool maybe_relu(int position);
  void prepare_output(int ur_w);
  void store_output(int ur_w);
  void compute_loop(int ur_w, int pad_l, int pad_r);

  void prepare_1x1output(int ur_w);
  void store_1x1output(int ur_w, int ocb1x1);

  void generate();
};

}
}
