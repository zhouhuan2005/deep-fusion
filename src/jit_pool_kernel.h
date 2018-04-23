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

#ifndef JIT_POOL_KERNEL_H
#define JIT_POOL_KERNEL_H


#include "jit_call_conf.h"
#include "jit_generator.h"

namespace deepfusion {
namespace jit {

using namespace Xbyak;
struct jit_pool_kernel : public jit_generator {
  DECLARE_JIT_KERNEL(jit_pool_kernel);

  Reg64 reg_ptr_src_i8 = r8;
  Reg64 reg_ptr_dst_i8 = r9;

  Reg64 ki = r10;
  Reg64 kj = r11;
  Reg64 reg_kw = r12;
  Reg64 reg_kh = r13;
  Reg64 c_iter = r14;

  Reg64 aux_reg_src_h = rax;
  Reg64 aux_reg_src_w = rbx;

  Reg64 reg_tmp = rdx;
  Reg64 reg_info = rbp;

  Reg64 reg_mask = r15;

  Opmask k_cmp_mask = Opmask(7);

  Opmask mask(int idx) {
    return Opmask(6 - idx);
  }

  Xmm xmm_tmp = Xmm(0);
  Zmm zmm_tmp = Zmm(1);
  Zmm vreg_tmp = Zmm(30);
  Zmm vreg_zeros = Zmm(31);

  size_t sizeof_src_dt() const { 
    using data_type = memory::dtype;
    switch (jpp.src_dt) {
        case data_type::f32: return sizeof(float);
        case data_type::s32: return sizeof(int32_t);
        case data_type::s8: return sizeof(int8_t);
        case data_type::u8: return sizeof(uint8_t);
        case data_type::undef:
        default: assert(!"unknown data_type");
        }
        return 0; /* not supposed to be reachable */
  }
  size_t sizeof_dst_dt() const {
    using data_type = memory::dtype;
    switch (jpp.dst_dt) {
    case data_type::f32: return sizeof(float);
    case data_type::s32: return sizeof(int32_t);
    case data_type::s8: return sizeof(int8_t);
    case data_type::u8: return sizeof(uint8_t);
    case data_type::undef:
    default: assert(!"unknown data_type");
    }
    return 0; /* not supposed to be reachable */
  }

  /* max pooling */
  Zmm vreg_src(int idx) {
    return Zmm(idx);
  }

  Zmm vreg_dst(int idx) {
    return Zmm(jpp.ur_c + idx);
  }

  /* avg pooling */
  Zmm vreg_src_s32(int jj, int ll) {
    return Zmm(12*jj + ll);
  }

  Zmm vreg_dst_s32(int jj, int ll) {
    return Zmm(12*jj + ll + 4);
  }

  Zmm vreg_dst_f32(int jj, int ll) {
    return Zmm(12*jj + ll + 8);
  }

  void (*ker_)(const jit_pool_call_t *);
  jit_pool_conf_t jpp;

  void init_tmp_reg();
  void init_mask();

  void load_src(int jj, int ll, int c_tail);
  void store_dst(int jj, int ll, int c_tail);

  void compute_avg_step(int ur_c, int c_tail);
  void compute_max_step(int ur_c, int c_tail);
  void compute_step(int ur_c, int c_tail);

  void compute_c_block();
  void generate();

  static bool init_conf(jit_pool_conf_t &jpp,
        const std::unique_ptr<memory> &src,
        std::unique_ptr<memory> &dst,
        std::array<int, 2> stride,
        std::array<int, 2> padding,
        std::array<int, 2> kernel,
        alg_kind_t alg);

  jit_pool_kernel(jit_pool_conf_t ajpp) : jpp(ajpp) {
    generate();
    ker_ = (void (*)(const jit_pool_call_t*))getCode();
  }
};


}
}

#endif
