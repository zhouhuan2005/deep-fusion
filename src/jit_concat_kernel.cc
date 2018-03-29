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
#include "jit_concat_kernel.h"
#include "util_jitinfer.h"

#define GET_OFF(field) offsetof(jit_concat_call_s, field)

namespace jitinfer {

namespace jit {

using namespace Xbyak;

void jit_concat_kernel::compute_one_input() {
  Label l_next_block;
  int shift_c = jcp.typesize * jcp.block;
  mov(reg_nb, dword[reg_ptr_nb_ic]);
  mov(reg_ptr_src_i, ptr[reg_ptr_src]);

  L(l_next_block);
  {
    auto src_addr = EVEX_compress_addr(reg_ptr_src_i, 0);
    auto dst_addr = EVEX_compress_addr(reg_ptr_dst, 0);
    if (jcp.typesize == 4) {
      // load from dst
      vmovups(zmm_src, src_addr);
      if (jcp.with_relu) {  // relu
        switch (jcp.dt) {
          case memory::dtype::s32:
            vpmaxsw(zmm_src, zmm_src, zmm_zero);
            break;
          case memory::dtype::f32:
            vmaxps(zmm_src, zmm_zero, zmm_src);
            break;
          default:
            assert(!"error dtype");
        }
      }
      // save to dst
      vmovups(dst_addr, zmm_src);
    } else {
      vmovups(xmm_src, src_addr);
      if (jcp.with_relu) {
        vpmaxsb(xmm_src, xmm_src, xmm_zero);
      }
      // save to dst
      vmovups(dst_addr, xmm_src);
    }
    add(reg_ptr_src_i, shift_c);
    add(reg_ptr_dst, shift_c);
    dec(reg_nb);
    cmp(reg_nb, 0);
    jg(l_next_block, T_NEAR);
  }
}

void jit_concat_kernel::generate() {
  preamble();

  mov(reg_ptr_src, ptr[param + GET_OFF(src)]);
  mov(reg_ptr_nb_ic, ptr[param + GET_OFF(nb_ic)]);
  mov(reg_ptr_dst, ptr[param + GET_OFF(dst)]);

  if (jcp.typesize == 4) {
    vpxord(zmm_zero, zmm_zero, zmm_zero);
  } else {
    vpxord(xmm_zero, xmm_zero, xmm_zero);
  }

  xor_(reg_ninputs, reg_ninputs);
  Label l_next_input;
  L(l_next_input);
  {
    compute_one_input();
    add(reg_ptr_src, sizeof(void*));  // move 64bits
    add(reg_ptr_nb_ic, sizeof(int));  // move one int
    inc(reg_ninputs);
    cmp(reg_ninputs, jcp.n_inputs);
    jl(l_next_input, T_NEAR);
  }

  postamble();
}
bool jit_concat_kernel::init_conf(
    const std::vector<std::unique_ptr<memory>>& srcs,
    const std::unique_ptr<memory>& dst,
    bool post_relu) {
  jcp = jitinfer::util::zero<decltype(jcp)>();

  jcp.n_inputs = srcs.size();
  jcp.with_relu = post_relu;
  auto dm = dst->actual_dims();
  jcp.bs = dm[0];
  jcp.h = dm[1];
  jcp.w = dm[2];
  jcp.oc = dm[3];
  jcp.dt = dst->data_type();
  jcp.typesize = util::dtype_size(jcp.dt);
  if (!util::one_of(jcp.typesize, 1, 4)) {
    // only s8, u8, s32, f32
    return false;
  }
  check_eq(dst->dim_format(), memory::format::nhwc);

  // TODO: when s8 or s32, load more,  can use xmm ymm,
  jcp.block = 16;
  for (size_t i = 0; i < srcs.size(); ++i) {
    check_eq(srcs[i]->dim_format(), memory::format::nhwc);
    check_eq(srcs[i]->data_type(), jcp.dt);
    if (srcs[i]->actual_dims()[3] % jcp.block != 0) {
      return false;
    }
  }

  return true;
}
}
}
