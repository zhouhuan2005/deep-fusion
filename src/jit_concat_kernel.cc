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
#include "util.h"

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
                            // TODO: enable dtype
                            /*
                              switch (jcp.dtype) {
                                case data_type::s32: vpmaxsw(zmm_src, zmm_src, zmm_zero); break;
                                case data_type::f32: vmaxps(zmm_src, zmm_zero, zmm_src); break;
                                default: assert(!"error dtype");
                              }
                              */
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
bool jit_concat_kernel::init_conf() { return true; }
}
}
