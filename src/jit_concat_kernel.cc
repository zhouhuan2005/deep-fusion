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

// @note: do not use any MACRO or #define inside JIT kernel
// it would have some uncertain issue in JIT, need figure out why

namespace jitinfer {
namespace jit {

using namespace Xbyak;

void jit_concat_kernel::compute_one_input_with_zmm() {
  Label l_next_block;
  int shift_c = jcp_.typesize * jcp_.block;
  mov(reg_nb, dword[reg_ptr_nb_ic]);
  mov(reg_ptr_src_i, ptr[reg_ptr_src]);
  L(l_next_block);
  {
    auto src_addr = EVEX_compress_addr(reg_ptr_src_i, 0);
    auto dst_addr = EVEX_compress_addr(reg_ptr_dst, 0);
    // load to src
    vmovups(zmm_src, src_addr);
    if (jcp_.with_relu) {
      if (jcp_.dt == memory::dtype::s32) {
        vpmaxsw(zmm_src, zmm_src, zmm_zero);
      } else if (jcp_.dt == memory::dtype::f32) {
        vmaxps(zmm_src, zmm_zero, zmm_src);
      } else {  // s8 or u8
        vpmaxsb(zmm_src, zmm_src, zmm_zero);
      }
    }
    // save to dst
    vmovups(dst_addr, zmm_src);
    add(reg_ptr_src_i, shift_c);
    add(reg_ptr_dst, shift_c);
    dec(reg_nb);
    cmp(reg_nb, 0);
    jg(l_next_block, T_NEAR);
  }
}

void jit_concat_kernel::compute_one_input_with_ymm() {
  Label l_next_block;
  int shift_c = jcp_.typesize * jcp_.block;
  mov(reg_nb, dword[reg_ptr_nb_ic]);
  mov(reg_ptr_src_i, ptr[reg_ptr_src]);
  L(l_next_block);
  {
    auto src_addr = EVEX_compress_addr(reg_ptr_src_i, 0);
    auto dst_addr = EVEX_compress_addr(reg_ptr_dst, 0);
    // load to src
    vmovups(ymm_src, src_addr);
    if (jcp_.with_relu) {
      if (jcp_.dt == memory::dtype::s32) {
        vpmaxsw(ymm_src, ymm_src, ymm_zero);
      } else if (jcp_.dt == memory::dtype::f32) {
        vmaxps(ymm_src, ymm_zero, ymm_src);
      } else {  // s8 or u8
        vpmaxsb(ymm_src, ymm_src, ymm_zero);
      }
    }
    // save to dst
    vmovups(dst_addr, ymm_src);
    add(reg_ptr_src_i, shift_c);
    add(reg_ptr_dst, shift_c);
    dec(reg_nb);
    cmp(reg_nb, 0);
    jg(l_next_block, T_NEAR);
  }
}

void jit_concat_kernel::compute_one_input_with_xmm() {
  Label l_next_block;
  int shift_c = jcp_.typesize * jcp_.block;
  mov(reg_nb, dword[reg_ptr_nb_ic]);
  mov(reg_ptr_src_i, ptr[reg_ptr_src]);
  L(l_next_block);
  {
    auto src_addr = EVEX_compress_addr(reg_ptr_src_i, 0);
    auto dst_addr = EVEX_compress_addr(reg_ptr_dst, 0);
    // load to src
    vmovups(xmm_src, src_addr);
    if (jcp_.with_relu) {
      if (jcp_.dt == memory::dtype::s32) {
        vpmaxsw(xmm_src, xmm_src, xmm_zero);
      } else if (jcp_.dt == memory::dtype::f32) {
        vmaxps(xmm_src, xmm_zero, xmm_src);
      } else {  // s8 or u8
        vpmaxsb(xmm_src, xmm_src, xmm_zero);
      }
    }
    // save to dst
    vmovups(dst_addr, xmm_src);
    add(reg_ptr_src_i, shift_c);
    add(reg_ptr_dst, shift_c);
    dec(reg_nb);
    cmp(reg_nb, 0);
    jg(l_next_block, T_NEAR);
  }
}

void jit_concat_kernel::compute_with_zmm() {
  xor_(reg_ninputs, reg_ninputs);
  Label l_next_input;
  L(l_next_input);
  {
    compute_one_input_with_zmm();
    add(reg_ptr_src, sizeof(void*));  // move 64bits
    add(reg_ptr_nb_ic, sizeof(int));  // move one int
    inc(reg_ninputs);
    cmp(reg_ninputs, jcp_.n_inputs);
    jl(l_next_input, T_NEAR);
  }
}

void jit_concat_kernel::compute_with_ymm() {
  xor_(reg_ninputs, reg_ninputs);
  Label l_next_input;
  L(l_next_input);
  {
    compute_one_input_with_ymm();
    add(reg_ptr_src, sizeof(void*));  // move 64bits
    add(reg_ptr_nb_ic, sizeof(int));  // move one int
    inc(reg_ninputs);
    cmp(reg_ninputs, jcp_.n_inputs);
    jl(l_next_input, T_NEAR);
  }
}

void jit_concat_kernel::compute_with_xmm() {
  xor_(reg_ninputs, reg_ninputs);
  Label l_next_input;
  L(l_next_input);
  {
    compute_one_input_with_xmm();
    add(reg_ptr_src, sizeof(void*));  // move 64bits
    add(reg_ptr_nb_ic, sizeof(int));  // move one int
    inc(reg_ninputs);
    cmp(reg_ninputs, jcp_.n_inputs);
    jl(l_next_input, T_NEAR);
  }
}

void jit_concat_kernel::generate() {
  preamble();

  mov(reg_ptr_src, ptr[param + GET_OFF(src)]);
  mov(reg_ptr_nb_ic, ptr[param + GET_OFF(nb_ic)]);
  mov(reg_ptr_dst, ptr[param + GET_OFF(dst)]);

  // one kernel move one dst oc from all srcs
  mov(reg_bitsize, jcp_.bits_size);
  Label l_use_ymm, l_use_xmm, l_ret;
  cmp(reg_bitsize, 512);
  jne(l_use_ymm, T_NEAR);
  vpxord(zmm_zero, zmm_zero, zmm_zero);
  compute_with_zmm();
  jmp(l_ret, T_NEAR);

  L(l_use_ymm);
  cmp(reg_bitsize, 256);
  jne(l_use_xmm, T_NEAR);
  vpxord(ymm_zero, ymm_zero, ymm_zero);
  compute_with_ymm();
  jmp(l_ret, T_NEAR);

  L(l_use_xmm);
  vpxord(xmm_zero, xmm_zero, xmm_zero);
  compute_with_xmm();

  L(l_ret);
  postamble();
}
bool jit_concat_kernel::init_conf(
    jit_concat_conf_t& jcp,
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
  if (!util::one_of(dst->dim_format(), memory::format::nhwc)) {
    // only support nhwc yet
    return false;
  }

  // when 4bytes, work on 16x, 8x or 4x channels
  // when 1byte, work on 64x, 32x, 16x channels
  // fine the workable block.
  std::vector<int> blocks;
  if (jcp.typesize == 1) {
    blocks = {64, 32, 16};
  } else {  // typesize == 4
    blocks = {16, 8, 4};
  }
  for (size_t k = 0; k < blocks.size(); ++k) {
    jcp.block = blocks[k];
    size_t i;
    for (i = 0; i < srcs.size(); ++i) {
      if (srcs[i]->actual_dims()[3] % jcp.block != 0) {
        // not dividable
        break;
      }
    }
    if (i == srcs.size()) {  // this block is dividable by all inputs channels
      break;
    }
  }

  for (size_t i = 0; i < srcs.size(); ++i) {
    if (srcs[i]->dim_format() != dst->dim_format()) {
      // all format should be equal
      return false;
    }
    if (srcs[i]->data_type() != jcp.dt) {
      // all data type must equals
      return false;
    }
    if (srcs[i]->actual_dims()[3] % jcp.block != 0) {
      return false;
    }
  }

  jcp.bits_size = 8 * jcp.typesize * jcp.block;
  if (!util::one_of(jcp.bits_size, 128, 256, 512)) {
    // xmm, ymm, zmm
    return false;
  }
  return true;
}
}
}
