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

#include "jit_pool_kernel.h"
#include "deepfusion_utils.h"

namespace deepfusion {
namespace jit {

using namespace Xbyak;
using namespace deepfusion::alg_kind;

void jit_pool_kernel::load_src(int jj, int ll, int c_tail) {
    using data_type = memory::dtype;

    int c_block = jpp.c_block;
    int ur_c = jpp.ur_c;

    switch (jpp.alg) {
        case pooling_max: {
            auto offset = jj*c_block*sizeof_src_dt();
            if (jj == ur_c - 1 && c_tail) {
                if (jpp.src_dt == data_type::s32) {
                    vmovups(vreg_src(jj) | mask(0),
                            ptr[aux_reg_src_w + offset]);
                } else {
                    vmovdqu8(vreg_src(jj) | mask(0),
                            ptr[aux_reg_src_w + offset]);
                }
            } else {
                vmovups(vreg_src(jj), ptr[aux_reg_src_w + offset]);
            }
            break;
        }
        case pooling_avg_include_padding:
        case pooling_avg_exclude_padding: {
            auto offset = (ll*(c_block/4) + jj*c_block)*sizeof_src_dt();
            if (jj == jpp.ur_c - 1 && c_tail) {
                if (jpp.tail[ll]) {
                    switch (jpp.src_dt) {
                        case data_type::s32:
                            vmovups(vreg_src_s32(jj, ll) | mask(ll),
                                    ptr[aux_reg_src_w + offset]);
                            break;
                        case data_type::s8:
                            vpmovsxbd(vreg_src_s32(jj, ll) | mask(ll),
                                    ptr[aux_reg_src_w + offset]);
                            break;
                        case data_type::u8:
                            vpmovzxbd(vreg_src_s32(jj, ll) | mask(ll),
                                    ptr[aux_reg_src_w + offset]);
                            break;
                        default: assert(!"unsopported src data type");
                    }
                }
            } else {
                switch (jpp.src_dt) {
                    case data_type::s32:
                        vmovups(vreg_src_s32(jj, ll),
                                ptr[aux_reg_src_w + offset]);
                        break;
                    case data_type::s8:
                        vpmovsxbd(vreg_src_s32(jj, ll),
                                ptr[aux_reg_src_w + offset]);
                        break;
                    case data_type::u8:
                        vpmovzxbd(vreg_src_s32(jj, ll),
                                ptr[aux_reg_src_w + offset]);
                        break;
                    default: assert(!"unsopported src data type");
                }
            }
            break;
        }
        default: assert(!"unsupported algorithm");
    }
}

void jit_pool_kernel::store_dst(int jj, int ll,
        int c_tail) {
    using data_type = memory::dtype;

    int c_block = jpp.c_block;
    int ur_c = jpp.ur_c;

    switch(jpp.alg) {
        case pooling_max: {
            auto offset = jj*c_block*sizeof_dst_dt();
            if (jj == ur_c - 1 && c_tail) {
                if (jpp.src_dt == data_type::s32) {
                    vmovups(ptr[reg_ptr_dst_i8 + offset],
                           vreg_dst(jj) | mask(0));
                } else {
                    vmovdqu8(ptr[reg_ptr_dst_i8 + offset],
                            vreg_dst(jj) | mask(0));
                }
            } else {
                vmovups(ptr[reg_ptr_dst_i8 + offset], vreg_dst(jj));
            }
            break;
        }
        case pooling_avg_include_padding:
        case pooling_avg_exclude_padding: {
            auto offset = (ll*(c_block/4) + jj*c_block)*sizeof_dst_dt();
            if (jj == ur_c - 1 && c_tail) {
                if (jpp.tail[ll]) {
                    switch (jpp.dst_dt) {
                        case data_type::s32:
                            vmovups(ptr[reg_ptr_dst_i8 + offset],
                                vreg_dst_s32(jj, ll) | mask(ll));
                            break;
                        case data_type::s8:
                            vpmovdb(ptr[reg_ptr_dst_i8 + offset],
                                vreg_dst_s32(jj, ll) | mask(ll));
                            break;
                        case data_type::u8:
                            vpmovusdb(ptr[reg_ptr_dst_i8 + offset],
                                vreg_dst_s32(jj, ll) | mask(ll));
                            break;
                        default: assert(!"unsupported dst data_type");
                    }
                }
            } else {
                switch (jpp.dst_dt) {
                    case data_type::s32:
                        vmovups(ptr[reg_ptr_dst_i8 + offset],
                            vreg_dst_s32(jj, ll));
                        break;
                    case data_type::s8:
                        vpmovdb(ptr[reg_ptr_dst_i8 + offset],
                            vreg_dst_s32(jj, ll));
                        break;
                    case data_type::u8:
                        vpmovusdb(ptr[reg_ptr_dst_i8 + offset],
                            vreg_dst_s32(jj, ll));
                        break;
                    default: assert(!"unsuppotred dst data_type");
                }
            }
            break;
        }
        default: assert(!"unsupported pooling algorithm");
    }
}

void jit_pool_kernel::compute_max_step(int ur_c, int c_tail)
{
    using data_type = memory::dtype;
    Label l_kw, l_kh;

    int iw = jpp.iw;
    int c = jpp.c;

    for (int jj = 0; jj < ur_c; jj++)
        vmovups(vreg_dst(jj), vreg_tmp);

    mov(aux_reg_src_h, reg_ptr_src_i8);

    xor_(kj, kj);
    L(l_kh);
    {
        mov(aux_reg_src_w, aux_reg_src_h);
        xor_(ki, ki);
        L(l_kw);
        {
            for (int jj = 0; jj < ur_c; jj++) {
                load_src(jj, 0, c_tail);
                if (jpp.src_dt == data_type::s32) {
                    vpcmpd(k_cmp_mask, vreg_dst(jj), vreg_src(jj), _cmp_lt_os);
                    vpblendmd(vreg_dst(jj) | k_cmp_mask, vreg_dst(jj),
                            vreg_src(jj));
                } else {
                    vpcmpb(k_cmp_mask, vreg_dst(jj), vreg_src(jj), _cmp_lt_os);
                    vpblendmb(vreg_dst(jj) | k_cmp_mask, vreg_dst(jj),
                            vreg_src(jj));
                }
            }
            add(aux_reg_src_w, c * sizeof_src_dt());
            inc(ki);
            cmp(ki, reg_kw);
            jl(l_kw, T_NEAR);
        }
        add(aux_reg_src_h, iw * c * sizeof_src_dt());
        inc(kj);
        cmp(kj, reg_kh);
        jl(l_kh, T_NEAR);
    }

    for (int jj = 0; jj < ur_c; jj++)
        store_dst(jj, 0, c_tail);
}

void jit_pool_kernel::compute_avg_step(int ur_c, int c_tail)
{
    using data_type = memory::dtype;

    Label l_kw, l_kh, l_optimize, l_finish, l_loop, l_skip;

    int iw = jpp.iw;
    int c = jpp.c;

    int num_ll = jpp.src_dt == data_type::s32 ? 1 : 4;

    for (int jj = 0; jj < ur_c; jj++) {
        for (int ll = 0; ll < 4; ll++) {
            uni_vpxor(vreg_src_s32(jj, ll),
                    vreg_src_s32(jj, ll), vreg_src_s32(jj, ll));
            uni_vpxor(vreg_dst_s32(jj, ll),
                    vreg_dst_s32(jj, ll), vreg_dst_s32(jj, ll));
        }
    }

    mov(aux_reg_src_h, reg_ptr_src_i8);

    xor_(kj, kj);
    L(l_kh);
    {
        mov(aux_reg_src_w, aux_reg_src_h);
        xor_(ki, ki);
        L(l_kw);
        {
            for (int jj = 0; jj < ur_c; jj++) {
                for (int ll = 0; ll < num_ll; ll++) {
                    load_src(jj, ll, c_tail);
                    vpaddd(vreg_dst_s32(jj, ll),
                            vreg_dst_s32(jj, ll), vreg_src_s32(jj, ll));
                }
            }
            add(aux_reg_src_w, c * sizeof_src_dt());
            inc(ki);
            cmp(ki, reg_kw);
            jl(l_kw, T_NEAR);
        }
        add(aux_reg_src_h, iw * c * sizeof_src_dt());
        inc(kj);
        cmp(kj, reg_kh);
        jl(l_kh, T_NEAR);
    }
    
    xor_(ki, ki);
    cmp(reg_info, ki);
    jge(l_optimize, T_NEAR);
    {
        for (int jj = 0; jj < ur_c; jj++) {
            for (int ll = 0; ll < num_ll; ll++) {
                vcvtdq2ps(vreg_dst_f32(jj, ll), vreg_dst_s32(jj, ll));
                vfmadd132ps(vreg_dst_f32(jj, ll), vreg_zeros, vreg_tmp);
                vcvtps2dq(vreg_dst_s32(jj, ll) | T_rd_sae, vreg_dst_f32(jj, ll));

                store_dst(jj, ll, c_tail);
            }
        }
    }
    jmp(l_finish, T_NEAR);
    L(l_optimize); 
    {
        {
            for (int jj = 0; jj < ur_c; jj++) {
                for (int ll = 0; ll < num_ll; ll++) {
                    movq(xmm_tmp, reg_info);
                    vpsrad(vreg_dst_s32(jj, ll), vreg_dst_s32(jj, ll), xmm_tmp);

                    store_dst(jj, ll, c_tail);
                }
            }
        }
    }
    L(l_finish);
}

void jit_pool_kernel::compute_step(int ur_c, int c_tail) {
    switch (jpp.alg) {
        case pooling_max:
            compute_max_step(ur_c, c_tail); break;
        case pooling_avg_include_padding:
        case pooling_avg_exclude_padding:
            compute_avg_step(ur_c, c_tail); break;
        default: assert(!"unsupported pooling algorithm");
    }
}

void jit_pool_kernel::compute_c_block(){
    Label l_main_loop;

    int nb_c = jpp.nb_c;
    int c_block = jpp.c_block;
    int ur_c = jpp.ur_c;
    int ur_c_tail = jpp.ur_c_tail;
    int c_steps = nb_c / ur_c;
    int c_tail = jpp.c_tail;

    xor_(c_iter, c_iter);
    if (c_steps > 0) {
        L(l_main_loop); {
            compute_step(ur_c, 0);
            add(reg_ptr_src_i8, ur_c*c_block*sizeof_src_dt());
            add(reg_ptr_dst_i8, ur_c*c_block*sizeof_dst_dt());
            inc(c_iter);
            cmp(c_iter, c_steps);
            jl(l_main_loop, T_NEAR);
        }
    }

    if (ur_c_tail != 0) {
        compute_step(ur_c_tail, c_tail);
    }
}

void jit_pool_kernel::init_mask() {
    for (int i = 0; i < 4; i++) {
        mov(reg_mask, jpp.tail[i]);
        kmovq(mask(i), reg_mask);
    }
}

void jit_pool_kernel::init_tmp_reg() {
    using data_type = memory::dtype;

    switch (jpp.alg) {
        case pooling_avg_include_padding:
        case pooling_avg_exclude_padding:
            mov(reg_tmp, ptr[abi_param1 + offsetof(jit_pool_call_t, idivider)]);
            movq(xmm_tmp, reg_tmp);
            vpbroadcastd(vreg_tmp, xmm_tmp);
            break;
        case pooling_max:
            switch (jpp.src_dt) {
                case data_type::s32:
                    mov(reg_tmp, std::numeric_limits<int32_t>::lowest());
                    break;
                case data_type::s8:
                    mov(reg_tmp, std::numeric_limits<int8_t>::lowest());
                    break;
                case data_type::u8:
                    mov(reg_tmp, std::numeric_limits<uint8_t>::lowest());
                    break;
                default: assert(!"unsupported src data_type");
            }

            movq(xmm_tmp, reg_tmp);
            if (jpp.src_dt == data_type::s32)
                vpbroadcastd(vreg_tmp, xmm_tmp);
            else
                vpbroadcastb(vreg_tmp, xmm_tmp);
            break;
        default: assert(!"unsupported pooling algorithm");
    }

}

void jit_pool_kernel::generate() {
    preamble();

#   define READ_PARAM(reg, field) \
        mov(reg, ptr[abi_param1 + offsetof(jit_pool_call_t, field)])
    READ_PARAM(reg_ptr_src_i8, src_i8);
    READ_PARAM(reg_ptr_dst_i8, dst_i8);
    READ_PARAM(reg_kw, kw_range);
    READ_PARAM(reg_kh, kh_range);
    READ_PARAM(reg_info, move_bits);

#   undef READ_PARAM

    init_tmp_reg();
    init_mask();

    uni_vpxor(vreg_zeros, vreg_zeros, vreg_zeros);

    compute_c_block();

    postamble();
}

bool jit_pool_kernel::init_conf(jit_pool_conf_t &jpp,
        const std::unique_ptr<memory> &src,
        std::unique_ptr<memory> &dst,
        std::array<int, 2> stride,
        std::array<int, 2> padding,
        std::array<int, 2> kernel,
        alg_kind_t alg) {
    using data_type = memory::dtype;
    if (!mayiuse(avx512_core)) {
        return false;
    }

    auto src_d = src->std_dims();
    auto dst_d = dst->std_dims();

    jpp.mb = src_d[0];
    jpp.c = src_d[1];
    jpp.ih = src_d[2];
    jpp.iw = src_d[3];
    jpp.oh = dst_d[2];
    jpp.ow = dst_d[3];

    jpp.stride_h = stride[0];
    jpp.stride_w = stride[1];
    jpp.kh = kernel[0];
    jpp.kw = kernel[1];

    jpp.t_pad = padding[0];
    jpp.l_pad = padding[1];

    jpp.alg = alg;

    jpp.src_dt =  data_type::s32;
    jpp.dst_dt =  data_type::s32;

    jpp.c_block = 64 / (jpp.src_dt == data_type::s32 ? 4 : 1);
    jpp.c_tail = jpp.c % jpp.c_block;
    jpp.nb_c = jpp.c / jpp.c_block;
    jpp.ur_c = 1;
    jpp.ur_c_tail = jpp.nb_c - (jpp.nb_c / jpp.ur_c)*jpp.ur_c +
            (jpp.c_tail != 0);

    size_t tail_mask = (1ULL << jpp.c_tail) - 1;

    switch(jpp.alg) {
        case pooling_max:
            jpp.tail[0] = tail_mask;
            jpp.tail[1] = 0;
            jpp.tail[2] = 0;
            jpp.tail[3] = 0;
            break;
        case pooling_avg_include_padding:
        case pooling_avg_exclude_padding:
            jpp.tail[0] = tail_mask & 0xffff;
            for (size_t i = 1, m = tail_mask; i < 4; i++) {
                m = m >> 16;
                jpp.tail[i] = m & 0xffff;
            }
            break;
        default: return false;
    }

    return true;
}


}
}
