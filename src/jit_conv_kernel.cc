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
                                int ngroups,
                                std::array<int, 2> sz_stride,
                                std::array<int, 2> sz_padding,
                                std::unique_ptr<memory> &dst,
                                std::vector<float> conv0_scales,
                                std::vector<float> conv1_scales,
                                const std::unique_ptr<memory> &wei1x1,
                                const std::unique_ptr<memory> &bia1x1,
                                bool conv0_relu,
                                bool conv1_relu) {
  using namespace util;
  jcp = zero<decltype(jcp)>();
  // Check data type
  if (!all_true(src->data_type() == memory::dtype::u8,
                wei->data_type() == memory::dtype::s8,
                wei1x1 == nullptr || wei1x1->data_type() == memory::dtype::s8,
                one_of(dst->data_type(),
                       memory::dtype::f32,
                       memory::dtype::s32,
                       memory::dtype::s8,
                       memory::dtype::u8),
                bia == nullptr || one_of(bia->data_type(),
                                         memory::dtype::f32,
                                         memory::dtype::s32,
                                         memory::dtype::s8,
                                         memory::dtype::u8),
                bia1x1 == nullptr || one_of(bia1x1->data_type(),
                                            memory::dtype::f32,
                                            memory::dtype::s32,
                                            memory::dtype::s8,
                                            memory::dtype::u8))) {
    return false;
  }
  // Check format
  if (!all_true(one_of(src->dim_format(), memory::format::nhwc),
                one_of(dst->dim_format(), memory::format::nhwc),
                one_of(wei->dim_format(),
                       memory::format::OIhw4i16o4i,
                       memory::format::gOIhw4i16o4i),
                bia == nullptr || one_of(bia->dim_format(), memory::format::x),
                wei1x1 == nullptr || one_of(wei1x1->dim_format(),
                                            memory::format::OIhw4i16o4i,
                                            memory::format::gOIhw4i16o4i),
                bia1x1 == nullptr ||
                    one_of(bia1x1->dim_format(), memory::format::x))) {
    return false;
  }

  jcp.gp = ngroups;
  assert(ngroups == 1);
  auto src_dims = src->std_dims();  // nchw
  auto wei_dims = wei->std_dims();  // oihw
  auto dst_dims = dst->std_dims();  // nchw
  jcp.bs = src_dims[0];
  jcp.ic = src_dims[1] / jcp.gp;
  jcp.ih = src_dims[2];
  jcp.iw = src_dims[3];
  jcp.oc = dst_dims[1] / jcp.gp;
  jcp.oh = dst_dims[2];
  jcp.ow = dst_dims[3];
  jcp.kh = wei_dims[2];
  jcp.kw = wei_dims[3];
  jcp.ph = sz_padding[0];
  jcp.pw = sz_padding[1];
  jcp.sh = sz_stride[0];
  jcp.sw = sz_stride[1];
  jcp.ic_block = 16;
  jcp.oc_block = 16;
  jcp.nb_ic = jcp.ic / jcp.ic_block;
  jcp.nb_oc = jcp.oc / jcp.oc_block;
  if (!all_true(jcp.ic % jcp.ic_block == 0, jcp.oc % jcp.oc_block == 0)) {
    return false;
  }

  jcp.fuse_conv1x1 = wei1x1 != nullptr;
  if (jcp.fuse_conv1x1) {
    if (jcp.oc_block % 4 != 0) {
      // for 4 bcast
      return false;
    }
    assert(wei1x1 != nullptr);
    auto wei1x1_dims = wei1x1->std_dims();  // oihw
    jcp.oc1x1 = wei1x1_dims[0];
    if (!all_true(jcp.oc == wei1x1_dims[1],
                  jcp.oh == wei1x1_dims[2],
                  jcp.ow == wei1x1_dims[3])) {
      return false;
    }
    jcp.oc1x1_block = 16;
    jcp.nb_oc1x1 = jcp.oc1x1 / jcp.oc1x1_block;
    if (jcp.oc1x1 % jcp.oc1x1_block != 0) {
      return false;
    }
  }

  auto undef_dt = memory::dtype::undef;
  jcp.conv0_with_bias = bia != nullptr;
  jcp.conv1_with_bias = bia1x1 != nullptr;
  jcp.conv0_bias_dt = jcp.conv0_with_bias ? bia->data_type() : undef_dt;
  jcp.conv1_bias_dt = jcp.conv1_with_bias ? bia1x1->data_type() : undef_dt;
  jcp.dst_dt = dst->data_type();
  jcp.typesize_in = dtype_size(src->data_type());
  jcp.typesize_out = dtype_size(dst->data_type());
  jcp.typesize_acc = sizeof(s32);
  jcp.typesize_conv0_bia =
      jcp.conv0_with_bias ? dtype_size(bia->data_type()) : 0;
  jcp.typesize_conv1_bia =
      jcp.conv1_with_bias ? dtype_size(bia1x1->data_type()) : 0;
  jcp.conv0_with_relu = conv0_relu;
  jcp.conv1_with_relu = conv1_relu;

  // conv 3x3 blocking settings
  jcp.nb_ic_blocking = dividable_of(jcp.nb_ic, 8, 4, 2, 1);
  if (jcp.kh >= 7 || jcp.kw >= 7) {  // Note: maybe have large code issue on SKX
    jcp.nb_ic_blocking = dividable_of(jcp.nb_ic, 4, 2, 1);
  }
  jcp.nb_oc_blocking = jcp.nb_oc > 4 ? 4 : jcp.nb_oc;
  if (jcp.nb_oc % jcp.nb_oc_blocking != 0) {
    jcp.nb_oc_blocking = find_dividable(jcp.nb_oc, jcp.nb_oc_blocking);
  }

  // the rest 1 size of ur_w is for src input zmm
  jcp.ur_w = ker_reg_base_idx / (jcp.nb_oc_blocking + 1);
  if (jcp.ow < jcp.ur_w) jcp.ur_w = jcp.ow;
  jcp.ur_w_tail = jcp.ow % jcp.ur_w;

  int r_pad_no_tail = std::max(
      0, (jcp.ow - jcp.ur_w_tail - 1) * jcp.sw + jcp.kw - jcp.iw - jcp.pw);
  if (jcp.pw > jcp.ur_w || r_pad_no_tail > jcp.ur_w) {
    return false;
  }

  jcp.conv0_multi_oc_scale = conv0_scales.size() > 1;
  jcp.conv1_multi_oc_scale = conv1_scales.size() > 1;
  if (conv0_scales.size() > 0 && conv0_scales.size() != jcp.oc) {
    return false;
  }
  if (jcp.fuse_conv1x1 &&
      (conv1_scales.size() > 0 && conv1_scales.size() != jcp.oc1x1)) {
    return false;
  }
  return true;
}
}
}
