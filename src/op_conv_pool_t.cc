/*******************************************************************************
* Copyright 2016-2018 Intel Corporation
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

#include "op_conv_pool_t.h"
#include "deepfusion_utils.h"

namespace deepfusion {

using namespace utils;

template <typename dst_data_t>
void op_conv_pool_t<dst_data_t>::
infer() 
{
  infer_conv();
  infer_pool();
}

template <typename dst_data_t>
void op_conv_pool_t<dst_data_t>::
infer_conv() 
{

  const auto &jcp = conv_kernel_->jcp;
  assert(jcp.nb_oc % jcp.nb_oc_blocking == 0);
  // bias data type can be any of u8,s8,s32,f32
  auto bias_data = reinterpret_cast<const char *>(bia_data_);

  #pragma omp parallel
  {
    int ithr = omp_get_thread_num(), nthr = omp_get_num_threads();
    int oc_chunks = jcp.nb_oc / jcp.nb_oc_blocking;
    int ic_chunks = jcp.nb_ic / jcp.nb_ic_blocking;

    int start{0}, end{0};
    int work_amount = jcp.bs * jcp.gp * oc_chunks * jcp.oh;
    balance211(work_amount, nthr, ithr, start, end);

    jit::jit_conv_call_t p = {0};
    auto ws_l = ws_ + ithr * ws_per_thread_;
    size_t src_h_stride = jcp.iw * jcp.ic;
    size_t dst_h_stride = jcp.ow * jcp.oc;
    // TODO: below stride should be wrong!
    size_t wht_h_stride = jcp.kw;
    size_t wht_ic_stride = jcp.kh * jcp.kw;

    int n{0}, g{0}, occ{0}, oh_s{0};
    if (jcp.loop_order == loop_cgn) {  // this is default
      nd_iterator_init(
          start, occ, oc_chunks, g, jcp.gp, n, jcp.bs, oh_s, jcp.oh);
    } else if (jcp.loop_order == loop_gnc) {
      nd_iterator_init(
          start, g, jcp.gp, n, jcp.bs, occ, oc_chunks, oh_s, jcp.oh);
    } else if (jcp.loop_order == loop_ngc) {
      nd_iterator_init(
          start, n, jcp.bs, g, jcp.gp, occ, oc_chunks, oh_s, jcp.oh);
    } else {
      assert(!"unsupported loop order");
    }

    while (start < end) {
      int ocb = occ * jcp.nb_oc_blocking;
      int g_oc = (g * jcp.nb_oc + ocb) * jcp.oc_block;
      int g_ic = g * jcp.nb_ic * jcp.oc_block;

      int work_rem = end - start;
      int ih_s = -jcp.t_pad + oh_s * jcp.sh;
      int oh_e = oh_s + work_rem > jcp.oh ? jcp.oh : oh_s + work_rem;

      auto bias_w = bias_data
                        ? bias_data + ((size_t)g_oc * jcp.oc / jcp.gp *
                                       jcp.typesize_conv0_bia)
                        : 0;
      auto dst_w = dst_data_ + (size_t)n * jcp.oc * jcp.oh * jcp.ow +
                   g_oc * jcp.oh * jcp.ow + oh_s * jcp.ow;
      auto src_w = src_data_ + (size_t)n * jcp.ic * jcp.ih * jcp.iw +
                   g_ic * jcp.ih * jcp.iw + ih_s * jcp.iw;
      auto wht_w =
          wei_data_ + (jcp.gp > 1 ? ((size_t)g * jcp.oc * jcp.ic * jcp.kh *
                                         jcp.kw / jcp.gp +
                                     ocb * jcp.ic * jcp.kh * jcp.kw / jcp.gp)
                                  : ((size_t)ocb * jcp.ic * jcp.kh * jcp.kw));
      auto scales =
          conv_scales_data_ ? conv_scales_data_ + g_oc : conv_scales_data_;

      for (int icc = 0; icc < ic_chunks; ++icc) {
        auto src_c = src_w;
        auto dst_c = dst_w;
        auto ws_c = ws_l;
        int icb = icc * jcp.nb_ic_blocking;
        for (int oj = oh_s, ij = ih_s; oj < oh_e; ++oj, ij += jcp.sh) {
          int i_t_overflow = -std::min(0, ij);
          int i_b_overflow = std::max(jcp.ih, ij + jcp.kh) - jcp.ih;
          int kh_padding = std::max(0, jcp.kh - i_t_overflow - i_b_overflow);

          p.src = src_c + i_t_overflow * src_h_stride;
          p.wei = wht_w + i_t_overflow * wht_h_stride;
          p.bia = bias_w;
          p.acc_s32 = ws_c;
          p.channel = icb;
          p.kh_padding = kh_padding;
          p.scales = scales;
          p.dst = dst_c;
          conv_kernel_->jit_ker_(&p);

          src_c += src_h_stride * jcp.sh;
          dst_c += dst_h_stride;
          ws_c += jcp.ow * jcp.oc_block * jcp.nb_oc_blocking;
        }
        src_w += jcp.ic_block * jcp.nb_ic_blocking;
        wht_w += wht_ic_stride * jcp.nb_ic_blocking;
      }

      if (jcp.loop_order == loop_cgn) {
        nd_iterator_jump(
            start, end, occ, oc_chunks, g, jcp.gp, n, jcp.bs, oh_s, jcp.oh);
      } else if (jcp.loop_order == loop_gnc) {
        nd_iterator_jump(
            start, end, g, jcp.gp, n, jcp.bs, occ, oc_chunks, oh_s, jcp.oh);
      } else if (jcp.loop_order == loop_ngc) {
        nd_iterator_jump(
            start, end, n, jcp.bs, g, jcp.gp, occ, oc_chunks, oh_s, jcp.oh);
      } else {
        assert(!"unsupported loop order");
      }
    }
  }
}

template <typename dst_data_t>
void op_conv_pool_t<dst_data_t>::
infer_pool() 
{
  using namespace deepfusion::alg_kind;
  auto src_i8 = reinterpret_cast<const char *>(tmp_data_);
  auto dst_i8 = reinterpret_cast<char *>(dst_data_);

  const auto &jpp = pool_kernel_->jpp;

  auto ker = [&](int ithr, int nthr) {
    const int work_amount = jpp.mb * jpp.oh * jpp.ow;

    int start{0}, end{0}, calc_info, bits_info;
    balance211(work_amount, nthr, ithr, start, end);

    int n{0}, oh{0}, ow{0};
    nd_iterator_init(start, n, jpp.mb, oh, jpp.oh, ow, jpp.ow);

    jit::jit_pool_call_t  p = {};

    for (int iwork = start; iwork < end; ++iwork) {
      const int ih = std::max(oh*jpp.stride_h - jpp.t_pad, 0);
      const int iw = std::max(ow*jpp.stride_w - jpp.l_pad, 0);

      const int kh_start = std::max(0, jpp.t_pad - oh * jpp.stride_h);
      const int kh_end = std::min(jpp.kh,
                    jpp.ih + jpp.t_pad - oh * jpp.stride_h);
      const int kw_start = std::max(0, jpp.l_pad - ow * jpp.stride_w);
      const int kw_end = std::min(jpp.kw,
                    jpp.iw + jpp.l_pad - ow * jpp.stride_w);
      //todo need update calc method
      //p.src_i8 = &src_i8[
      //    src_d.blk_off(n, 0, ih, iw) * src_d.data_type_size()];
      //p.dst_i8 = &dst_i8[
      //    dst_d.blk_off(n, 0, oh, ow) * dst_d.data_type_size()];
      p.kw_range = (size_t)(kw_end - kw_start);
      p.kh_range = (size_t)(kh_end - kh_start);
      p.idivider = 1.0f / ((jpp.alg == pooling_avg_exclude_padding) ?
                p.kh_range*p.kw_range : jpp.kw*jpp.kh);
      calc_info = ((jpp.alg == pooling_avg_exclude_padding) ?
                p.kh_range*p.kw_range : jpp.kw*jpp.kh);
      if (!(calc_info & (calc_info - 1))) {
        for(bits_info = 0; calc_info > 1; bits_info ++) {
          calc_info >>= 1;
        }    
      } else {
        bits_info = -1;
      }
      p.move_bits = (size_t)bits_info;
      pool_kernel_->ker_(&p);

      nd_iterator_step(n, jpp.mb, oh, jpp.oh, ow, jpp.ow);

    }
  };
#   pragma omp parallel
    {
        ker(omp_get_thread_num(), omp_get_num_threads());
    }
  
}

template <typename dst_data_t>
bool op_conv_pool_t<dst_data_t>::init_conf(jit::jit_conv_conf_t &conv_conf,
                 jit::jit_pool_conf_t &pool_conf,
                 const std::unique_ptr<memory> &conv_src,
                 const std::unique_ptr<memory> &conv_wei,
                 const std::unique_ptr<memory> &conv_bia,
                 std::unique_ptr<memory> &conv_dst,
                 int ngroups,  // only enabled on conv0
                 std::array<int, 2> conv_stride,
                 std::array<int, 2> conv_padding,
                 std::vector<float> conv_scales,
                 const std::unique_ptr<memory>  &pool_src,
                 std::unique_ptr<memory>  &pool_dst,
                 std::array<int, 2> pool_stride,
                 std::array<int, 2> pool_padding,
                 std::array<int, 2> pool_kernel,
                 alg_kind_t pool_alg) {
  using namespace utils;
  // check data type
  if (pool_dst->data_type() != type2dtype<dst_data_t>::dtype) {
    info("Dst data type do not match");
    return false;
  }

  // check image size and channels
  constexpr int C = 1, H = 2, W = 3;  // channel, height, width
  auto src_dims = conv_src->std_dims();    // nchw
  auto wei_dims = conv_wei->std_dims();    // oihw
  auto dst_dims = conv_dst->std_dims();    // nchw
  for (size_t i = 0; i < 2; ++i) {
    if (dst_dims[i + 2] !=
        conv_output_size(
            src_dims[i + 2], wei_dims[i + 2], conv_stride[i], conv_padding[i])) {
      info("Output image size do not match: %d", i);
      return false;
    }
  }

  if (src_dims[0] != dst_dims[0]) {
    info("Batch size do not equal");
    return false;
  }

  if (src_dims[C] != wei_dims[C]) {
    info("Input channel do not match");
    return false;
  }

  if (dst_dims[C] != wei_dims[0]) {
    info("Output channel do not match");
    return false;
  }
  if (conv_bia != nullptr && conv_bia->std_dims()[0] != wei_dims[0]) {
    info("Bias channel do not match");
    return false;
  }
  if (!one_of(conv_scales.size(), 1UL, size_t(dst_dims[C]))) {
    return false;
  }

  check_eq(ngroups, 1);  // only verified gp==1 yet
  bool ret = jit::jit_conv_kernel_4_conv_pool::init_conf(conv_conf,
                                         conv_src,
                                         conv_wei,
                                         conv_bia,
                                         ngroups,
                                         conv_stride,
                                         conv_padding,
                                         conv_dst,
                                         conv_scales,
                                         true);
  return ret && jit::jit_pool_kernel::init_conf(pool_conf, 
                                     pool_src, 
                                     pool_dst, 
                                     pool_stride, 
                                     pool_padding, 
                                     pool_kernel, 
                                     pool_alg);
}

template class op_conv_pool_t<f32>;
template class op_conv_pool_t<s32>;
template class op_conv_pool_t<s8>;
template class op_conv_pool_t<u8>;

}

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
