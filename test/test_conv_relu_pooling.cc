#include "test_utils.h"
#include "log.h"

using namespace deepfusion;
using format = deepfusion::memory::format;

struct test_conv_relu_pool_params {
   memory::nchw_dims src_dims;
   memory::pair_dims conv_kernel;
   memory::pair_dims conv_pad;
   memory::pair_dims conv_stride;
   memory::nchw_dims conv_dst_dims;

   bool with_eltwise_sum;
   bool with_relu;

   memory::pair_dims pool_kernel;
   memory::pair_dims pool_pad;
   memory::pair_dims pool_stride;
   memory::nchw_dims dst_dims;

   bool pooling_avg_include_padding;
   bool pooling_avg_exclude_padding;  
   bool max_pooling;
};


template <typename data_t_src, typename data_t_wei,
         typename data_t_acc, typename data_t_dst>
class conv_relu_pooling_test : public ::testing::TestWithParam<test_conv_relu_pool_params> {

  void check_result(const test_conv_relu_pool_params& pm,
                    const std::unique_ptr<memory>& src,
                    const std::unique_ptr<memory>& wei,
                    const std::unique_ptr<memory>& bias,
                    const std::unique_ptr<memory>& dst){

      mkldnn::engine eng = mkldnn::engine(mkldnn::engine::cpu, 0);

      auto mkldnn_dt_src = testutils::to_mkldnn_dtype(src->data_type());
      auto mkldnn_dt_wei = testutils::to_mkldnn_dtype(wei->data_type());
      auto mkldnn_dt_dst = testutils::to_mkldnn_dtype(dst->data_type());
      
      auto mkldnn_fmt_src = mkldnn::memory::format::nhwc;
      auto mkldnn_fmt_wei = mkldnn::memory::format::OIhw4i16o4i;
      if (pm.src_dims[1] % 16 != 0 || pm.conv_dst_dims[1] % 16 != 0)
          mkldnn_fmt_wei = mkldnn::memory::format::oihw;

      auto mkldnn_fmt_dst = mkldnn::memory::format::nhwc;
      auto mkldnn_fmt_bia = mkldnn::memory::format::x;
      
      /****************************** conv ***************************/
      // src memory preparation
      //mkldnn::memory mkldnn_src;
      //mkldnn::memory::primitive_desc src_pd;
      auto mkldnn_src_dims = testutils::to_mkldnn_dims(pm.src_dims);
      auto src_desc = mkldnn::memory::desc(mkldnn_src_dims, mkldnn_dt_src, mkldnn_fmt_src);
      auto src_pd = mkldnn::memory::primitive_desc(src_desc, eng);
      auto src_memory = mkldnn::memory(src_pd);
      utils::copy_array<data_t_src>((data_t_src*)(src_memory.get_data_handle()), 
                                    (data_t_src*)(src->data()), 
                                    src->size());

      // weight memory preparation
      memory::nchw_dims wei_dims = {pm.conv_dst_dims[1], pm.src_dims[1],
          pm.conv_kernel[0], pm.conv_kernel[1]};
      auto mkldnn_wei_dims = testutils::to_mkldnn_dims(wei_dims);
      auto wei_desc = mkldnn::memory::desc(mkldnn_wei_dims, mkldnn_dt_wei, mkldnn_fmt_wei);
      auto wei_pd = mkldnn::memory::primitive_desc(wei_desc, eng);
      auto wei_memory = mkldnn::memory(wei_pd);
      utils::copy_array<data_t_wei>((data_t_wei*)(wei_memory.get_data_handle()), 
                                    (data_t_wei*)(wei->data()), 
                                    wei->size());

      
      // bias memory preparation
      memory::dims bias_dims = {pm.conv_dst_dims[1]};
      auto mkldnn_bias_dims = testutils::to_mkldnn_dims(bias_dims);
      auto bias_desc = mkldnn::memory::desc(mkldnn_bias_dims, mkldnn_dt_dst, mkldnn_fmt_bia);
      auto bias_pd = mkldnn::memory::primitive_desc(bias_desc, eng);
      auto bias_memory = mkldnn::memory(bias_pd);
      utils::copy_array<data_t_dst>((data_t_dst*)(bias_memory.get_data_handle()), 
                                    (data_t_dst*)(bias->data()), 
                                    bias->size());

      // dst memory preparation
      //memory::nchw_dims conv_dst_dims;
      //conv_dst_dims[0] = pm.dst_dims[0];
      //conv_dst_dims[1] = pm.dst_dims[1];
      //int tail_h = (pm.src_dims[2] + 2 * pm.conv_pad[0] - pm.conv_kernel[0] ) % pm.conv_stride[0];
      //int tail_w = (pm.src_dims[3] + 2 * pm.conv_pad[1] - pm.conv_kernel[1] ) % pm.conv_stride[1];
      //if (tail_h != 0 || tail_w != 0)
      //    error_and_exit("bad input size and padding size");
      //conv_dst_dims[2] = (pm.src_dims[2] + 2 * pm.conv_pad[0] - pm.conv_kernel[0] ) / pm.conv_stride[0] + 1;
      //conv_dst_dims[3] = (pm.src_dims[3] + 2 * pm.conv_pad[1] - pm.conv_kernel[1] ) / pm.conv_stride[1] + 1;
     
      std::vector<int> padR = { pm.conv_pad[0]-1, pm.conv_pad[1]-1 };
      for (int i = 0; i < 4; ++i) {
          if ((pm.src_dims[2] + pm.conv_pad[0] + padR[0] - pm.conv_kernel[0] ) * 1.f / pm.conv_stride[0] + 1 < pm.conv_dst_dims[2]) ++padR[0];
          if ((pm.src_dims[3] + pm.conv_pad[1] + padR[1] - pm.conv_kernel[1] ) * 1.f / pm.conv_stride[1] + 1 < pm.conv_dst_dims[3]) ++padR[1]; 
      }

      auto mkldnn_dst_dims = testutils::to_mkldnn_dims(pm.conv_dst_dims);
      auto conv_dst_desc = mkldnn::memory::desc(mkldnn_dst_dims, mkldnn_dt_dst, mkldnn_fmt_dst);
      auto conv_dst_pd = mkldnn::memory::primitive_desc(conv_dst_desc, eng);
      auto conv_dst_memory = mkldnn::memory(conv_dst_pd);

      // conv_desc
      std::unique_ptr<mkldnn::convolution_forward::desc> convFwd_desc;
      convFwd_desc.reset(new mkldnn::convolution_forward::desc(
                  mkldnn::prop_kind::forward_scoring,
                  mkldnn::algorithm::convolution_direct, 
                  src_desc, wei_desc, bias_desc, conv_dst_desc, 
                  {pm.conv_stride[0], pm.conv_stride[1]},
                  {pm.conv_pad[0], pm.conv_pad[1]},
                  padR,
                  mkldnn::padding_kind::zero));

      //add shortcut sum for resnet
      mkldnn::primitive_attr attr;
      mkldnn::post_ops ops_;
      float scale = 1.f;
      ops_.append_sum(scale);
      attr.set_post_ops(ops_);
      /*     
      // add scale
      mkldnn::primitive_attr attr;
      int mask = 0;
      int count = pm.dst_dims[3];
      std::vector<float> scales(count);
      attr.set_output_scales(mask, scales);
      attr.set_int_output_round_mode(mkldnn::round_mode::round_nearest);

      // add relu
      mkldnn::post_ops ops;
      float scale = 1.f;
      float negative_slope = 0.f;
      float alpha = negative_slope; //negative slope for mkldnn_eltwise_relu.
      float beta = 0.f; //ignored for mkldnn_eltwise_relu.
      ops.append_eltwise(scale, mkldnn::algorithm::eltwise_relu, alpha, beta);
      attr.set_post_ops(ops);
      */

     // conv_primitive_desc
      std::unique_ptr<mkldnn::convolution_forward::primitive_desc> convFwd_pd;
      if (pm.with_eltwise_sum)
          convFwd_pd.reset(new mkldnn::convolution_forward::primitive_desc(*convFwd_desc, attr, eng));
      else
          convFwd_pd.reset(new mkldnn::convolution_forward::primitive_desc(*convFwd_desc, eng));


      // conv_prmitive
      std::unique_ptr<mkldnn::convolution_forward> convFwd;
      convFwd.reset(new mkldnn::convolution_forward(*convFwd_pd, 
                  src_memory, wei_memory, bias_memory, conv_dst_memory));

      std::vector<mkldnn::primitive> conv_relu_pool;
      conv_relu_pool.push_back(*convFwd);



      /******************************* relu *****************************/
      if (pm.with_relu){
          // relu
          std::unique_ptr<mkldnn::eltwise_forward::primitive_desc> reluFwd_pd;
          reluFwd_pd = testutils::get_mkldnn_relu_pd(conv_dst_desc, eng);
          std::unique_ptr<mkldnn::eltwise_forward> reluFwd;
          reluFwd.reset(new mkldnn::eltwise_forward(*reluFwd_pd, conv_dst_memory, conv_dst_memory));
          conv_relu_pool.push_back(*reluFwd);
      }


      /****************************** pooling ***************************/
      /*// check pooling dst
      memory::nchw_dims ref_dst_dims = pm.conv_dst_dims;
      int pool_tail_h = (pm.conv_dst_dims[2] + 2 * pm.pool_pad[0]) % pm.pool_stride[0];
      int pool_tail_w = (pm.conv_dst_dims[3] + 2 * pm.pool_pad[1]) % pm.pool_stride[1];
      if (pool_tail_h != 0 || pool_tail_w != 0)
          error_and_exit("bad input size and pooling_padding size");
      ref_dst_dims[2] = (pm.conv_dst_dims[2] + 2 * pm.pool_pad[0]) / pm.pool_stride[0];
      ref_dst_dims[3] = (pm.conv_dst_dims[3] + 2 * pm.pool_pad[1]) / pm.pool_stride[1];
      if (ref_dst_dims != pm.dst_dims) 
          error_and_exit("bad input size and pooling_padding size");
      */
      std::vector<int> pool_padR = { pm.pool_pad[0], pm.pool_pad[1] };
      for (int i = 0; i < 1; ++i) {
          if ((pm.conv_dst_dims[2] + pm.pool_pad[0] + pool_padR[0]) / pm.pool_stride[0] + 1 < pm.dst_dims[2]) ++pool_padR[0];
          if ((pm.conv_dst_dims[3] + pm.pool_pad[1] + pool_padR[1]) / pm.pool_stride[1] + 1 < pm.dst_dims[3]) ++pool_padR[1]; 
      }

      // pooling_dst memory preparation
      auto pool_mkldnn_dst_dims = testutils::to_mkldnn_dims(pm.dst_dims);
      auto pool_dst_desc = mkldnn::memory::desc(pool_mkldnn_dst_dims, mkldnn_dt_dst, mkldnn_fmt_dst);
      auto pool_dst_pd = mkldnn::memory::primitive_desc(pool_dst_desc, eng);
      auto pool_dst_memory = mkldnn::memory(pool_dst_pd);

      // pooling_desc
      mkldnn::algorithm pooling_algorithm = mkldnn::algorithm::pooling_max;
      if (pm.pooling_avg_include_padding)
          pooling_algorithm = mkldnn::algorithm::pooling_avg_include_padding;
      else if (pm.pooling_avg_exclude_padding)
          pooling_algorithm = mkldnn::algorithm::pooling_avg_exclude_padding;

      std::unique_ptr<mkldnn::pooling_forward::desc> poolFwd_desc;
      poolFwd_desc.reset(new mkldnn::pooling_forward::desc(
                  mkldnn::prop_kind::forward_inference,
                  pooling_algorithm,
                  conv_dst_desc,
                  pool_dst_desc,
                  {pm.pool_stride[0], pm.pool_stride[1]},
                  {pm.pool_kernel[0], pm.pool_kernel[1]},
                  {pm.pool_pad[0], pm.pool_pad[1]},
                  pool_padR,
                  mkldnn::padding_kind::zero));

      // pooling_primitive_desc
      std::unique_ptr<mkldnn::pooling_forward::primitive_desc> poolFwd_pd;
      poolFwd_pd.reset(new mkldnn::pooling_forward::primitive_desc(
                  *poolFwd_desc, eng));

      // pooling_primitive
      std::unique_ptr<mkldnn::pooling_forward> poolFwd;
      poolFwd.reset(new mkldnn::pooling_forward(*poolFwd_pd, conv_dst_memory, pool_dst_memory));
      conv_relu_pool.push_back(*poolFwd);

      // submit
      mkldnn::stream(mkldnn::stream::kind::eager).submit(conv_relu_pool).wait();
      
      
      /****************************** compare result ***************************/      
     /* // compare result
      data_t_dst* ref_data = (data_t_dst*)(pool_dst_memory.get_data_handle());
      data_t_dst* jit_data = (data_t_dst*)(dst->data());
      testutils::compare_array<data_t_dst>(jit_data, ref_data, dst->size());
  */
  }

protected:
    virtual void SetUp(){

        test_conv_relu_pool_params p = ::testing::TestWithParam<test_conv_relu_pool_params>::GetParam();
        auto dtype_src = utils::type2dtype<data_t_src>::dtype;
        auto dtype_wei = utils::type2dtype<data_t_wei>::dtype;
        auto dtype_acc = utils::type2dtype<data_t_acc>::dtype;
        auto dtype_dst = utils::type2dtype<data_t_dst>::dtype;
        
        std::unique_ptr<memory> src;
        src.reset(new memory(p.src_dims, format::nhwc, dtype_src));
        testutils::fill_data<data_t_src>(static_cast<data_t_src*>(src->data()), src->size());
     
        std::unique_ptr<memory> weight;
        memory::nchw_dims weight_dims = {p.dst_dims[1], p.src_dims[1], p.conv_kernel[0], p.conv_kernel[1]};
        weight.reset(new memory(weight_dims, format::OIhw4i16o4i, dtype_wei));
        testutils::fill_data<data_t_wei>(static_cast<data_t_wei*>(weight->data()), weight->size());

        std::unique_ptr<memory> bias;
        memory::dims bias_dims = {p.dst_dims[1]};
        bias.reset(new memory(bias_dims, format::x, dtype_dst));
        testutils::fill_data<data_t_dst>(static_cast<data_t_dst*>(bias->data()), bias->size());

        std::unique_ptr<memory> dst;
        dst.reset(new memory(p.dst_dims, format::nhwc, dtype_dst));

        // TODO:add conv_relu_fuse
       /* jit_avx512_core_u8s8s32x_convolution_relu_pool_op
          (const std::unique_ptr<memory> &conv_src,
           const std::unique_ptr<memory> &conv_wei,
           const std::unique_ptr<memory> &conv_bia,
           std::array<int, 2> conv_stride,
           std::array<int, 2> conv_padding,
           std::array<int, 2> conv_kernel,
           const std::unique_ptr<memory> &conv_dst,
           std::vector<float> conv_scales = {1.f},
           bool conv_relu = true,
           round_mode conv_round_mode = round_mode::nearest,
           const std::unique_ptr<memory>  &pool_src,
           const std::unique_ptr<memory>  &pool_dst,
           std::array<int, 2> pool_stride,
           std::array<int, 2> pool_padding,
           std::array<int, 2> pool_kernel,
           round_mode pool_round_mode = round_mode::nearest)*/
        check_result(p, src, weight, bias, dst);
     
    }
};
/*
struct test_conv_relu_pool_params {
   memory::nchw_dims src_dims;
   memory::pair_dims conv_kernel;
   memory::pair_dims conv_pad;
   memory::pair_dims conv_stride;
   memory::nchw_dims conv_dst_dims;

   bool with_eltwise_sum;
   bool with_relu;

   memory::pair_dims pool_kernel;
   memory::pair_dims pool_pad;
   memory::pair_dims pool_stride;
   memory::nchw_dims dst_dims;

   bool pooling_avg_include_padding;
   bool pooling_avg_exclude_padding;  
   bool max_pooling;
   };*/

using pooling_test_u8 = conv_relu_pooling_test<u8, s8, s32, u8>;
using pooling_test_s8 = conv_relu_pooling_test<u8, s8, s32, s8>;
using pooling_test_s32 = conv_relu_pooling_test<u8, s8, s32, s32>;

TEST_P(pooling_test_s32, TestsPooling) {}

INSTANTIATE_TEST_CASE_P(
        TestConvReluPooling_VGG_s32, pooling_test_s32, ::testing::Values(
          test_conv_relu_pool_params{ {1, 16, 4, 4}, {3, 3}, {0, 0}, {1, 1}, {1, 16, 2, 2}, 
          false, true, {2, 2}, {0, 0}, {2, 2}, {1, 16, 1, 1}, false, false, true},
          test_conv_relu_pool_params{ {1, 64, 224, 224}, {3, 3}, {1, 1}, {1, 1}, {1, 128, 224, 224},
          false, true, {2, 2}, {0, 0}, {2, 2}, {1, 128, 112, 112}, false, false, true },
          test_conv_relu_pool_params{ {1, 128, 112, 112}, {3, 3}, {1, 1}, {1, 1}, {1, 256, 112, 112},
          false, true, {2, 2}, {0, 0}, {2, 2}, {1, 256, 56, 56}, false, false, true },
          test_conv_relu_pool_params{ {1, 256, 56, 56}, {3, 3}, {1, 1}, {1, 1}, {1, 512, 56, 56},
          false, true, {2, 2}, {0, 0}, {2, 2}, {1, 512, 28, 28}, false, false, true },
          test_conv_relu_pool_params{ {1, 512, 28, 28}, {3, 3}, {1, 1}, {1, 1}, {1, 512, 28, 28},
          false, true, {2, 2}, {0, 0}, {2, 2}, {1, 512, 14, 14}, false, false, true },
          test_conv_relu_pool_params{ {1, 512, 14, 14}, {3, 3}, {1, 1}, {1, 1}, {1, 512, 14, 14},
          false, true, {2, 2}, {0, 0}, {2, 2}, {1, 512, 7, 7}, false, false, true }
        ));

INSTANTIATE_TEST_CASE_P(
        TestConvReluPooling_Resnet, pooling_test_s32, ::testing::Values(
          test_conv_relu_pool_params{ {50, 3, 224, 224}, {7, 7}, {3, 3}, {2, 2}, {50, 64, 112, 112},
          false, true, {2, 2}, {0, 0}, {2, 2}, {50, 64, 56, 56}, false, false, true },
          test_conv_relu_pool_params{ {50, 2048, 7, 7}, {1, 1}, {0, 0}, {1, 1}, {50, 2048, 7, 7},
          true, true, {7, 7}, {0, 0}, {7, 7}, {50, 2048, 1, 1}, false, true, false }
        ));

//TODO: add params in Mobilenet 


TEST_P(pooling_test_u8, TestsPooling) {}

INSTANTIATE_TEST_CASE_P(
        TestConvReluPooling_VGG_s32, pooling_test_u8, ::testing::Values(
          test_conv_relu_pool_params{ {1, 16, 4, 4}, {3, 3}, {0, 0}, {1, 1}, {1, 16, 2, 2}, 
          false, true, {2, 2}, {0, 0}, {2, 2}, {1, 16, 1, 1}, false, false, true},
          test_conv_relu_pool_params{ {1, 64, 224, 224}, {3, 3}, {1, 1}, {1, 1}, {1, 128, 224, 224},
          false, true, {2, 2}, {0, 0}, {2, 2}, {1, 128, 112, 112}, false, false, true },
          test_conv_relu_pool_params{ {1, 128, 112, 112}, {3, 3}, {1, 1}, {1, 1}, {1, 256, 112, 112},
          false, true, {2, 2}, {0, 0}, {2, 2}, {1, 256, 56, 56}, false, false, true },
          test_conv_relu_pool_params{ {1, 256, 56, 56}, {3, 3}, {1, 1}, {1, 1}, {1, 512, 56, 56},
          false, true, {2, 2}, {0, 0}, {2, 2}, {1, 512, 28, 28}, false, false, true },
          test_conv_relu_pool_params{ {1, 512, 28, 28}, {3, 3}, {1, 1}, {1, 1}, {1, 512, 28, 28},
          false, true, {2, 2}, {0, 0}, {2, 2}, {1, 512, 14, 14}, false, false, true },
          test_conv_relu_pool_params{ {1, 512, 14, 14}, {3, 3}, {1, 1}, {1, 1}, {1, 512, 14, 14},
          false, true, {2, 2}, {0, 0}, {2, 2}, {1, 512, 7, 7}, false, false, true }
        ));

INSTANTIATE_TEST_CASE_P(
        TestConvReluPooling_Resnet, pooling_test_u8, ::testing::Values(
          test_conv_relu_pool_params{ {50, 3, 224, 224}, {7, 7}, {3, 3}, {2, 2}, {50, 64, 112, 112},
          false, true, {2, 2}, {0, 0}, {2, 2}, {50, 64, 56, 56}, false, false, true },
          test_conv_relu_pool_params{ {50, 2048, 7, 7}, {1, 1}, {0, 0}, {1, 1}, {50, 2048, 7, 7},
          true, true, {7, 7}, {0, 0}, {7, 7}, {50, 2048, 1, 1}, false, true, false }
        ));


TEST_P(pooling_test_s8, TestsPooling) {}

INSTANTIATE_TEST_CASE_P(
        TestConvReluPooling_VGG_s32, pooling_test_s8, ::testing::Values(
          test_conv_relu_pool_params{ {1, 16, 4, 4}, {3, 3}, {0, 0}, {1, 1}, {1, 16, 2, 2}, 
          false, true, {2, 2}, {0, 0}, {2, 2}, {1, 16, 1, 1}, false, false, true},
          test_conv_relu_pool_params{ {1, 64, 224, 224}, {3, 3}, {1, 1}, {1, 1}, {1, 128, 224, 224},
          false, true, {2, 2}, {0, 0}, {2, 2}, {1, 128, 112, 112}, false, false, true },
          test_conv_relu_pool_params{ {1, 128, 112, 112}, {3, 3}, {1, 1}, {1, 1}, {1, 256, 112, 112},
          false, true, {2, 2}, {0, 0}, {2, 2}, {1, 256, 56, 56}, false, false, true },
          test_conv_relu_pool_params{ {1, 256, 56, 56}, {3, 3}, {1, 1}, {1, 1}, {1, 512, 56, 56},
          false, true, {2, 2}, {0, 0}, {2, 2}, {1, 512, 28, 28}, false, false, true },
          test_conv_relu_pool_params{ {1, 512, 28, 28}, {3, 3}, {1, 1}, {1, 1}, {1, 512, 28, 28},
          false, true, {2, 2}, {0, 0}, {2, 2}, {1, 512, 14, 14}, false, false, true },
          test_conv_relu_pool_params{ {1, 512, 14, 14}, {3, 3}, {1, 1}, {1, 1}, {1, 512, 14, 14},
          false, true, {2, 2}, {0, 0}, {2, 2}, {1, 512, 7, 7}, false, false, true }
        ));

INSTANTIATE_TEST_CASE_P(
        TestConvReluPooling_Resnet, pooling_test_s8, ::testing::Values(
          test_conv_relu_pool_params{ {50, 3, 224, 224}, {7, 7}, {3, 3}, {2, 2}, {50, 64, 112, 112},
          false, true, {2, 2}, {0, 0}, {2, 2}, {50, 64, 56, 56}, false, false, true },
          test_conv_relu_pool_params{ {50, 2048, 7, 7}, {1, 1}, {0, 0}, {1, 1}, {50, 2048, 7, 7},
          true, true, {7, 7}, {0, 0}, {7, 7}, {50, 2048, 1, 1}, false, true, false }
        ));

