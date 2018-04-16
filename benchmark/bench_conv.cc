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

#include <gflags/gflags.h>
#include <sstream>
#include "log.h"
#include "test_utils.h"

DEFINE_int32(burning_iter, 50, "Burning iterations");
DEFINE_int32(iter, 100, "Iterations for average");
DEFINE_int32(bs, 0, "Batch size, number of images");
DEFINE_int32(ih, 0, "Input image height");
DEFINE_int32(iw, 0, "Input image width");
DEFINE_int32(kh, 0, "Kernel height");
DEFINE_int32(kw, 0, "Kernel width");
DEFINE_int32(sh, 0, "Stride height");
DEFINE_int32(sw, 0, "Stride width");
DEFINE_int32(ph, 0, "Padding height");
DEFINE_int32(pw, 0, "Padding width");
DEFINE_int32(ic, 0, "Input channels of first conv");
DEFINE_int32(oc, 0, "Output channels of first conv");
DEFINE_int32(oc1x1, 0, "Output channels of 1x1 conv");
DEFINE_string(dtype, "s8", "Data type");
DEFINE_bool(post_relu, true, "Post ReLU after Conv");

static mkldnn::engine eng = mkldnn::engine(mkldnn::engine::cpu, 0);

int main(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  return 0;
}
