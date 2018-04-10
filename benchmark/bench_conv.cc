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
#include <gflags/gflags.h>
#include <mkldnn.hpp>
#include <sstream>
#include "jitinfer.h"
#include "log.h"
#include "util_benchmark.h"
#include "util_mkldnn.h"

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
