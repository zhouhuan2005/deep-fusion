# Deep Fusion
a deep-fused inference optimization primitives lib on Intel Xeon E5 platforms, bulks of codes are borrowed from [MKL-DNN](https://github.com/intel/mkl-dnn).

## Build & Use
### How to Build
```shell
$ mkdir -p build && cd build
$ cmake ..
$ make -j `nproc`
$ make test
$ make install
```
It will download, compile and install all dependencies automatically, including:
- [Xbyak](https://github.com/herumi/xbyak), used for JIT kernels.
- [Intel(R) MKL-DNN](https://github.com/intel/mkl-dnn), used for benchmark comparion and gtest reference.
- [Intel(R) MKLML](https://github.com/intel/mkl-dnn/releases/download/v0.13/mklml_lnx_2018.0.2.20180127.tgz) for Intel OpenMP library.
- [gtest](https://github.com/google/googletest)
- [gflags](https://github.com/gflags/gflags)

### How to Benchmark
Add "-DWITH_BENCHMARK=ON" in cmake comamnd. Once build done, you can run with:
```shell
$ bash ./build/benchmark/bench_concat
```

### How to Profile
Add "-DWITH_VERBOSE=ON" in cmake comamnd, and export below env variable:
```shell
$ export DEEPFUSION_VERBOSE=1
```
The **WITH_VERBOSE** option is enabled in Debug and disabled in Release by default.

### How to Dump Code
Add "-DWITH_DUMP_CODE=ON" in cmake comamnd, and export below env variable:
```shell
$ export DEEPFUSION_DUMP_CODE=1
```
The **WITH_DUMP_CODE** option is enabled in Debug and disabled in Release by default.

Then, when run some apps, you can get some file like `jit_dump_jit_concat_kernel.0.bin`. You can use **xed** to check the ASM. For exapmle:
```
$ xed -ir jit_dump_jit_concat_kernel.0.bin
XDIS 0: PUSH      BASE       53                       push ebx
XDIS 1: PUSH      BASE       55                       push ebp
XDIS 2: BINARY    BASE       41                       inc ecx
XDIS 3: PUSH      BASE       54                       push esp
XDIS 4: BINARY    BASE       41                       inc ecx
XDIS 5: PUSH      BASE       55                       push ebp
XDIS 6: BINARY    BASE       41                       inc ecx
XDIS 7: PUSH      BASE       56                       push esi
XDIS 8: BINARY    BASE       41                       inc ecx
XDIS 9: PUSH      BASE       57                       push edi
```

### Generate MinSizeRel
This will only generate deepfusion library without any benchmark utilities and gtests. 
``` shell
cmake .. -DCMAKE_BUILD_TYPE=MinSizeRel
```

## Supported Operators
 - [x] concat+relu fused op (u8/s8/s32/f32|AVX, AVX2 and AVX512)
 - [x] conv3x3+relu+conv1x1+relu fused op
   - supported multi channel scales
   - supported various data type
| Memory | Supported Data Type |
|---|--- |
| src | u8 |
| weight | s8 |
| bias | u8/s8/s32/f32 |
| dst | u8/s8/s32/f32 |

## Docker
Docker images is provied for compiling and debuging.
 - `docker pull tensortang/ubuntu` for gcc8.0, gdb and some necessary env.
 - `docker pull tensortang/ubuntu:16.04` for gcc5.4
