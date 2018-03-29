# JIT(Just-In-Time) Inference library
JIT Inference library. \
This library will only work on AVX512 and above yet, maybe support AVX2 or below later.


## Compile
1. mkdir -p build && cd build
2. cmake ..
3. make -j `nproc`
4. make test
5. make install

It will download, compile and install all dependencies, including:
- [Xbyak](https://github.com/herumi/xbyak), used for JIT kernels.
- [Intel(R) MKL-DNN](https://github.com/intel/mkl-dnn), used for benchmark comparion and refernce in gtest.
- [Intel(R) MKLML](https://github.com/intel/mkl-dnn/releases/download/v0.13/mklml_lnx_2018.0.2.20180127.tgz), used for Intel OpenMP library.
- [gtest](https://github.com/google/googletest)

### Benchmark
`cmake -DWITH_BENCHMARK=ON`, default is enabled. \
Then can run `sh ./build/benchmark/bench_concat`

### How to profiling
`cmake -DWITH_VERBOSE=ON` and `export JITINFER_VERBOSE=1`

### MinSizeRel
This will only generate jitinfer library without any benchmark utilities and gtests. \
`cmake .. -DCMAKE_BUILD_TYPE=MinSizeRel`

## Operators
1. concat op do not use any VNNI yet.
2. conv fusion later will support VNNI.

## How to contribute
1. `pip install pre-commit`, about pre-commit check [here](http://pre-commit.com/#plugins).
2. `pre-commit install`. (use `pre-commit uninstall` to uninstall)
3. Then git commit ...
