#!/bin/bash

set -e

export KMP_AFFINITY="granularity=fine,compact,0,0" # when HT if OFF
# export KMP_AFFINITY="granularity=fine,compact,1,0" # when HT is ON

# export DEEPFUSION_VERBOSE=1
# export DEEPFUSION_DUMP_CODE=1
cd build
make -j `nproc`
cd ..

# 1 socket
# echo 0 > /proc/sys/kernel/numa_balancing
export OMP_NUM_THREADS=28
export MKL_NUM_THREADS=28
taskset -c 0-27 numactl -l ./build/test/test_conv_relu_pooling
#echo 1 > /proc/sys/kernel/numa_balancing

# 2 socket
# export OMP_NUM_THREADS=56
# export MKL_NUM_THREADS=56
# taskset -c 0-55 numactl -l ./build/benchmark/bench_concat
