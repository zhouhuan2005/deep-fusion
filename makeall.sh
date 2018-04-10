#!/bin/bash

set -e

#rm -rf build
mkdir -p build && cd build
# debug cmake
#cmake .. -DCMAKE_BUILD_TYPE=DEBUG -DCMAKE_INSTALL_PREFIX=./tmp # -DWITH_VERBOSE=ON -DWITH_COLD_CACHE=ON

# release cmake
cmake .. -DCMAKE_INSTALL_PREFIX=./tmp # -DWITH_VERBOSE=ON -DWITH_DUMP_CODE=ON # -DWITH_COLD_CACHE=OFF

#cmake .. -DCMAKE_BUILD_TYPE=MinSizeRel

make clean
make -j `nproc`
make test
make install

cd -
