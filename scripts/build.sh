#!/bin/bash
BUILD_MODE=$1

mkdir -p build
cd build
cmake -DCMAKE_BUILD_TYPE=$BUILD_MODE -GNinja .. && ninja -j 3
cd ..