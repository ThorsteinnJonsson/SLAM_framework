#!/bin/bash

mkdir -p build
cd build
# cmake -DCMAKE_BUILD_TYPE=RelWithDebInfo .. && make
cmake -DCMAKE_BUILD_TYPE=RelWithDebInfo -GNinja .. && ninja -j 3
cd ..