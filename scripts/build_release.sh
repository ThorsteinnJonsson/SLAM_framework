#!/bin/bash

mkdir -p build
cd build
cmake -DCMAKE_BUILD_TYPE=Release -GNinja .. && ninja -j 3
cd ..