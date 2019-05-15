#!/usr/bin/env bash

cd gpu/
./build_gpu.sh

cd ../cpu/
./build_cpu.sh

cd ../pytorch/
./build.sh
