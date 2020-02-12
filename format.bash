#!/bin/bash

clang-format -style=file -i ./cufhe/lib/*.cc
clang-format -style=file -i ./cufhe/lib/*.cu
clang-format -style=file -i ./cufhe/include/*.h
clang-format -style=file -i ./cufhe/include/*.cuh
clang-format -style=file -i ./cufhe/test/*.cc
clang-format -style=file -i ./cufhe/test/*.cu