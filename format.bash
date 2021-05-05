#!/bin/bash

clang-format -style=file -i ./src/*.cc
clang-format -style=file -i ./src/*.cu
clang-format -style=file -i ./include/*.h
clang-format -style=file -i ./include/*.cuh
clang-format -style=file -i ./test/*.cc
clang-format -style=file -i ./test/*.cu