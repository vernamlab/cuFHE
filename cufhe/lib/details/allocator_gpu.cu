/**
 * Copyright 2018 Wei Dai <wdai3141@gmail.com>
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#include <include/details/allocator_gpu.cuh>
#include <include/details/error_gpu.cuh>

namespace cufhe {

std::pair<void*, MemoryDeleter> AllocatorCPU::New(size_t nbytes) {
  void* ptr = nullptr;
  //CuSafeCall(cudaMallocHost(&ptr, nbytes));
  CuSafeCall(cudaHostAlloc(&ptr, nbytes, cudaHostAllocMapped));
  return {ptr, Delete};
}

void AllocatorCPU::Delete(void* ptr) { CuSafeCall(cudaFreeHost(ptr)); }

MemoryDeleter AllocatorCPU::GetDeleter() { return Delete; }


std::pair<void*, MemoryDeleter> AllocatorBoth::New(size_t nbytes) {
  void* ptr = nullptr;
  CuSafeCall(cudaMallocManaged(&ptr, nbytes));
  return {ptr, Delete};
}

void AllocatorBoth::Delete(void* ptr) { CuSafeCall(cudaFree(ptr)); }

MemoryDeleter AllocatorBoth::GetDeleter() { return Delete; }


std::pair<void*, MemoryDeleter> AllocatorGPU::New(size_t nbytes) {
  void* ptr = nullptr;
  CuSafeCall(cudaMalloc(&ptr, nbytes));
  return {ptr, Delete};
}

void AllocatorGPU::Delete(void* ptr) { CuSafeCall(cudaFree(ptr)); }

MemoryDeleter AllocatorGPU::GetDeleter() { return Delete; }

} // namespace cufhe
