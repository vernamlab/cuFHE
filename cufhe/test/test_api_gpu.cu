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

// Include these two files for GPU computing.
#include <include/cufhe_gpu.cuh>
using namespace cufhe;

#include <iostream>
using namespace std;

void NandCheck(Ptxt& out, const Ptxt& in0, const Ptxt& in1) {
  out.message_ = 1 - in0.message_ * in1.message_;
}

void OrCheck(Ptxt& out, const Ptxt& in0, const Ptxt& in1) {
  out.message_ = (in0.message_ + in1.message_) > 0;
}

void AndCheck(Ptxt& out, const Ptxt& in0, const Ptxt& in1) {
  out.message_ = in0.message_ * in1.message_;
}

void XorCheck(Ptxt& out, const Ptxt& in0, const Ptxt& in1) {
  out.message_ = (in0.message_ + in1.message_) & 0x1;
}

int main() {
  cudaSetDevice(0);
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  uint32_t kNumSMs = prop.multiProcessorCount;
  uint32_t kNumTests = kNumSMs * 32;// * 8;
  uint32_t kNumLevels = 4;

  SetSeed(); // set random seed

  PriKey pri_key; // private key
  PubKey pub_key; // public key
  Ptxt* pt = new Ptxt[2 * kNumTests];
  Ctxt* ct = new Ctxt[2 * kNumTests];
  Synchronize();
  bool correct;

  cout<< "------ Key Generation ------" <<endl;
  KeyGen(pub_key, pri_key);
  // Alternatively ...
  // PriKeyGen(pri_key);
  // PubKeyGen(pub_key, pri_key);

  cout<< "------ Test Encryption/Decryption ------" <<endl;
  cout<< "Number of tests:\t" << kNumTests <<endl;
  correct = true;
  for (int i = 0; i < kNumTests; i ++) {
    pt[i].message_ = rand() % Ptxt::kPtxtSpace;
    Encrypt(ct[i], pt[i], pri_key);
    Decrypt(pt[kNumTests + i], ct[i], pri_key);
    if (pt[kNumTests + i].message_ != pt[i].message_) {
      correct = false;
      break;
    }
  }
  if (correct)
    cout<< "PASS" <<endl;
  else
    cout<< "FAIL" <<endl;

  cout<< "------ Initilizating Data on GPU(s) ------" <<endl;
  Initialize(pub_key); // essential for GPU computing

  cout<< "------ Test NAND Gate ------" <<endl;
  cout<< "Number of tests:\t" << kNumTests <<endl;
  // Create CUDA streams for parallel gates.
  Stream* st = new Stream[kNumSMs];
  for (int i = 0; i < kNumSMs; i ++)
    st[i].Create();

  correct = true;
  for (int i = 0; i < 2 * kNumTests; i ++) {
    pt[i] = rand() % Ptxt::kPtxtSpace;
    Encrypt(ct[i], pt[i], pri_key);
  }
  Synchronize();

  float et;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);

  // Here, pass streams to gates for parallel gates.
  for (int i = 0; i < kNumTests; i ++) {
    Nand(ct[i], ct[i], ct[i + kNumTests], st[i % kNumSMs]);
    Or(ct[i], ct[i], ct[i + kNumTests], st[i % kNumSMs]);
    And(ct[i], ct[i], ct[i + kNumTests], st[i % kNumSMs]);
    Xor(ct[i], ct[i], ct[i + kNumTests], st[i % kNumSMs]);
  }
  Synchronize();

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&et, start, stop);
  cout<< et / kNumTests / kNumLevels << " ms / gate" <<endl;
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  int cnt_failures = 0;
  for (int i = 0; i < kNumTests; i ++) {
    NandCheck(pt[i], pt[i], pt[i + kNumTests]);
    OrCheck(pt[i], pt[i], pt[i + kNumTests]);
    AndCheck(pt[i], pt[i], pt[i + kNumTests]);
    XorCheck(pt[i], pt[i], pt[i + kNumTests]);
    Decrypt(pt[i + kNumTests], ct[i], pri_key);
    if (pt[i + kNumTests].message_ != pt[i].message_) {
      correct = false;
      cnt_failures += 1;
      //std::cout<< "Fail at iteration: " << i <<std::endl;
    }
  }
  if (correct)
    cout<< "PASS" <<endl;
  else
    cout<< "FAIL:\t" << cnt_failures << "/" << kNumTests <<endl;
  for (int i = 0; i < kNumSMs; i ++)
    st[i].Destroy();
  delete [] st;

  cout<< "------ Cleaning Data on GPU(s) ------" <<endl;
  CleanUp(); // essential to clean and deallocate data
  delete [] ct;
  delete [] pt;
  return 0;
}
