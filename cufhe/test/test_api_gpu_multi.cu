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
#include <unistd.h>

#include <include/cufhe_gpu.cuh>
using namespace cufhe;

#include <iostream>
#include <memory>
#include <chrono>
using namespace std;

void NandCheck(Ptxt& out, const Ptxt& in0, const Ptxt& in1) {
  out.message_ = 1 - in0.message_ * in1.message_;
}

void OrCheck(Ptxt& out, const Ptxt& in0, const Ptxt& in1) {
  out.message_ = (in0.message_ + in1.message_) > 0;
}

void OrYNCheck(Ptxt& out, const Ptxt& in0, const Ptxt& in1) {
  out.message_ = (in0.message_ + (1-in1.message_)) > 0;
}

void OrNYCheck(Ptxt& out, const Ptxt& in0, const Ptxt& in1) {
  out.message_ = ((1-in0.message_) + in1.message_) > 0;
}

void AndCheck(Ptxt& out, const Ptxt& in0, const Ptxt& in1) {
  out.message_ = in0.message_ * in1.message_;
}

void AndYNCheck(Ptxt& out, const Ptxt& in0, const Ptxt& in1) {
  out.message_ = in0.message_ * (1-in1.message_);
}

void AndNYCheck(Ptxt& out, const Ptxt& in0, const Ptxt& in1) {
  out.message_ = (1-in0.message_) * in1.message_;
}

void XorCheck(Ptxt& out, const Ptxt& in0, const Ptxt& in1) {
  out.message_ = (in0.message_ + in1.message_) & 0x1;
}

void MuxCheck(Ptxt& out, const Ptxt& inc, const Ptxt& in1, const Ptxt& in0){
  out.message_ = inc.message_?in1.message_:in0.message_;
}

int main() {
  cudaSetDevice(0);
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  uint32_t kNumSMs = prop.multiProcessorCount;
  uint32_t kNumTests = kNumSMs * 64;// * 8;
  uint32_t kNumLevels = 10; 

  SetSeed(); // set random seed
  int gpuNum = 1;

  PriKey pri_key; // private key
  PubKey pub_key; // public key
  //MUX Need 3 input
  vector<shared_ptr<Ctxt>> ct;
  vector<shared_ptr<Ptxt>> pt;
  for(int i=0;i< 3*kNumTests;i++){
    ct.push_back(make_shared<Ctxt>(gpuNum));
    pt.push_back(make_shared<Ptxt>());
  }
  Synchronize(gpuNum);
  bool correct;

  cout<< "------ Key Generation ------" <<endl;
  KeyGen(pub_key, pri_key);

  cout<< "------ Test Encryption/Decryption ------" <<endl;
  cout<< "Number of tests:\t" << kNumTests <<endl;
  correct = true;
  for (int i = 0; i < kNumTests; i ++) {
    pt[i].get()->message_ = rand() % Ptxt::kPtxtSpace;
    Encrypt(*ct[i].get(), *pt[i].get(), pri_key);
    Decrypt(*pt[kNumTests + i].get(), *ct[i].get(), pri_key);
    if (pt[kNumTests + i].get()->message_ != pt[i].get()->message_) {
      correct = false;
      break;
    }
  }
  if (correct)
    cout<< "PASS" <<endl;
  else
    cout<< "FAIL" <<endl;

  cout<< "------ Initilizating Data on GPU(s) ------" <<endl;
  Initialize(pub_key, gpuNum); // essential for GPU computing

  cout << "Finished Initialize" << endl;

  // Create CUDA streams for parallel gates.
  cout << "------ Initializing Stream ------" << endl;
  vector<shared_ptr<Stream>> st;
  for (int i = 0; i < kNumSMs*gpuNum; i ++){
    st.push_back(make_shared<Stream>(i%gpuNum, 0));
    st[i]->Create();
  }

  cout<< "Number of tests:\t" << kNumTests <<endl;
  correct = true;
  for (int i = 0; i < 3* kNumTests; i ++) {
    *pt[i].get() = rand() % Ptxt::kPtxtSpace;
    Encrypt(*ct[i].get(), *pt[i].get(), pri_key);
  }
  Synchronize(gpuNum);

  chrono::system_clock::time_point start, end;
  start = chrono::system_clock::now();
  // Here, pass streams to gates for parallel gates.
  cout<< "------ Test NAND Gate ------" <<endl;
  for (int i = 0; i < kNumTests; i ++)
    mNand(*ct[i].get(), *ct[i].get(), *ct[i + kNumTests].get(), *st[i % (kNumSMs*gpuNum)].get());

  cout<< "------ Test OR Gate ------" <<endl;
  for (int i = 0; i < kNumTests; i ++)
    mOr(*ct[i].get(), *ct[i].get(), *ct[i + kNumTests].get(), *st[i % (kNumSMs*gpuNum)].get());

  cout<< "------ Test ORYN Gate ------" <<endl;
  for (int i = 0; i < kNumTests; i ++)
    mOrYN(*ct[i].get(), *ct[i].get(), *ct[i + kNumTests].get(), *st[i % (kNumSMs*gpuNum)].get());

  cout<< "------ Test ORNY Gate ------" <<endl;
  for (int i = 0; i < kNumTests; i ++)
    mOrNY(*ct[i].get(), *ct[i].get(), *ct[i + kNumTests].get(), *st[i % (kNumSMs*gpuNum)].get());

  cout<< "------ Test AND Gate ------" <<endl;
  for (int i = 0; i < kNumTests; i ++)
    mAnd(*ct[i].get(), *ct[i].get(), *ct[i + kNumTests].get(), *st[i % (kNumSMs*gpuNum)].get());

  cout<< "------ Test ANDYN Gate ------" <<endl;
  for (int i = 0; i < kNumTests; i ++)
    mAndYN(*ct[i].get(), *ct[i].get(), *ct[i + kNumTests].get(), *st[i % (kNumSMs*gpuNum)].get());

  cout<< "------ Test ANDNY Gate ------" <<endl;
  for (int i = 0; i < kNumTests; i ++)
    mAndNY(*ct[i].get(), *ct[i].get(), *ct[i + kNumTests].get(), *st[i % (kNumSMs*gpuNum)].get());

  cout<< "------ Test XOR Gate ------" <<endl;
  for (int i = 0; i < kNumTests; i ++)
    mXor(*ct[i].get(), *ct[i].get(), *ct[i + kNumTests].get(), *st[i % (kNumSMs*gpuNum)].get());

  cout<< "------ Test MUX Gate ------" <<endl;
  for (int i = 0; i < kNumTests; i ++)
    mMux(*ct[i].get(), *ct[i].get(), *ct[i + kNumTests].get(), *ct[i+ 2*kNumTests].get(),
	 *st[i % (kNumSMs*gpuNum)].get());

  Synchronize(gpuNum);
  end = chrono::system_clock::now();
  double elapsed = chrono::duration_cast<chrono::milliseconds>(end-start).count();

  cout<< elapsed / kNumTests / kNumLevels << " ms / gate" <<endl;

  int cnt_failures = 0;
  for (int i = 0; i < kNumTests; i ++) {
    NandCheck(*pt[i].get(), *pt[i].get(), *pt[i + kNumTests].get());
    OrCheck(*pt[i].get(), *pt[i].get(), *pt[i + kNumTests].get());
    OrYNCheck(*pt[i].get(), *pt[i].get(), *pt[i + kNumTests].get());
    OrNYCheck(*pt[i].get(), *pt[i].get(), *pt[i + kNumTests].get());
    AndCheck(*pt[i].get(), *pt[i].get(), *pt[i + kNumTests].get());
    AndYNCheck(*pt[i].get(), *pt[i].get(), *pt[i + kNumTests].get());
    AndNYCheck(*pt[i].get(), *pt[i].get(), *pt[i + kNumTests].get());
    XorCheck(*pt[i].get(), *pt[i].get(), *pt[i + kNumTests].get());
    MuxCheck(*pt[i].get(), *pt[i].get(), *pt[i + kNumTests].get(), *pt[i + 2*kNumTests].get());
    Decrypt(*pt[i + kNumTests].get(), *ct[i].get(), pri_key);
    if (pt[i + kNumTests].get()->message_ != pt[i].get()->message_) {
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
    st[i].get()->Destroy();
  st.clear();

  CleanUp(gpuNum);
  ct.clear();
  pt.clear();
  cout << "Deleted!" << endl;
}
