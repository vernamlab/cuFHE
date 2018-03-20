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

// Include these two files for CPU computing.
#include <include/cufhe_cpu.h>
using namespace cufhe;

#include <iostream>
using namespace std;

void NandCheck(Ptxt& out, const Ptxt& in0, const Ptxt& in1) {
  out.message_ = 1 - in0.message_ * in1.message_;
}

int main() {
  uint32_t kNumTests = 1024;

  SetSeed();  // set random seed

  bool correct;
  PriKey pri_key_old;
  PubKey pub_key_old;
  Ptxt* pt = new Ptxt[2];
  Ctxt* ct = new Ctxt[2];

  cout<< "------ Key Generation ------" <<endl;
  KeyGen(pub_key_old, pri_key_old);
  // Alternatively ...
  // PriKeyGen(pri_key);
  // PubKeyGen(pub_key, pri_key);

  // This part shows how to write/read keys to/from files.
  // Same for Ctxt.
  PriKey pri_key;
  PubKey pub_key;
  WritePriKeyToFile(pri_key, "data/pri_key.txt");
  WritePubKeyToFile(pub_key, "data/pub_key.txt");
  ReadPriKeyFromFile(pri_key, "data/pri_key.txt");
  ReadPubKeyFromFile(pub_key, "data/pub_key.txt");

  cout<< "------ Test Encryption/Decryption ------" <<endl;
  cout<< "Number of tests:\t" << kNumTests <<endl;
  correct = true;
  for (int i = 0; i < kNumTests; i ++) {
    pt[0].message_ = rand() % Ptxt::kPtxtSpace;
    Encrypt(ct[0], pt[0], pri_key);
    Decrypt(pt[1], ct[0], pri_key);
    if (pt[1].message_ != pt[0].message_) {
      correct = false;
      break;
    }
  }
  if (correct)
    cout<< "PASS" <<endl;
  else
    cout<< "FAIL" <<endl;

  cout<< "------ Test NAND Gate ------" <<endl;
  kNumTests = 4;
  cout<< "Number of tests:\t" << kNumTests <<endl;
  correct = true;
  for (int i = 0; i < kNumTests; i ++) {
    pt[0].message_ = rand() % Ptxt::kPtxtSpace;
    pt[1].message_ = rand() % Ptxt::kPtxtSpace;
    Encrypt(ct[0], pt[0], pri_key);
    Encrypt(ct[1], pt[1], pri_key);
    Nand(ct[0], ct[0], ct[1], pub_key);
    NandCheck(pt[1], pt[0], pt[1]);
    Decrypt(pt[0], ct[0], pri_key);
    if (pt[1].message_ != pt[0].message_) {
      correct = false;
      break;
    }
  }
  if (correct)
    cout<< "PASS" <<endl;
  else
    cout<< "FAIL" <<endl;

  delete [] pt;
  delete [] ct;
  return 0;
}
