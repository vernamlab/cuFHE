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
#include <include/plain.h>
#include <test/test_util.h>
using namespace cufhe;

#include <iostream>
#include <memory>
#include <vector>
using namespace std;

const int gpuNum = 2;

int main()
{
    SetGPUNum(gpuNum);
    cudaSetDevice(0);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    const uint32_t kNumSMs = prop.multiProcessorCount*gpuNum*10;
    cout << "Number of streams per GPU:\t" << prop.multiProcessorCount << endl;
    const uint32_t kNumTests = kNumSMs*10;   // * 8;
    constexpr uint32_t kNumLevels = 10;  // Gate Types, Mux is counted as 2.

    TFHEpp::SecretKey* sk = new TFHEpp::SecretKey();
    TFHEpp::GateKeywoFFT* gk = new TFHEpp::GateKeywoFFT(*sk);

    cout << "n:" << sk->params.lvl0.n << endl;

    // MUX Need 3 input
    vector<uint8_t> pt(4 * kNumTests);
    vector<Ctxt> ct(4 * kNumTests);
    Synchronize();
    bool correct;

    cout << "Number of tests:\t" << kNumTests << endl;

    cout << "------ Initilizating Data on GPU(s) ------" << endl;
    Initialize(*gk);  // essential for GPU computing

    Stream* st = new Stream[kNumSMs*gpuNum];
    for (int i = 0; i < kNumSMs; i++) st[i].Create();

    Test("NAND", Nand, NandCheck, pt, ct, st, kNumTests, kNumSMs, *sk);
    Test("OR", Or, OrCheck, pt, ct, st, kNumTests, kNumSMs, *sk);
    Test("ORYN", OrYN, OrYNCheck, pt, ct, st, kNumTests, kNumSMs, *sk);
    Test("ORNY", OrNY, OrNYCheck, pt, ct, st, kNumTests, kNumSMs, *sk);
    Test("AND", And, AndCheck, pt, ct, st, kNumTests, kNumSMs, *sk);
    Test("ANDYN", AndYN, AndYNCheck, pt, ct, st, kNumTests, kNumSMs, *sk);
    Test("ANDNY", AndNY, AndNYCheck, pt, ct, st, kNumTests, kNumSMs, *sk);
    Test("XOR", Xor, XorCheck, pt, ct, st, kNumTests, kNumSMs, *sk);
    Test("XNOR", Xnor, XnorCheck, pt, ct, st, kNumTests, kNumSMs, *sk);
    Test("MUX", Mux, MuxCheck, pt, ct, st, kNumTests, kNumSMs, *sk);
    Test("NOT", Not, NotCheck, pt, ct, st, kNumTests, kNumSMs, *sk);
    Test("COPY", Copy, CopyCheck, pt, ct, st, kNumTests, kNumSMs, *sk);

    for (int i = 0; i < kNumSMs; i++) st[i].Destroy();
    delete[] st;

    cout << "------ Cleaning Data on GPU(s) ------" << endl;
    CleanUp();  // essential to clean and deallocate data
    return 0;
}
