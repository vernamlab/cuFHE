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
using namespace cufhe;

#include <iostream>
#include <random>
#include <vector>
using namespace std;

template <class Func, class Check>
void Test(string type, Func func, Check check, vector<uint8_t>& pt,
          vector<Ctxt>& ct, Stream* st, int kNumTests, int kNumSMs,
          TFHEpp::SecretKey& sk)
{
    cout << "------ Test " << type << " Gate ------" << endl;
    cout << "Number of tests:\t" << kNumTests << endl;
    bool correct = true;
    int cnt_failures = 0;

    random_device seed_gen;
    default_random_engine engine(seed_gen());
    uniform_int_distribution<uint32_t> binary(0, 1);
    for (int i = 0; i < 4 * kNumTests; i++) {
        pt[i] = binary(engine) > 0;
        ct[i].tlwehost = TFHEpp::tlweSymEncrypt<TFHEpp::lvl0param>(
            pt[i] ? TFHEpp::lvl0param::μ : -TFHEpp::lvl0param::μ,
            TFHEpp::lvl0param::α, sk.key.lvl0);
    }

    for (int i = 0; i < kNumTests; i++) {
        if constexpr (std::is_invocable_v<Func, Ctxt&>) {
            func(ct[i]);
            check(pt[i]);
        }
        else if constexpr (std::is_invocable_v<Func, Ctxt&, Ctxt&, Stream>) {
            func(ct[i], ct[i + kNumTests], st[i % kNumSMs]);
            check(pt[i], pt[i + kNumTests]);
        }
        else if constexpr (std::is_invocable_v<Func, Ctxt&, Ctxt&, Ctxt&,
                                               Stream>) {
            func(ct[i], ct[i + kNumTests], ct[i + kNumTests * 2],
                 st[i % kNumSMs]);
            check(pt[i], pt[i + kNumTests], pt[i + kNumTests * 2]);
        }
        else if constexpr (std::is_invocable_v<Func, Ctxt&, Ctxt&, Ctxt&, Ctxt&,
                                               Stream>) {
            func(ct[i], ct[i + kNumTests], ct[i + kNumTests * 2],
                 ct[i + kNumTests * 3], st[i % kNumSMs]);
            check(pt[i], pt[i + kNumTests], pt[i + kNumTests * 2],
                  pt[i + kNumTests * 3]);
        }
        else {
            std::cout << "Invalid Function" << std::endl;
        }
    }
    Synchronize();
    for (int i = 0; i < kNumTests; i++) {
        uint8_t res;
        res = TFHEpp::tlweSymDecrypt<TFHEpp::lvl0param>(ct[i].tlwehost,
                                                        sk.key.lvl0);
        if (res != pt[i]) {
            correct = false;
            cnt_failures += 1;
            // std::cout << type << " Fail at iteration: " << i << std::endl;
        }
    }
    if (correct)
        cout << "PASS" << endl;
    else
        cout << "FAIL:\t" << cnt_failures << "/" << kNumTests << endl;
}

int main()
{
    cudaSetDevice(0);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    const uint32_t kNumSMs = prop.multiProcessorCount;
    const uint32_t kNumTests = kNumSMs * 32;   // * 8;
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

    Stream* st = new Stream[kNumSMs];
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
