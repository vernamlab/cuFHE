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
#include <memory>
#include <vector>
using namespace std;

const int gpuNum = 2;

void ConstantZeroCheck(Ptxt& out) { out.message_ = 0; }

void ConstantOneCheck(Ptxt& out) { out.message_ = 1; }

void NotCheck(Ptxt& out, const Ptxt& in)
{
    out.message_ = (~in.message_) & 0x1;
}

void CopyCheck(Ptxt& out, const Ptxt& in) { out.message_ = in.message_; }

void NandCheck(Ptxt& out, const Ptxt& in0, const Ptxt& in1)
{
    out.message_ = 1 - in0.message_ * in1.message_;
}

void OrCheck(Ptxt& out, const Ptxt& in0, const Ptxt& in1)
{
    out.message_ = (in0.message_ + in1.message_) > 0;
}

void OrYNCheck(Ptxt& out, const Ptxt& in0, const Ptxt& in1)
{
    out.message_ = (in0.message_ + (1 - in1.message_)) > 0;
}

void OrNYCheck(Ptxt& out, const Ptxt& in0, const Ptxt& in1)
{
    out.message_ = ((1 - in0.message_) + in1.message_) > 0;
}

void AndCheck(Ptxt& out, const Ptxt& in0, const Ptxt& in1)
{
    out.message_ = in0.message_ * in1.message_;
}

void AndYNCheck(Ptxt& out, const Ptxt& in0, const Ptxt& in1)
{
    out.message_ = in0.message_ * (1 - in1.message_);
}

void AndNYCheck(Ptxt& out, const Ptxt& in0, const Ptxt& in1)
{
    out.message_ = (1 - in0.message_) * in1.message_;
}

void XorCheck(Ptxt& out, const Ptxt& in0, const Ptxt& in1)
{
    out.message_ = (in0.message_ + in1.message_) & 0x1;
}

void XnorCheck(Ptxt& out, const Ptxt& in0, const Ptxt& in1)
{
    out.message_ = (~(in0.message_ ^ in1.message_)) & 0x1;
}

void MuxCheck(Ptxt& out, const Ptxt& inc, const Ptxt& in1, const Ptxt& in0)
{
    out.message_ = inc.message_ ? in1.message_ : in0.message_;
}

bool checkFinish(vector<shared_ptr<Stream>> sts)
{
    for (auto st : sts) {
        if (!StreamQuery(*st.get())) {
            return false;
        }
    }
    return true;
}

template <class Func, class Check>
void Test(string type, Func func, Check check, vector<shared_ptr<Ptxt>> pt,
          vector<shared_ptr<Ctxt>> ct, vector<shared_ptr<Stream>> st,
          int kNumTests, int kNumSMs, PriKey& pri_key)
{
    cout << "------ Test " << type << " Gate ------" << endl;
    cout << "Number of tests:\t" << kNumTests << endl;
    bool correct = true;
    int cnt_failures = 0;

    for (int i = 0; i < 4 * kNumTests; i++) {
        *pt[i].get() = rand() % Ptxt::kPtxtSpace;
        Encrypt(*ct[i].get(), *pt[i].get(), pri_key);
    }
    Synchronize();

    for (int i = 0; i < kNumTests; i++) {
        if constexpr (std::is_invocable_v<Func, Ctxt&, Stream>) {
            func(*ct[i].get(), *st[i % kNumSMs].get());
            check(*pt[i].get());
        }
        else if constexpr (std::is_invocable_v<Func, Ctxt&, const Ctxt&,
                                               Stream>) {
            func(*ct[i].get(), *ct[i + kNumTests].get(),
                 *st[i % kNumSMs].get());
            check(*pt[i].get(), *pt[i + kNumTests].get());
        }
        else if constexpr (std::is_invocable_v<Func, Ctxt&, const Ctxt&,
                                               const Ctxt&, Stream>) {
            func(*ct[i].get(), *ct[i + kNumTests].get(),
                 *ct[i + kNumTests * 2].get(), *st[i % kNumSMs].get());
            check(*pt[i].get(), *pt[i + kNumTests].get(),
                  *pt[i + kNumTests * 2].get());
        }
        else if constexpr (std::is_invocable_v<Func, Ctxt&, const Ctxt&,
                                               const Ctxt&, const Ctxt&,
                                               Stream>) {
            func(*ct[i].get(), *ct[i + kNumTests].get(),
                 *ct[i + kNumTests * 2].get(), *ct[i + kNumTests * 3].get(),
                 *st[i % kNumSMs].get());
            check(*pt[i].get(), *pt[i + kNumTests].get(),
                  *pt[i + kNumTests * 2].get(), *pt[i + kNumTests * 3].get());
        }
        else {
            std::cout << "Invalid Function" << std::endl;
        }
    }
    Synchronize();

    for (int i = 0; i < kNumTests; i++) {
        if constexpr (std::is_invocable_v<Func, Ctxt&, Stream>) {
            func(*ct[i + kNumTests * 3].get(),
                 *st[kNumSMs - (i % kNumSMs) - 1].get());
            check(*pt[i + kNumTests * 3].get());
        }
        else if constexpr (std::is_invocable_v<Func, Ctxt&, const Ctxt&,
                                               Stream>) {
            func(*ct[i + kNumTests * 3].get(), *ct[i].get(),
                 *st[kNumSMs - (i % kNumSMs) - 1].get());
            check(*pt[i + kNumTests * 3].get(), *pt[i].get());
        }
        else if constexpr (std::is_invocable_v<Func, Ctxt&, const Ctxt&,
                                               const Ctxt&, Stream>) {
            func(*ct[i + kNumTests * 3].get(), *ct[i].get(),
                 *ct[i + kNumTests].get(),
                 *st[kNumSMs - (i % kNumSMs) - 1].get());
            check(*pt[i + kNumTests * 3].get(), *pt[i].get(),
                  *pt[i + kNumTests].get());
        }
        else if constexpr (std::is_invocable_v<Func, Ctxt&, const Ctxt&,
                                               const Ctxt&, const Ctxt&,
                                               Stream>) {
            func(*ct[i + kNumTests * 3].get(), *ct[i + kNumTests * 2].get(),
                 *ct[i + kNumTests].get(), *ct[i].get(),
                 *st[kNumSMs - (i % kNumSMs) - 1].get());
            check(*pt[i + kNumTests * 3].get(), *pt[i + kNumTests * 2].get(),
                  *pt[i + kNumTests].get(), *pt[i].get());
        }
        else {
            std::cout << "Invalid Function" << std::endl;
        }
    }
    Synchronize();

    for (int i = 0; i < kNumTests; i++) {
        Ptxt res;
        Decrypt(res, *ct[i + kNumTests * 3].get(), pri_key);
        if (res.message_ != pt[i + kNumTests * 3].get()->message_) {
            correct = false;
            cnt_failures += 1;
            std::cout << type << " Fail at iteration: " << i << std::endl;
        }
    }
    if (correct)
        cout << "PASS" << endl;
    else
        cout << "FAIL:\t" << cnt_failures << "/" << kNumTests << endl;
}

int main()
{
    SetGPUNum(gpuNum);
    cudaSetDevice(0);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    uint32_t kNumSMs = prop.multiProcessorCount;
    uint32_t kNumTests = kNumSMs * 32;  // * 8;
    uint32_t kNumLevels = 10;           // Gate Types, Mux is counted as 2.

    SetSeed();  // set random seed

    PriKey pri_key;  // private key
    PubKey pub_key;  // public key
    // MUX Need 3 input
    vector<shared_ptr<Ctxt>> ct;
    vector<shared_ptr<Ptxt>> pt;
    for (int i = 0; i < 4 * kNumTests; i++) {
        ct.push_back(make_shared<Ctxt>());
        pt.push_back(make_shared<Ptxt>());
    }
    Synchronize();
    bool correct;

    cout << "------ Key Generation ------" << endl;
    KeyGen(pub_key, pri_key);

    cout << "------ Test Encryption/Decryption ------" << endl;
    cout << "Number of tests:\t" << kNumTests << endl;
    correct = true;
    for (int i = 0; i < kNumTests; i++) {
        pt[i].get()->message_ = rand() % Ptxt::kPtxtSpace;
        Encrypt(*ct[i].get(), *pt[i].get(), pri_key);
        Decrypt(*pt[kNumTests + i].get(), *ct[i].get(), pri_key);
        if (pt[kNumTests + i].get()->message_ != pt[i].get()->message_) {
            correct = false;
            break;
        }
    }
    if (correct)
        cout << "PASS" << endl;
    else
        cout << "FAIL" << endl;

    cout << "------ Initilizating Data on GPU(s) ------" << endl;
    Initialize(pub_key);  // essential for GPU computing

    vector<shared_ptr<Stream>> st;
    for (int i = 0; i < kNumSMs * gpuNum; i++) {
        st.push_back(make_shared<Stream>(i % gpuNum));
        st[i].get()->Create();
    }

    Test("NOT", Not, NotCheck, pt, ct, st, kNumTests, kNumSMs, pri_key);
    Test("COPY", Copy, CopyCheck, pt, ct, st, kNumTests, kNumSMs, pri_key);
    Test("NAND", Nand, NandCheck, pt, ct, st, kNumTests, kNumSMs, pri_key);
    Test("OR", Or, OrCheck, pt, ct, st, kNumTests, kNumSMs, pri_key);
    Test("ORYN", OrYN, OrYNCheck, pt, ct, st, kNumTests, kNumSMs, pri_key);
    Test("ORNY", OrNY, OrNYCheck, pt, ct, st, kNumTests, kNumSMs, pri_key);
    Test("AND", And, AndCheck, pt, ct, st, kNumTests, kNumSMs, pri_key);
    Test("ANDYN", AndYN, AndYNCheck, pt, ct, st, kNumTests, kNumSMs, pri_key);
    Test("ANDNY", AndNY, AndNYCheck, pt, ct, st, kNumTests, kNumSMs, pri_key);
    Test("XOR", Xor, XorCheck, pt, ct, st, kNumTests, kNumSMs, pri_key);
    Test("XNOR", Xnor, XnorCheck, pt, ct, st, kNumTests, kNumSMs, pri_key);
    Test("MUX", Mux, MuxCheck, pt, ct, st, kNumTests, kNumSMs, pri_key);
    Test("ConstantZero", ConstantZero, ConstantZeroCheck, pt, ct, st, kNumTests,
         kNumSMs, pri_key);
    Test("ConstantOne", ConstantOne, ConstantOneCheck, pt, ct, st, kNumTests,
         kNumSMs, pri_key);

    for (int i = 0; i < kNumSMs * gpuNum; i++) st[i].get()->Destroy();

    st.clear();

    cout << "------ Cleaning Data on GPU(s) ------" << endl;
    CleanUp();  // essential to clean and deallocate data
    ct.clear();
    pt.clear();
    return 0;
}
