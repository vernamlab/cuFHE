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

#include <include/cufhe.h>
#include <unistd.h>
#include <array>
#include <include/bootstrap_gpu.cuh>
#include <include/cufhe_gpu.cuh>

namespace cufhe {

int _gpuNum = 1;

int streamCount = 0;

void SetGPUNum(int gpuNum) { _gpuNum = gpuNum; }

void Initialize(const PubKey& pub_key)
{
    BootstrappingKeyToNTT(pub_key.bk_, _gpuNum);
    KeySwitchingKeyToDevice(pub_key.ksk_, _gpuNum);
}

void CleanUp()
{
    DeleteBootstrappingKeyNTT(_gpuNum);
    DeleteKeySwitchingKey(_gpuNum);
}

inline void CtxtCopyH2D(const Ctxt& c, Stream st)
{
    cudaSetDevice(st.device_id());
    cudaMemcpyAsync(c.lwe_sample_devices_[st.device_id()]->data(),
                    c.lwe_sample_->data(), c.lwe_sample_->SizeData(),
                    cudaMemcpyHostToDevice, st.st());
}

inline void CtxtCopyD2H(const Ctxt& c, Stream st)
{
    cudaSetDevice(st.device_id());
    cudaMemcpyAsync(c.lwe_sample_->data(),
                    c.lwe_sample_devices_[st.device_id()]->data(),
                    c.lwe_sample_->SizeData(), cudaMemcpyDeviceToHost, st.st());
}

void GateBootstrappingTLWE2TRLWElvl01NTT(cuFHETRLWElvl1& out, const Ctxt& in,
                                         Stream st)
{
    cudaSetDevice(st.device_id());
    static const Torus mu = ModSwitchToTorus(1, 8);
    CtxtCopyH2D(in, st);
    BootstrapTLWE2TRLWE(out.trlwedevices[st.device_id()],
                        in.lwe_sample_devices_[st.device_id()], mu, st.st(),
                        st.device_id());
    cudaMemcpyAsync(out.trlwehost.data(), out.trlwedevices[st.device_id()],
                    sizeof(out.trlwehost), cudaMemcpyDeviceToHost, st.st());
}

void SampleExtractAndKeySwitch(Ctxt& out, const cuFHETRLWElvl1& in, Stream st)
{
    cudaSetDevice(st.device_id());
    cudaMemcpyAsync(in.trlwedevices[st.device_id()], in.trlwehost.data(),
                    sizeof(in.trlwehost), cudaMemcpyHostToDevice, st.st());
    SEandKS(out.lwe_sample_devices_[st.device_id()],
            in.trlwedevices[st.device_id()], st.st(), st.device_id());
    CtxtCopyD2H(out, st);
}

void Nand(Ctxt& out, const Ctxt& in0, const Ctxt& in1, Stream st)
{
    cudaSetDevice(st.device_id());
    CtxtCopyH2D(in0, st);
    CtxtCopyH2D(in1, st);
    NandBootstrap(out.lwe_sample_devices_[st.device_id()],
                  in0.lwe_sample_devices_[st.device_id()],
                  in1.lwe_sample_devices_[st.device_id()], st.st(),
                  st.device_id());
    CtxtCopyD2H(out, st);
}

void gNand(Ctxt& out, const Ctxt& in0, const Ctxt& in1, Stream st)
{
    cudaSetDevice(st.device_id());
    NandBootstrap(out.lwe_sample_devices_[st.device_id()],
                  in0.lwe_sample_devices_[st.device_id()],
                  in1.lwe_sample_devices_[st.device_id()], st.st(),
                  st.device_id());
}

void Or(Ctxt& out, const Ctxt& in0, const Ctxt& in1, Stream st)
{
    cudaSetDevice(st.device_id());
    CtxtCopyH2D(in0, st);
    CtxtCopyH2D(in1, st);
    OrBootstrap(out.lwe_sample_devices_[st.device_id()],
                in0.lwe_sample_devices_[st.device_id()],
                in1.lwe_sample_devices_[st.device_id()], st.st(),
                st.device_id());
    CtxtCopyD2H(out, st);
}

void gOr(Ctxt& out, const Ctxt& in0, const Ctxt& in1, Stream st)
{
    cudaSetDevice(st.device_id());
    OrBootstrap(out.lwe_sample_devices_[st.device_id()],
                in0.lwe_sample_devices_[st.device_id()],
                in1.lwe_sample_devices_[st.device_id()], st.st(),
                st.device_id());
}

void OrYN(Ctxt& out, const Ctxt& in0, const Ctxt& in1, Stream st)
{
    cudaSetDevice(st.device_id());
    CtxtCopyH2D(in0, st);
    CtxtCopyH2D(in1, st);
    OrYNBootstrap(out.lwe_sample_devices_[st.device_id()],
                  in0.lwe_sample_devices_[st.device_id()],
                  in1.lwe_sample_devices_[st.device_id()], st.st(),
                  st.device_id());
    CtxtCopyD2H(out, st);
}

void gOrYN(Ctxt& out, const Ctxt& in0, const Ctxt& in1, Stream st)
{
    cudaSetDevice(st.device_id());
    OrYNBootstrap(out.lwe_sample_devices_[st.device_id()],
                  in0.lwe_sample_devices_[st.device_id()],
                  in1.lwe_sample_devices_[st.device_id()], st.st(),
                  st.device_id());
}

void OrNY(Ctxt& out, const Ctxt& in0, const Ctxt& in1, Stream st)
{
    cudaSetDevice(st.device_id());
    CtxtCopyH2D(in0, st);
    CtxtCopyH2D(in1, st);
    OrNYBootstrap(out.lwe_sample_devices_[st.device_id()],
                  in0.lwe_sample_devices_[st.device_id()],
                  in1.lwe_sample_devices_[st.device_id()], st.st(),
                  st.device_id());
    CtxtCopyD2H(out, st);
}

void gOrNY(Ctxt& out, const Ctxt& in0, const Ctxt& in1, Stream st)
{
    cudaSetDevice(st.device_id());
    OrNYBootstrap(out.lwe_sample_devices_[st.device_id()],
                  in0.lwe_sample_devices_[st.device_id()],
                  in1.lwe_sample_devices_[st.device_id()], st.st(),
                  st.device_id());
}

void And(Ctxt& out, const Ctxt& in0, const Ctxt& in1, Stream st)
{
    cudaSetDevice(st.device_id());
    CtxtCopyH2D(in0, st);
    CtxtCopyH2D(in1, st);
    AndBootstrap(out.lwe_sample_devices_[st.device_id()],
                 in0.lwe_sample_devices_[st.device_id()],
                 in1.lwe_sample_devices_[st.device_id()], st.st(),
                 st.device_id());
    CtxtCopyD2H(out, st);
}

void gAnd(Ctxt& out, const Ctxt& in0, const Ctxt& in1, Stream st)
{
    cudaSetDevice(st.device_id());
    AndBootstrap(out.lwe_sample_devices_[st.device_id()],
                 in0.lwe_sample_devices_[st.device_id()],
                 in1.lwe_sample_devices_[st.device_id()], st.st(),
                 st.device_id());
}

void AndYN(Ctxt& out, const Ctxt& in0, const Ctxt& in1, Stream st)
{
    cudaSetDevice(st.device_id());
    CtxtCopyH2D(in0, st);
    CtxtCopyH2D(in1, st);
    AndYNBootstrap(out.lwe_sample_devices_[st.device_id()],
                   in0.lwe_sample_devices_[st.device_id()],
                   in1.lwe_sample_devices_[st.device_id()], st.st(),
                   st.device_id());
    CtxtCopyD2H(out, st);
}

void gAndYN(Ctxt& out, const Ctxt& in0, const Ctxt& in1, Stream st)
{
    cudaSetDevice(st.device_id());
    AndYNBootstrap(out.lwe_sample_devices_[st.device_id()],
                   in0.lwe_sample_devices_[st.device_id()],
                   in1.lwe_sample_devices_[st.device_id()], st.st(),
                   st.device_id());
}

void AndNY(Ctxt& out, const Ctxt& in0, const Ctxt& in1, Stream st)
{
    cudaSetDevice(st.device_id());
    CtxtCopyH2D(in0, st);
    CtxtCopyH2D(in1, st);
    AndNYBootstrap(out.lwe_sample_devices_[st.device_id()],
                   in0.lwe_sample_devices_[st.device_id()],
                   in1.lwe_sample_devices_[st.device_id()], st.st(),
                   st.device_id());
    CtxtCopyD2H(out, st);
}

void gAndNY(Ctxt& out, const Ctxt& in0, const Ctxt& in1, Stream st)
{
    cudaSetDevice(st.device_id());
    AndNYBootstrap(out.lwe_sample_devices_[st.device_id()],
                   in0.lwe_sample_devices_[st.device_id()],
                   in1.lwe_sample_devices_[st.device_id()], st.st(),
                   st.device_id());
}

void Nor(Ctxt& out, const Ctxt& in0, const Ctxt& in1, Stream st)
{
    cudaSetDevice(st.device_id());
    CtxtCopyH2D(in0, st);
    CtxtCopyH2D(in1, st);
    NorBootstrap(out.lwe_sample_devices_[st.device_id()],
                 in0.lwe_sample_devices_[st.device_id()],
                 in1.lwe_sample_devices_[st.device_id()], st.st(),
                 st.device_id());
    CtxtCopyD2H(out, st);
}

void gNor(Ctxt& out, const Ctxt& in0, const Ctxt& in1, Stream st)
{
    cudaSetDevice(st.device_id());
    NorBootstrap(out.lwe_sample_devices_[st.device_id()],
                 in0.lwe_sample_devices_[st.device_id()],
                 in1.lwe_sample_devices_[st.device_id()], st.st(),
                 st.device_id());
}

void Xor(Ctxt& out, const Ctxt& in0, const Ctxt& in1, Stream st)
{
    cudaSetDevice(st.device_id());
    CtxtCopyH2D(in0, st);
    CtxtCopyH2D(in1, st);
    XorBootstrap(out.lwe_sample_devices_[st.device_id()],
                 in0.lwe_sample_devices_[st.device_id()],
                 in1.lwe_sample_devices_[st.device_id()], st.st(),
                 st.device_id());
    CtxtCopyD2H(out, st);
}

void gXor(Ctxt& out, const Ctxt& in0, const Ctxt& in1, Stream st)
{
    cudaSetDevice(st.device_id());
    XorBootstrap(out.lwe_sample_devices_[st.device_id()],
                 in0.lwe_sample_devices_[st.device_id()],
                 in1.lwe_sample_devices_[st.device_id()], st.st(),
                 st.device_id());
}

void Xnor(Ctxt& out, const Ctxt& in0, const Ctxt& in1, Stream st)
{
    cudaSetDevice(st.device_id());
    CtxtCopyH2D(in0, st);
    CtxtCopyH2D(in1, st);
    XnorBootstrap(out.lwe_sample_devices_[st.device_id()],
                  in0.lwe_sample_devices_[st.device_id()],
                  in1.lwe_sample_devices_[st.device_id()], st.st(),
                  st.device_id());
    CtxtCopyD2H(out, st);
}

void gXnor(Ctxt& out, const Ctxt& in0, const Ctxt& in1, Stream st)
{
    cudaSetDevice(st.device_id());
    XnorBootstrap(out.lwe_sample_devices_[st.device_id()],
                  in0.lwe_sample_devices_[st.device_id()],
                  in1.lwe_sample_devices_[st.device_id()], st.st(),
                  st.device_id());
}

void Not(Ctxt& out, const Ctxt& in, Stream st)
{
    cudaSetDevice(st.device_id());
    CtxtCopyH2D(in, st);
    NotBootstrap(out.lwe_sample_devices_[st.device_id()],
                 in.lwe_sample_devices_[st.device_id()], in.lwe_sample_->n(),
                 st.st(), st.device_id());
    CtxtCopyD2H(out, st);
}

void gNot(Ctxt& out, const Ctxt& in, Stream st)
{
    cudaSetDevice(st.device_id());
    NotBootstrap(out.lwe_sample_devices_[st.device_id()],
                 in.lwe_sample_devices_[st.device_id()], in.lwe_sample_->n(),
                 st.st(), st.device_id());
}

void Copy(Ctxt& out, const Ctxt& in, Stream st)
{
    cudaSetDevice(st.device_id());
    CtxtCopyH2D(in, st);
    CopyBootstrap(out.lwe_sample_devices_[st.device_id()],
                  in.lwe_sample_devices_[st.device_id()], st.st(),
                  st.device_id());
    CtxtCopyD2H(out, st);
}

void gCopy(Ctxt& out, const Ctxt& in, Stream st)
{
    cudaSetDevice(st.device_id());
    CopyBootstrap(out.lwe_sample_devices_[st.device_id()],
                  in.lwe_sample_devices_[st.device_id()], st.st(),
                  st.device_id());
}

void CopyOnHost(Ctxt& out, const Ctxt& in)
{
    for (int i = 0; i <= in.lwe_sample_->n(); i++) {
        out.lwe_sample_->data()[i] = in.lwe_sample_->data()[i];
    }
}

// Mux(inc,in1,in0) = inc?in1:in0 = inc&in1 + (!inc)&in0
void Mux(Ctxt& out, const Ctxt& inc, const Ctxt& in1, const Ctxt& in0,
         Stream st)
{
    cudaSetDevice(st.device_id());
    static const Torus mu = ModSwitchToTorus(1, 8);
    static const Torus fix = ModSwitchToTorus(-1, 8);
    static const Torus muxfix = ModSwitchToTorus(1, 8);
    CtxtCopyH2D(inc, st);
    CtxtCopyH2D(in1, st);
    CtxtCopyH2D(in0, st);
    MuxBootstrap(out.lwe_sample_devices_[st.device_id()],
                 inc.lwe_sample_devices_[st.device_id()],
                 in1.lwe_sample_devices_[st.device_id()],
                 in0.lwe_sample_devices_[st.device_id()], mu, fix, muxfix,
                 st.st(), st.device_id());
    CtxtCopyD2H(out, st);
}

void gMux(Ctxt& out, const Ctxt& inc, const Ctxt& in1, const Ctxt& in0,
          Stream st)
{
    cudaSetDevice(st.device_id());
    static const Torus mu = ModSwitchToTorus(1, 8);
    static const Torus fix = ModSwitchToTorus(-1, 8);
    static const Torus muxfix = ModSwitchToTorus(1, 8);
    MuxBootstrap(out.lwe_sample_devices_[st.device_id()],
                 inc.lwe_sample_devices_[st.device_id()],
                 in1.lwe_sample_devices_[st.device_id()],
                 in0.lwe_sample_devices_[st.device_id()], mu, fix, muxfix,
                 st.st(), st.device_id());
}

void ConstantZero(Ctxt& out)
{
    static const Torus mu = ModSwitchToTorus(1, 8);
    for (int i = 0; i < out.lwe_sample_->n(); i++) {
        out.lwe_sample_->data()[i] = 0;
    }
    out.lwe_sample_->data()[out.lwe_sample_->n()] = -mu;
}

void gConstantZero(Ctxt& out, Stream st)
{
    static const Torus mu = ModSwitchToTorus(1, 8);
    NoiselessTrivial(out.lwe_sample_device_, 0, mu, st.st());
}

void ConstantOne(Ctxt& out)
{
    static const Torus mu = ModSwitchToTorus(1, 8);
    for (int i = 0; i < out.lwe_sample_->n(); i++) {
        out.lwe_sample_->data()[i] = 0;
    }
    out.lwe_sample_->data()[out.lwe_sample_->n()] = mu;
}

void gConstantOne(Ctxt& out, Stream st)
{
    static const Torus mu = ModSwitchToTorus(1, 8);
    NoiselessTrivial(out.lwe_sample_device_, 1, mu, st.st());
}

void SetToGPU(const Ctxt& in)
{
    cudaMemcpy(in.lwe_sample_device_->data(), in.lwe_sample_->data(),
               in.lwe_sample_->SizeData(), cudaMemcpyHostToDevice);
}

void GetFromGPU(Ctxt& out)
{
    cudaMemcpy(out.lwe_sample_->data(), out.lwe_sample_device_->data(),
               out.lwe_sample_->SizeData(), cudaMemcpyDeviceToHost);
}

bool StreamQuery(Stream st)
{
    cudaSetDevice(st.device_id());
    cudaError_t res = cudaStreamQuery(st.st());
    if (res == cudaSuccess) {
        return true;
    }
    else {
        return false;
    }
}
}  // namespace cufhe
