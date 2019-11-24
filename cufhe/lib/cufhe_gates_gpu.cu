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
#include <include/cufhe_gpu.cuh>
#include <include/bootstrap_gpu.cuh>

namespace cufhe {

void Initialize(const PubKey& pub_key) {
  BootstrappingKeyToNTT(pub_key.bk_);
  KeySwitchingKeyToDevice(pub_key.ksk_);
}

void CleanUp() {
  DeleteBootstrappingKeyNTT();
  DeleteKeySwitchingKey();
}

inline void CtxtCopyH2D(const Ctxt& c, Stream st) {
  cudaMemcpyAsync(c.lwe_sample_device_->data(),
                  c.lwe_sample_->data(),
                  c.lwe_sample_->SizeData(),
                  cudaMemcpyHostToDevice,
                  st.st());
}

inline void CtxtCopyD2H(const Ctxt& c, Stream st) {
  cudaMemcpyAsync(c.lwe_sample_->data(),
                  c.lwe_sample_device_->data(),
                  c.lwe_sample_->SizeData(),
                  cudaMemcpyDeviceToHost,
                  st.st());
}

void Nand(Ctxt& out,
          const Ctxt& in0,
          const Ctxt& in1,
          Stream st) {
  static const Torus mu = ModSwitchToTorus(1, 8);
  static const Torus fix = ModSwitchToTorus(1, 8);
  CtxtCopyH2D(in0, st);
  CtxtCopyH2D(in1, st);
  NandBootstrap(out.lwe_sample_device_, in0.lwe_sample_device_,
      in1.lwe_sample_device_, mu, fix, st.st());
  CtxtCopyD2H(out, st);
}

void gNand(Ctxt& out,
          const Ctxt& in0,
          const Ctxt& in1,
          Stream st) {
  static const Torus mu = ModSwitchToTorus(1, 8);
  static const Torus fix = ModSwitchToTorus(1, 8);
  NandBootstrap(out.lwe_sample_device_, in0.lwe_sample_device_,
      in1.lwe_sample_device_, mu, fix, st.st());
}

void Or(Ctxt& out,
        const Ctxt& in0,
        const Ctxt& in1,
        Stream st) {
  static const Torus mu = ModSwitchToTorus(1, 8);
  static const Torus fix = ModSwitchToTorus(1, 8);
  CtxtCopyH2D(in0, st);
  CtxtCopyH2D(in1, st);
  OrBootstrap(out.lwe_sample_device_, in0.lwe_sample_device_,
      in1.lwe_sample_device_, mu, fix, st.st());
  CtxtCopyD2H(out, st);
}

void gOr(Ctxt& out,
        const Ctxt& in0,
        const Ctxt& in1,
        Stream st) {
  static const Torus mu = ModSwitchToTorus(1, 8);
  static const Torus fix = ModSwitchToTorus(1, 8);
  OrBootstrap(out.lwe_sample_device_, in0.lwe_sample_device_,
      in1.lwe_sample_device_, mu, fix, st.st());
}

void And(Ctxt& out,
         const Ctxt& in0,
         const Ctxt& in1,
         Stream st) {
  static const Torus mu = ModSwitchToTorus(1, 8);
  static const Torus fix = ModSwitchToTorus(-1, 8);
  CtxtCopyH2D(in0, st);
  CtxtCopyH2D(in1, st);
  AndBootstrap(out.lwe_sample_device_, in0.lwe_sample_device_,
      in1.lwe_sample_device_, mu, fix, st.st());
  CtxtCopyD2H(out, st);
}

void gAnd(Ctxt& out,
         const Ctxt& in0,
         const Ctxt& in1,
         Stream st) {
  static const Torus mu = ModSwitchToTorus(1, 8);
  static const Torus fix = ModSwitchToTorus(-1, 8);
  AndBootstrap(out.lwe_sample_device_, in0.lwe_sample_device_,
      in1.lwe_sample_device_, mu, fix, st.st());
}

void Nor(Ctxt& out,
         const Ctxt& in0,
         const Ctxt& in1,
         Stream st) {
  static const Torus mu = ModSwitchToTorus(1, 8);
  static const Torus fix = ModSwitchToTorus(-1, 8);
  CtxtCopyH2D(in0, st);
  CtxtCopyH2D(in1, st);
  NorBootstrap(out.lwe_sample_device_, in0.lwe_sample_device_,
      in1.lwe_sample_device_, mu, fix, st.st());
  CtxtCopyD2H(out, st);
}

void gNor(Ctxt& out,
         const Ctxt& in0,
         const Ctxt& in1,
         Stream st) {
  static const Torus mu = ModSwitchToTorus(1, 8);
  static const Torus fix = ModSwitchToTorus(-1, 8);
  NorBootstrap(out.lwe_sample_device_, in0.lwe_sample_device_,
      in1.lwe_sample_device_, mu, fix, st.st());
}

void Xor(Ctxt& out,
         const Ctxt& in0,
         const Ctxt& in1,
         Stream st) {
  static const Torus mu = ModSwitchToTorus(1, 8);
  static const Torus fix = ModSwitchToTorus(1, 4);
  CtxtCopyH2D(in0, st);
  CtxtCopyH2D(in1, st);
  XorBootstrap(out.lwe_sample_device_, in0.lwe_sample_device_,
      in1.lwe_sample_device_, mu, fix, st.st());
  CtxtCopyD2H(out, st);
}

void gXor(Ctxt& out,
         const Ctxt& in0,
         const Ctxt& in1,
         Stream st) {
  static const Torus mu = ModSwitchToTorus(1, 8);
  static const Torus fix = ModSwitchToTorus(1, 4);
  XorBootstrap(out.lwe_sample_device_, in0.lwe_sample_device_,
      in1.lwe_sample_device_, mu, fix, st.st());
}

void Xnor(Ctxt& out,
          const Ctxt& in0,
          const Ctxt& in1,
          Stream st) {
  static const Torus mu = ModSwitchToTorus(1, 8);
  static const Torus fix = ModSwitchToTorus(-1, 4);
  CtxtCopyH2D(in0, st);
  CtxtCopyH2D(in1, st);
  XnorBootstrap(out.lwe_sample_device_, in0.lwe_sample_device_,
      in1.lwe_sample_device_, mu, fix, st.st());
  CtxtCopyD2H(out, st);
}

void gXnor(Ctxt& out,
          const Ctxt& in0,
          const Ctxt& in1,
          Stream st) {
  static const Torus mu = ModSwitchToTorus(1, 8);
  static const Torus fix = ModSwitchToTorus(-1, 4);
  XnorBootstrap(out.lwe_sample_device_, in0.lwe_sample_device_,
      in1.lwe_sample_device_, mu, fix, st.st());
}

void Not(Ctxt& out,
         const Ctxt& in,
         Stream st) {
  CtxtCopyH2D(in, st);
  NotBootstrap(out.lwe_sample_device_, in.lwe_sample_device_, in.lwe_sample_->n(), st.st());
  CtxtCopyD2H(out, st);
}

void gNot(Ctxt& out,
         const Ctxt& in,
         Stream st) {
  NotBootstrap(out.lwe_sample_device_, in.lwe_sample_device_, in.lwe_sample_->n(), st.st());
}

void Copy(Ctxt& out,
          const Ctxt& in,
          Stream st) {
  for (int i = 0; i <= in.lwe_sample_->n(); i ++)
    out.lwe_sample_->data()[i] = in.lwe_sample_->data()[i];
}

void gCopy(Ctxt& out,
          const Ctxt& in,
          Stream st) {
  cudaMemcpyAsync(out.lwe_sample_device_->data(),
                  in.lwe_sample_device_->data(),
                  in.lwe_sample_->n(),
                  cudaMemcpyDeviceToDevice,
                  st.st());
}

void Mux(Ctxt& out,
         const Ctxt& inc,
         const Ctxt& in1,
         const Ctxt& in0,
         Stream st) {
  static const Torus mu = ModSwitchToTorus(1, 8);
  static const Torus fix = ModSwitchToTorus(-1, 8);
  CtxtCopyH2D(inc, st);
  CtxtCopyH2D(in1, st);
  CtxtCopyH2D(in0, st);
  MuxBootstrap(out.lwe_sample_device_, inc.lwe_sample_device_, in1.lwe_sample_device_,
      in0.lwe_sample_device_, mu, fix, st.st());
  CtxtCopyD2H(out, st);
}

void gMux(Ctxt& out,
         const Ctxt& inc,
         const Ctxt& in1,
         const Ctxt& in0,
         Stream st) {
  static const Torus mu = ModSwitchToTorus(1, 8);
  static const Torus fix = ModSwitchToTorus(-1, 8);
  MuxBootstrap(out.lwe_sample_device_, inc.lwe_sample_device_, in1.lwe_sample_device_,
      in0.lwe_sample_device_, mu, fix, st.st());
}

void SetToGPU(const Ctxt& in){
  cudaMemcpy(in.lwe_sample_device_->data(),
                  in.lwe_sample_->data(),
                  in.lwe_sample_->SizeData(),
                  cudaMemcpyHostToDevice);
}

void GetFromGPU(Ctxt& out){
  cudaMemcpy(out.lwe_sample_->data(),
                  out.lwe_sample_device_->data(),
                  out.lwe_sample_->SizeData(),
                  cudaMemcpyDeviceToHost);
}

} // namespace cufhe
