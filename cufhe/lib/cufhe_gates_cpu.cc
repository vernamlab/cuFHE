/**
 * Copyright 2013-2017 Wei Dai <wdai3141@gmail.com>
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
#include <include/bootstrap_cpu.h>

namespace cufhe {

void Initialize(const PubKey& pub_key) {}

void CleanUp() {}

void LWENoiselessTrivial(LWESample* ct, Torus mu) {
  for (int i = 0; i < ct->n(); i ++)
    ct->a()[i] = 0;
  ct->b() = mu;
}

void LWESubFrom(LWESample* res, const LWESample* sub) {
  for (int i = 0; i <= sub->n(); i ++)
    res->data()[i] -= sub->data()[i];
}

//void Initialize(PubKey pub_key);
//void And (Ctxt& out, const Ctxt& in0, const Ctxt& in1, const PubKey& pub_key);
//void Or  (Ctxt& out, const Ctxt& in0, const Ctxt& in1, const PubKey& pub_key);
//void Xor (Ctxt& out, const Ctxt& in0, const Ctxt& in1, const PubKey& pub_key);
void Nand(Ctxt& out,
          const Ctxt& in0,
          const Ctxt& in1,
          const PubKey& pub_key) {
  static const Torus mu = ModSwitchToTorus(1, 8);
  LWESample* temp = new LWESample(in0.lwe_sample_->n());
  std::pair<void*, MemoryDeleter> pair;
  pair = AllocatorCPU::New(temp->SizeMalloc());
  temp->set_data((LWESample::PointerType)pair.first);
  MemoryDeleter temp_deleter = pair.second;

  static const Torus nand_fix = ModSwitchToTorus(1, 8);
  LWENoiselessTrivial(temp, nand_fix);
  LWESubFrom(temp, in0.lwe_sample_);
  LWESubFrom(temp, in1.lwe_sample_);

  Bootstrap(out.lwe_sample_, temp, mu, pub_key.bk_, pub_key.ksk_);

  temp_deleter(temp->data());
  delete temp;
}

} // namespace cufhe
