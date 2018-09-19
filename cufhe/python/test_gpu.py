################################################################################
# Copyright 2018 Gizem S. Cetin <gscetin@wpi.edu>
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
################################################################################


# This test file uses "fhepy_gpu.so" library directly.
# All the function prototypes are exactly as in Cuda library.
# This gives more flexibility to the designer, if they want full
# control over Cuda streams.
# If you want a simple Python interface, check "test_circuits_gpu.py" out.

import lib.fhepy_gpu as fhe
import time
import random
import timeit

# Rand Seeds
random.seed()
fhe.SetSeed(int(time.time()))
kNumTests = 3000;

# Keys
pubkey = fhe.PubKey()
prikey = fhe.PriKey()
fhe.KeyGen(pubkey, prikey)
fhe.Initialize(pubkey)

# Plaintexts & Ciphertexts & Cuda Stream
ptxts1 = []
ptxts2 = []
ctxts1 = []
ctxts2 = []
st = []
for i in range(kNumTests):
	ptxts1.append(fhe.Ptxt())
	ptxts2.append(fhe.Ptxt())
	ptxts1[i].message = random.randint(0,1)
	ptxts2[i].message = random.randint(0,1)

	ctxts1.append(fhe.Ctxt())
	ctxts2.append(fhe.Ctxt())
	fhe.Encrypt(ctxts1[i], ptxts1[i], prikey)
	fhe.Encrypt(ctxts2[i], ptxts2[i], prikey)

	st.append(fhe.Stream())
	st[i].Create()

fhe.Synchronize()
start_time = timeit.default_timer()
for i in range(kNumTests):
	fhe.NAND(ctxts1[i], ctxts1[i], ctxts2[i], st[i])
fhe.Synchronize()
elapsed = timeit.default_timer() - start_time

passed = True
for i in range(kNumTests):
	ptxt = fhe.Ptxt()
	fhe.Decrypt(ptxt, ctxts1[i], prikey)
	if ptxt.message != (1 - ptxts1[i].message * ptxts2[i].message):
		passed = False
		break

if passed:
	print("\nPASS\n")
else:
	print("\nFAIL\n")
print(str(elapsed/kNumTests) + " sec / gate\n")

fhe.CleanUp()
for i in range(kNumTests):
	st[i].Destroy()
