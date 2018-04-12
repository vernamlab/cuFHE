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

import fhepy
import time
import random
import timeit

# Rand Seeds
random.seed()
pyfhe_gpu.SetSeed(int(time.time()))
kNumTests = 30;

# Keys
pubkey = pyfhe_gpu.PubKey()
prikey = pyfhe_gpu.PriKey()
pyfhe_gpu.KeyGen(pubkey, prikey)
pyfhe_gpu.Initialize(pubkey)

# Plaintexts & Ciphertexts & Cuda Stream
ptxts1 = []
ptxts2 = []
ctxts1 = []
ctxts2 = []
st = []
for i in range(kNumTests):
	ptxts1.append(pyfhe_gpu.Ptxt())
	ptxts2.append(pyfhe_gpu.Ptxt())
	ptxts1[i].message = random.randint(0,1)
	ptxts2[i].message = random.randint(0,1)

	ctxts1.append(pyfhe_gpu.Ctxt())
	ctxts2.append(pyfhe_gpu.Ctxt())
	pyfhe_gpu.Encrypt(ctxts1[i], ptxts1[i], prikey)
	pyfhe_gpu.Encrypt(ctxts2[i], ptxts2[i], prikey)

	st.append(pyfhe_gpu.Stream())
	st[i].Create()

pyfhe_gpu.Synchronize()
start_time = timeit.default_timer()
for i in range(kNumTests):
	pyfhe_gpu.NAND(ctxts1[i], ctxts1[i], ctxts2[i], st[i])
pyfhe_gpu.Synchronize()
elapsed = timeit.default_timer() - start_time

passed = True
for i in range(kNumTests):
	ptxt = pyfhe_gpu.Ptxt()
	pyfhe_gpu.Decrypt(ptxt, ctxts1[i], prikey)
	if ptxt.message != (1 - ptxts1[i].message * ptxts2[i].message):
		passed = False
		break
	#print(str(ptxts1[i].message) + " NAND " + str(ptxts2[i].message) +  " : " + str(ptxt.message))

if passed:
	print("\nPASS\n")
else:
	print("\nFAIL\n")
print(str(elapsed/kNumTests) + " sec / gate\n")

pyfhe_gpu.CleanUp()
for i in range(kNumTests):
	st[i].Destroy()
