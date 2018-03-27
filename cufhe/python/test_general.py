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

import pyfhe
import time
import random


# Rand Seeds
random.seed()
pyfhe.SetSeed(int(time.time()))


# If you want to see the default FHE parameters,
# you can use:
params = pyfhe.Param()
print(params.lwe_n)
print(params.tlwe_n)
# ...

# Keys
pubkey = pyfhe.PubKey()
prikey = pyfhe.PriKey()
pyfhe.KeyGen(pubkey, prikey)
# If you want to save the keys, you can store them
# in a file:
#pyfhe.WritePriKeyToFile(prikey, "prikey.txt")
#pyfhe.WritePubKeyToFile(pubkey, "pubkey.txt")
# You can also read them from a file
#pyfhe.ReadPubKeyFromFile(pubkey, "pubkey.txt")
#pyfhe.ReadPriKeyFromFile(prikey, "prikey.txt")


# Plaintext, default ptext space is 2
# If you want to change it, you can use:
# P.PtxtSpace = some_integer;
P = pyfhe.Ptxt()
P.message = random.randint(0,1)
print("Plaintext message : " + str(P.message))


# Ciphertext
C = pyfhe.Ctxt()


# Encrypt - Decrypt
pyfhe.Encrypt(C, P, prikey)
print("Writing computed ciphertext to file ./ctxt.txt")
pyfhe.WriteCtxtToFile(C, "ctxt.txt")
P_dec = pyfhe.Ptxt()
pyfhe.Decrypt(P_dec, C, prikey)
print("Decrypted message : " + str(P_dec.message))
