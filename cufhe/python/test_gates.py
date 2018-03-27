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


random.seed()
pyfhe.SetSeed(int(time.time()))

pubkey = pyfhe.PubKey()
prikey = pyfhe.PriKey()
pyfhe.KeyGen(pubkey, prikey)

P1 = pyfhe.Ptxt()
P1.message = random.randint(0,1)
C1 = pyfhe.Ctxt()
pyfhe.Encrypt(C1, P1, prikey)

P2 = pyfhe.Ptxt()
P2.message = random.randint(0,1)
C2 = pyfhe.Ctxt()
pyfhe.Encrypt(C2, P2, prikey)


# AND
C = pyfhe.Ctxt()
P = pyfhe.Ptxt()
pyfhe.AND(C, C1, C2, pubkey)
pyfhe.Decrypt(P, C, prikey)
print(str(P1.message) + " AND " + str(P2.message) + " : " + str(P.message))


# NAND
C = pyfhe.Ctxt()
P = pyfhe.Ptxt()
pyfhe.NAND(C, C1, C2, pubkey)
pyfhe.Decrypt(P, C, prikey)
print(str(P1.message) + " NAND " + str(P2.message) + " : " + str(P.message))


# OR
C = pyfhe.Ctxt()
P = pyfhe.Ptxt()
pyfhe.OR(C, C1, C2, pubkey)
pyfhe.Decrypt(P, C, prikey)
print(str(P1.message) + " OR " + str(P2.message) + " : " + str(P.message))


# XOR
C = pyfhe.Ctxt()
P = pyfhe.Ptxt()
pyfhe.XOR(C, C1, C2, pubkey)
pyfhe.Decrypt(P, C, prikey)
print(str(P1.message) + " XOR " + str(P2.message) + " : " + str(P.message))
